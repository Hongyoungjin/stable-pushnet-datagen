# import python modules
import time
import os
import yaml

# import isaacgym modules
from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import *

# import pytorch modules
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import sys
import os
  
current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
  
sys.path.append(parent_directory)

import argparse

# import 3rd party modules
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# import local modules
from data_generator.utils.contact_point_sampler import ContactPoint, ContactPointSampler
from data_generator.utils.crop_image_parallel import crop_image
from utils.push_dof_tools import *
from stable_pushnet.model import PushNet

import pickle


class PushSim(object):
    def __init__(self):
        
        # parse arguments
        parser = argparse.ArgumentParser(description="Push Sim: Push simulation of tableware for stable pushing network training data generation")
        parser.add_argument('--config', type=str, default="/home/hong/ws/twc-stable-pushnet/config/config_pushsim.yaml", help='Configuration file')
        parser.add_argument('--headless', type=bool, default=False, help='Turn on the viewer')
        parser.add_argument('--analytical', type=bool, default=False, help='Whether to use analytical stable region or not')
        parser.add_argument('--use_gpu_pipeline', type=bool, default=False, help='Whether to use gpu pipeline')
        # parser.add_argument('-d','--slider_dataset', type=str, default='dish_urdf', help='slider dataset name')
        parser.add_argument('-n','--slider_name', type=str, default='glass', help='slider name')
        self.args = parser.parse_args()
        
        # set default push velocities
        self.num_samples = 1000
        self.pushing_directions = fibonacci_sphere(self.num_samples * 2)
        self.icrs = velocities2icrs(self.pushing_directions)
        
        # Set pushing labels
        self.labels = []
        
        # Set push_contact
        self.push_contact = None
        
        # initialize gym
        self.gym = gymapi.acquire_gym()

        self._load_configuration()
        self._create_sim()
        self.gym.prepare_sim(self.sim) # Prepare simulation with buffer allocations
        self._create_ground()
        self._create_viewer()
        self._create_envs()
        
    def _load_configuration(self):
        ''' Configurate the entire simulation by conveying config data from configuration file '''
        config_file = self.args.config
        self.headless = self.args.headless
        self.analytical = self.args.analytical
        self.slider_name = self.args.slider_name
        
        # Load configuation file
        with open(config_file,'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        
        sim_cfg = cfg["simulation"]
        cam_cfg = sim_cfg["camera"]["ZividTwo"]
        
         
        
        # simulation setup (default)
        self.FILE_ZERO_PADDING_NUM = sim_cfg["FILE_ZERO_PADDING_NUM"]
        self.physics_engine = sim_cfg["physics_engine"]
        self.num_threads = sim_cfg["num_threads"]
        self.compute_device_id = sim_cfg["compute_device_id"]
        self.graphics_device_id = sim_cfg["graphics_device_id"]
        self.num_envs = 125
        self.use_gpu = sim_cfg['use_gpu']
        self.use_gpu_pipeline = sim_cfg['use_gpu_pipeline']
        self.dt = sim_cfg["dt"]
        self.render_freq = 1/sim_cfg["render_freq"]
        self.num_iters = 8
        
        # Camera configuration
        fx, fy, cx, cy = cam_cfg["fx"], cam_cfg["fy"], cam_cfg["cx"], cam_cfg["cy"]
        self.camera_intrinsic = np.array([[fx,0,cx],
                                          [0,fy,cy],
                                          [0, 0, 1]])
        self.camera_rand_position_range = sim_cfg["camera_rand_position_range"]
        self.camera_rand_rotation_range = sim_cfg["camera_rand_rotation_range"]
        
        self.camera_poses = None # Camera poses will be set in reset_camera_poses()
        
        # Asset name setup
        self.slider_dataset_name = sim_cfg["slider_dataset"]
        
        self.pusher_dataset_name = sim_cfg["pusher_dataset"]
        self.pusher_name = sim_cfg["pusher_name"]
        
        # Pushing velocity
        self.push_speed = sim_cfg["push_speed"]
        
        # Pusher friction coefficient
        # self.pusher_friction_coefficient = sim_cfg['pusher_friction_coefficient']
        self.pusher_friction_coefficient = sim_cfg['friction_coefficient']
        self.friction_coefficient = sim_cfg['friction_coefficient']
        
        # Initial approach distance of the pusher to the slider
        self.translational_push_distance = sim_cfg["translational_push_distance"]
        
        # Initial distance between pusher and the slider
        self.initial_distance = sim_cfg["initial_distance"]
        
        # Gripper width
        self.gripper_width = sim_cfg["gripper_width"]
        
        # Slider's stable pose
        stable_pose = np.load("assets/{}/{}/stable_poses.npy".format(self.slider_dataset_name, self.slider_name))
        quat = R.from_matrix(stable_pose[:3,:3]).as_quat()
        self.slider_stable_pose = gymapi.Transform()
        self.slider_stable_pose.p = gymapi.Vec3(0.5, 0.1, stable_pose[2,3])
        self.slider_stable_pose.r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
        self.slider_default_pose = stable_pose
        # Random slider pose
        self.slider_rand_position_range = sim_cfg["slider_rand_position_range"]
        self.slider_rand_rotation_range = sim_cfg["slider_rand_rotation_range"]
        
        # Labeling threshold
        self.threshold_pos = sim_cfg["threshold_pos"]
        self.threshold_rot = sim_cfg["threshold_rot"]
        
        # Save directories
        self.save_dir = "../data/tensors"
        os.makedirs(os.path.join(self.save_dir), exist_ok=True)
        # Set the starting file index for saving the results
        self.init_file_idx = get_maximum_file_idx(self.save_dir)
        
        # statistical data for normalization
        self.image_mean = np.load(os.path.join("../data/", 'image_mean.npy'))
        self.image_std  = np.load(os.path.join("../data/", 'image_std.npy'))
        
        self.masked_image_mean = np.load(os.path.join("../data/", 'masked_image_mean.npy'))
        self.masked_image_std  = np.load(os.path.join("../data/", 'masked_image_std.npy'))
        
        self.velocity_mean = np.load(os.path.join("../data/", 'velocity_mean.npy'))
        self.velocity_std  = np.load(os.path.join("../data/", 'velocity_std.npy'))
        
    def _create_sim(self):
        """ Create the simulation """
        # Configure Sim Params
        sim_params = gymapi.SimParams()
        if self.physics_engine == "PHYSX":
            physics_engine = gymapi.SIM_PHYSX
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 6
            sim_params.physx.num_velocity_iterations = 0
            sim_params.physx.num_threads = self.num_threads
            sim_params.physx.use_gpu = self.use_gpu
            sim_params.use_gpu_pipeline = self.use_gpu_pipeline
            
        if self.physics_engine == "FLEX":
            physics_engine = gymapi.SIM_FLEX
            sim_params.flex.shape_collision_margin = 0.25
            sim_params.flex.num_outer_iterations = 4
            sim_params.flex.num_inner_iterations = 10
            
        # Set GPU pipeline
        if self.args.use_gpu_pipeline:
            self.device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        # Set up axis as Z-up
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        
        # Create sim
        self.sim = self.gym.create_sim(self.compute_device_id, self.graphics_device_id, physics_engine, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

    def _create_ground(self):
        '''Create ground'''
        # Configure ground parameters
        plane_params = gymapi.PlaneParams()
        plane_params.static_friction = 0.1
        plane_params.dynamic_friction = 0.1
        plane_params.restitution = 0
        
        # Set up axis as Z-up
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)

        # Create ground
        self.gym.add_ground(self.sim, plane_params)
        
    def _create_viewer(self):
        ''' Create viewer '''
        
        if self.headless == True:
            self.viewer = None
            
        else:
            # Set viewer
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            # position the camera
            # cam_pos = gymapi.Vec3(3, 5, 3)
            # cam_target = gymapi.Vec3(3, 5, 0)
            
            cam_pos = gymapi.Vec3(5.5, 2.001, 7)
            cam_target = gymapi.Vec3(5.5, 2, 0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            
            # key callback
            self.gym.subscribe_viewer_keyboard_event(self.viewer,gymapi.KEY_ESCAPE,"QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer,gymapi.KEY_V,"toggle_viewer_sync")

            if self.viewer is None:
                print("*** Failed to create viewer")
                quit()
    
    def _create_envs(self):
        ''' Create environments with pusher, slider, and pusher-attached camera'''
        
        # Load assets
        slider_asset, pusher_asset = self.load_assets()
        
        # Set default slider pose
        default_slider_pose = self.slider_stable_pose
        
        # Set pusher pose
        pusher_pose = gymapi.Transform()
        pusher_pose.p = gymapi.Vec3(0, 0, -1)
        pusher_pose.r = gymapi.Quat(0,0,0,1)
        
        #######
        # Env #
        #######
            
        # set up the env grid
        num_per_row = int(np.ceil(np.sqrt(self.num_envs)))
        spacing = 0.5
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        print(f'Creating {self.num_envs} environments')
        # cache useful handles
        self.envs = []
        self.slider_actor_handles = []
        self.pusher_actor_handles = []
        self.camera_handles = []
        
        for env_idx in range(self.num_envs):
            # Create environment
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            
            # Create actors
            slider_name = f"slider_{env_idx}"
            pusher_name = f"pusher_{env_idx}"
            slider_actor_handle = self.gym.create_actor(env, slider_asset, default_slider_pose, slider_name, env_idx, segmentationId=0)
            pusher_actor_handle = self.gym.create_actor(env, pusher_asset, pusher_pose, pusher_name, env_idx, segmentationId=0)
            
            # set visual property
            self.gym.set_rigid_body_color(env, slider_actor_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0., 1., 0.))
            # Color left finger
            self.gym.set_rigid_body_color(env, pusher_actor_handle, 4, gymapi.MESH_VISUAL, gymapi.Vec3(1., 0., 1.))
            # Color right finger
            self.gym.set_rigid_body_color(env, pusher_actor_handle, 5, gymapi.MESH_VISUAL, gymapi.Vec3(0., 0., 1.))
            
            if env_idx == 0:
                # Initialize pusher actor DOF states
                self.default_pusher_dof_state, pusher_dof_props = self.initialize_pusher_dof(env, pusher_actor_handle)
                
            # Set actor segmentation IDs
            self.gym.set_rigid_body_segmentation_id(env, slider_actor_handle, 0, 1)
            self.gym.set_rigid_body_segmentation_id(env, pusher_actor_handle, 0, 2)
            # Create hand-eye camera sensor and attach to the pusher eef
            # self.create_hand_eye_camera(env, pusher_actor_handle)
            
            # Create a camera sensor for the first env
            camera_handle = self.create_camera(env)
            
            # Set slider friction coefficient randomly
            # self.set_slider_friction_cofficient(env, slider_actor_handle, self.friction_cofefficient)
            self.set_slider_friction_cofficient(env, slider_actor_handle, self.friction_coefficient)
            
            # Set pusher friction coefficient in a constant value
            # self.set_pusher_friction_cofficient(env, pusher_actor_handle, self.pusher_friction_coefficient)
            self.set_pusher_friction_cofficient(env, pusher_actor_handle, self.pusher_friction_coefficient)
            
            # set pusher dof properties
            self.gym.set_actor_dof_properties(env, pusher_actor_handle, pusher_dof_props)
            
            # Set pusher DOF positions
            self.gym.set_actor_dof_states(env, pusher_actor_handle, self.default_pusher_dof_state, gymapi.STATE_ALL)
            
            # Store envs and handles
            self.envs.append(env)
            self.slider_actor_handles.append(slider_actor_handle)
            self.pusher_actor_handles.append(pusher_actor_handle)
            self.camera_handles.append(camera_handle)        
            
    def __del__(self):
        if not self.args.headless:
            self.gym.destroy_viewer(self.viewer)
            self.gym.destroy_sim(self.sim)
            
    def load_assets(self):
        # TODO: load all assets (cup to bowl) and create new actors with different assets every step
        asset_root = "./assets"
        
        # slider_names = ["melamineware_g_0151", "melamineware_g_0130", "dish_small", "rice_bowl", "soup_bowl"]
        # self.slider_name = slider_names[slider_id]
        
        # Load slider asset
        slider_asset_file = "{}/{}/{}.urdf".format(
            self.slider_dataset_name,
            self.slider_name,
            self.slider_name)

        slider_asset_options = gymapi.AssetOptions()
        slider_asset_options.armature = 0.001
        slider_asset_options.fix_base_link = False
        slider_asset_options.thickness = 0.001
        slider_asset_options.override_inertia = True
        slider_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        slider_asset_options.vhacd_enabled = True
        slider_asset_options.vhacd_params.resolution = 300000
        slider_asset_options.vhacd_params.max_convex_hulls = 50
        slider_asset_options.vhacd_params.max_num_vertices_per_ch = 1000
        slider_asset_options.density = 2739
        slider_asset_options.vhacd_enabled = True

        print("Loading asset '%s' from '%s'" % (slider_asset_file, asset_root))
        slider_asset = self.gym.load_asset(self.sim, asset_root, slider_asset_file, slider_asset_options)
        
        # Load pusher asset
        pusher_asset_file = "{}/{}/{}.urdf".format(
            self.pusher_dataset_name,
            self.pusher_name,
            self.pusher_name)
        
        pusher_asset_options = gymapi.AssetOptions()
        pusher_asset_options.density = 2e10
        pusher_asset_options.fix_base_link = True
        pusher_asset_options.flip_visual_attachments = True
        pusher_asset_options.armature = 0.01
        pusher_asset_options.disable_gravity = True

        print("Loading asset '%s' from '%s'" % (pusher_asset_file, asset_root))
        pusher_asset = self.gym.load_asset(self.sim, asset_root, pusher_asset_file, pusher_asset_options)
        
        return slider_asset, pusher_asset

    def set_slider_friction_cofficient(self, env, slider_handle, friction_coefficient):
        ''' Set slider actor's friction_cofficient randomly in each environment 
        
        Inputs:
        - env
        - slider_handle
        - friction_cofficient_list: list of possible friction_cofficients of the slider
        
        '''
        
        shape_props = self.gym.get_actor_rigid_shape_properties(env, slider_handle)
        shape_props[0].friction = friction_coefficient
        shape_props[0].rolling_friction = 0
        shape_props[0].torsion_friction = 0
        shape_props[0].restitution = 0
        
        self.gym.set_actor_rigid_shape_properties(env, slider_handle, shape_props)
        
    def set_pusher_friction_cofficient(self, env, pusher_handle, pusher_friction_coefficient):
        ''' Set pusher actor's friction_cofficient uniformly in all environments
        
        Inputs:
        - env
        - pusher_handle
        
        '''
        
        shape_props = self.gym.get_actor_rigid_shape_properties(env, pusher_handle)
        shape_props[0].friction = pusher_friction_coefficient
        shape_props[0].rolling_friction = 0
        shape_props[0].torsion_friction = 0
        shape_props[0].restitution = 0
        
        self.gym.set_actor_rigid_shape_properties(env, pusher_handle, shape_props)
        
    def initialize_pusher_dof(self, env, pusher_actor_handle):
        '''
        Set default pusher actor DOF states and properties
        
        Note: The pusher should be far away from the slider and camera scene to make a depth image only icluding the slider.
        
        '''
        pusher_num_dofs = self.gym.get_actor_dof_count(env, pusher_actor_handle)
        
        pusher_dof_props = self.gym.get_actor_dof_properties(env, pusher_actor_handle)
        pusher_lower_limits = pusher_dof_props['lower']
        pusher_upper_limits = pusher_dof_props['upper']
        pusher_mids = 0.5 * (pusher_upper_limits + pusher_lower_limits)

        pusher_dof_state = np.zeros(pusher_num_dofs, gymapi.DofState.dtype)
        pusher_dof_state["pos"] = pusher_mids
        
        # Give a desired pose for first 3 robot joints to improve stability
        pusher_dof_props["driveMode"][0:3] = gymapi.DOF_MODE_POS
        pusher_dof_props['stiffness'] = 5000
        pusher_dof_props['damping'] = 1000
        
        return pusher_dof_state, pusher_dof_props
    
    def create_camera(self, env):
        ''' Create a camera sensor fixed to the environment '''
        
        # Create camera sensor
        camera_props = gymapi.CameraProperties()
        camera_props.width = int(self.camera_intrinsic[0,2] * 2.0)
        camera_props.height = int(self.camera_intrinsic[1,2] * 2.0)
        camera_props.horizontal_fov = 2*np.arctan2(self.camera_intrinsic[0,2], self.camera_intrinsic[0,0]) * 180/np.pi
        camera_props.far_plane = 1
        camera_handle  = self.gym.create_camera_sensor(env,camera_props)
        
        # Local camera pose (relative to the pusher eef)
        self.cam_pose = gymapi.Transform()
        # Actual camera xy values are inverted (-0.67, -0.16, 0.77) - experimental setting
        self.cam_pose.p = gymapi.Vec3(-0.67, -0.16, 0.77)
        self.cam_pose.r = gymapi.Quat.from_euler_zyx(0, np.deg2rad(90 - 13.5), 0) # top view
        self.gym.set_camera_transform(camera_handle,env,self.cam_pose)
        
        return camera_handle
    
    def reset_actor_poses(self, push_contact):
        '''
        For all envs:
            Relocate pusher pose to head to the pushing direction
            Relocate slider pose to zero pose
        
        Input: 
        
        - push_contact: a ContactPoint object of the first environment. 

            - edge_xyz (numpy.ndarray): (N, 3) array of edge points in world frame.
            - contact_points (numpy.ndarray): (2, 2) contact points in world frame.
            - contact_points_uv (numpy.ndarray): (2,2) contact points in image coordinates.
            - push_direction (numpy.ndarray): (2,) array of pushing direction in world frame.
            
        '''
        # step the physics
        self.gym.simulate(self.sim)
        
        # refresh results
        self.gym.fetch_results(self.sim, True)
        
        # Get pusher pose reset parameters
        push_direction_xy = push_contact.push_direction
        contact_points_xy = push_contact.contact_points
        contact_center = contact_points_xy.mean(0)
        contact_center_xy = contact_center[0:2]
        push_offset_xy =  self.initial_distance * push_direction_xy / np.linalg.norm(push_direction_xy)
        
        # Reset pusher pose
        pusher_pos_xy = contact_center_xy - push_offset_xy
        push_angl_z = np.arctan2(push_direction_xy[1], push_direction_xy[0])
        pusher_rot = R.from_euler('z', push_angl_z, degrees=False).as_quat()
        
        new_pusher_pose = gymapi.Transform()
        new_pusher_pose.p = gymapi.Vec3(pusher_pos_xy[0], pusher_pos_xy[1], 0)
        new_pusher_pose.r = gymapi.Quat(pusher_rot[0], pusher_rot[1], pusher_rot[2], pusher_rot[3])
            
        for env_idx in range(self.num_envs):
            
            pusher_rigid_body_handle = self.gym.get_actor_rigid_body_handle(self.envs[env_idx], self.pusher_actor_handles[env_idx],0)
            self.gym.set_rigid_transform(self.envs[env_idx], pusher_rigid_body_handle, new_pusher_pose)
            self.gym.set_rigid_linear_velocity(self.envs[env_idx], pusher_rigid_body_handle, gymapi.Vec3(0,0,0))
            self.gym.set_rigid_angular_velocity(self.envs[env_idx], pusher_rigid_body_handle, gymapi.Vec3(0,0,0))
            
        # step rendering
        self.gym.step_graphics(self.sim)
        if not self.args.headless:
            self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)
        
    def reset_camera_poses(self):
        ''' Reset camera poses to random posese for all envs
        
        Output: reset_camera_poses (n_envs, (4,4))
        
        '''
        
        reset_camera_poses = []
        
        # Default camera pose
        p_stable = self.cam_pose.p
        q_stable = self.cam_pose.r
        
        for env_idx in range(self.num_envs):
            # Get random camera pose
            
            p_random = gymapi.Vec3(
                np.random.uniform(-self.camera_rand_position_range, self.camera_rand_position_range),
                np.random.uniform(-self.camera_rand_position_range, self.camera_rand_position_range),
                np.random.uniform(-self.camera_rand_position_range, self.camera_rand_position_range))
            
            q_random = R.from_euler('z', np.random.uniform(-self.camera_rand_rotation_range, self.camera_rand_rotation_range), degrees=True).as_quat()
            q_random = gymapi.Quat(q_random[0], q_random[1], q_random[2], q_random[3])
            
            # Get random stable pose
            p = q_random.rotate(p_stable) + p_random
            q = (q_random*q_stable).normalize()
            cam_pose = gymapi.Transform(p, q)
            
            # Reset camera pose
            self.gym.set_camera_transform(self.camera_handles[env_idx],self.envs[env_idx],cam_pose)
            
            # Get camera extrinsic matrix
            # convert z = x, x = -y, y = -z
            rot = R.from_quat([cam_pose.r.x, cam_pose.r.y, cam_pose.r.z, cam_pose.r.w]).as_matrix()
            rot_convert = np.array([[0,0,1], [-1,0,0], [0,-1,0]])
            rot = np.dot(rot,rot_convert)
            camera_extr = np.eye(4)
            camera_extr[:3,:3] = rot
            camera_extr[:3,3] = np.array([cam_pose.p.x, cam_pose.p.y, cam_pose.p.z])
            
            reset_camera_poses.append(camera_extr)
            
        self.camera_poses = np.array(reset_camera_poses)
    
    def reset_envs(self,num_steps):
        ''' Reset the slider pose and pusher's DOF states '''
        # step the physics
        self.gym.simulate(self.sim)
        
        # refresh results
        self.gym.fetch_results(self.sim, True)
        
        # Set default slider pose
        default_slider_pose = self.slider_stable_pose
        
        if num_steps == 0:
            for env_idx in range(self.num_envs):
                
                if not self.headless:
                    self.gym.clear_lines(self.viewer)
                
                # Reset slider friction coefficient
                self.set_slider_friction_cofficient(self.envs[env_idx], self.slider_actor_handles[env_idx], self.friction_coefficient)
                
                # Reset camera poses
                self.reset_camera_poses()
                
                if env_idx == 0:
                    # Reset slider pose of the first environment (Randomly)
                    p_random = gymapi.Vec3(np.random.uniform(-self.slider_rand_position_range, self.slider_rand_position_range),
                                        np.random.uniform(-self.slider_rand_position_range, self.slider_rand_position_range), 0)
                    
                    q_random = R.from_euler('z', np.random.uniform(-self.slider_rand_rotation_range, -self.slider_rand_rotation_range), degrees=True).as_quat()
                    q_random = gymapi.Quat(q_random[0], q_random[1], q_random[2], q_random[3])
                    
                    p = q_random.rotate(default_slider_pose.p) + p_random
                    q = (q_random*default_slider_pose.r).normalize()
                    self.slider_reset_pose = gymapi.Transform(p, q)
                    
                slider_rigid_body_handle = self.gym.find_actor_rigid_body_handle(self.envs[env_idx], self.slider_actor_handles[env_idx], self.slider_name)
                self.gym.set_rigid_transform(self.envs[env_idx], slider_rigid_body_handle, self.slider_reset_pose)
                self.gym.set_rigid_linear_velocity(self.envs[env_idx], slider_rigid_body_handle, gymapi.Vec3(0,0,0))
                self.gym.set_rigid_angular_velocity(self.envs[env_idx], slider_rigid_body_handle, gymapi.Vec3(0,0,0))
                    
        # Reset pusher pose (to default)
        pusher_pose = gymapi.Transform()
        pusher_pose.p = gymapi.Vec3(0, 0, -2)
        pusher_pose.r = gymapi.Quat(0,0,0,1)
        
        for env_idx in range(self.num_envs):
            
            # Reset pusher DOF states to zero
            self.gym.set_actor_dof_states(self.envs[env_idx], self.pusher_actor_handles[env_idx], self.default_pusher_dof_state, gymapi.STATE_ALL)
            # Reset pusher pose
            pusher_rigid_body_handle = self.gym.find_actor_rigid_body_handle(self.envs[env_idx], self.pusher_actor_handles[env_idx], "base_link")
            self.gym.set_rigid_transform(self.envs[env_idx], pusher_rigid_body_handle, pusher_pose)
            self.gym.set_rigid_linear_velocity(self.envs[env_idx], pusher_rigid_body_handle, gymapi.Vec3(0,0,0))
            self.gym.set_rigid_angular_velocity(self.envs[env_idx], pusher_rigid_body_handle, gymapi.Vec3(0,0,0))
            
        
        for env_idx in range(self.num_envs):
            
            # Reset slider to zero pose
            slider_rigid_body_handle = self.gym.get_actor_rigid_body_handle(self.envs[env_idx], self.slider_actor_handles[env_idx],0)
            self.gym.set_rigid_transform(self.envs[env_idx], slider_rigid_body_handle, self.slider_reset_pose)
            self.gym.set_rigid_linear_velocity(self.envs[env_idx], slider_rigid_body_handle, gymapi.Vec3(0,0,0))
            self.gym.set_rigid_angular_velocity(self.envs[env_idx], slider_rigid_body_handle, gymapi.Vec3(0,0,0))
            
        
        # step rendering
        self.gym.step_graphics(self.sim)
        if not self.args.headless:
            self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)

    def step(self, n_step):
        ''' Step the simulation '''
        # Reset the environment
        self.reset_envs(n_step)
        
        if n_step == 0: # First step
            # Sample push contacts and save network image inputs
            self.push_contact, self.network_input_image = self.sample_push_contact(n_step) 
        
        # visualize contacts
        # self.visualize_contact(self.push_contact)
        
        # Execute pushing, evaluate the result, and save network input velocities and labels
        self.simulate_pushing(n_step, self.push_contact)
        
        if n_step + 1 == self.num_iters:
            # Visualize the network input images and labels
            self.visualize_push_results(self.network_input_image)
        
    def sample_push_contact(self, n_step):
        ''' 
            - Save the depth image for the network input
            - Get the push offset for all the environments
            
        Output:
            - push_contact: Push contact of the first environment which will be used for pushing in all the environments
        
        '''
        
        # step the physics
        self.gym.simulate(self.sim)
        
        # refresh results
        self.gym.fetch_results(self.sim, True)
        
        # step rendering
        self.gym.step_graphics(self.sim)
        if not self.args.headless:
            self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)

        # get images
        depth_image, segmask = self.get_camera_image()
        
        contact_point_sampler = ContactPointSampler(self.camera_intrinsic, self.camera_poses[0], self.gripper_width)
        
        # Get push_contacts (list of ContactPoint objects) by 8 different push directions
        push_contacts = contact_point_sampler.sample( depth_image, segmask)
        selected_idx = np.random.randint(0,len(push_contacts))
        
        # Randomly select the push contact
        push_contact = push_contacts[selected_idx]
        
        # cropped_image = crop_image(depth_image, push_contact)
        cropped_image = crop_image(np.multiply(depth_image,segmask), push_contact)
        
        # Normalize
        # cropped_image = (cropped_image - self.image_mean) / self.image_std
        cropped_image = (cropped_image - self.masked_image_mean) / self.masked_image_std
        
        return push_contact, cropped_image
    
    def get_push_results(self, push_contact_list):
        ''' Get pushing results for the given push contacts (with or without simulation) and save results if needed
        
        - Using simulation:
            - Simulate the pushing for all the environments
            - Save the results if needed
            
        - Using analytical method:
            - Evaluate pushing stability by analytical method (2D image based)
        '''
        # Get random pushing directions
        random_args = np.random.choice(np.arange(self.num_samples), size = self.num_envs, replace=False)
        icrs = self.icrs[random_args]
        pushing_directions = self.pushing_directions[random_args]
        
        if self.analytical == True:
            self.get_analytical_results(push_contact_list, icrs, pushing_directions)
        
        if self.analytical == False:
            self.get_simulation_results(push_contact_list, icrs, pushing_directions)
            
    def simulate_pushing(self, n_step, push_contact):
        ''' Push the slider and evaluate the push for all envs '''
        
        # Relocate pushers' pose to the contact points
        self.reset_actor_poses(push_contact)
        
        # Approach the pusher to the slider
        self.approach_to_slider(push_contact, self.translational_push_distance, self.dt, self.num_envs)
        
        # Get the slider pose before pushing
        eef_slider_poses_initial = self.get_eef_slider_pose()
        
        # Execute pushing
        self.execute_pushing(n_step)
        
        # Get the slider pose after pushing
        eef_slider_poses_final = self.get_eef_slider_pose()
        
        # Evaluate the push
        labels = evaluate_push_stability(eef_slider_poses_initial, eef_slider_poses_final, self.threshold_pos, self.threshold_rot)
        
        # save data
        self.labels.append(labels)
    
    def get_camera_image(self):
        """Get images from camera from the first environment

        Returns:
            depth_images (numpy.ndarray): image of shape (num_envs, H, W, 3)
            segmasks (numpy.ndarray): segmentation mask of shape (num_envs, H, W)
        """
        
        self.gym.render_all_camera_sensors(self.sim)
        
        depth_image = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0], gymapi.IMAGE_DEPTH)
        segmask = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0], gymapi.IMAGE_SEGMENTATION)
        # Change data type for lighter storage
        depth_image = np.array(depth_image, dtype = np.float32)
        # segmask = np.array(segmask, dtype = np.bool8)
        segmask = np.array(segmask, dtype = np.uint8) # for line contoured image
            
        depth_image = depth_image * -1
        segmask = np.array(segmask)
        
        
        return depth_image, segmask
    
    def approach_to_slider(self, push_contact, translational_push_distance, dt, num_envs):
        ''' Approach to the pusher for all envs '''
        
        # Get approach trajectories
        trajectories = get_approach_trajectories(self.initial_distance + translational_push_distance, self.push_speed, dt, num_envs)
        
        # Draw trajectory lines
        # if self.args.headless is False:
        #     for env_idx in range(num_envs):
        #         # Visualize push contacts
        #         self.draw_viewer_push_contact(env_idx, push_contact[env_idx])
        #         # Visualize approach trajectories
        #         self.draw_viewer_trajectories(env_idx, trajectories[env_idx])
        
        # Move the pusher
        if self.physics_engine == "PHYSX":
            # Use tensor API for faster simulation (only available for PhysX)
            # self.move_pusher_tensor(trajectories)
            self.move_pusher(trajectories)
        else:
            self.move_pusher(trajectories)
        
    def execute_pushing(self, n_step):
        ''' Execute the pushing action 
        
        Output: 
            - pushing_directions: the pushing directions (shape: (num_envs, 3)).
                This will be the input of the network.
        '''
        # Get the pusher trajectory in the world frame (for all envs)
        icrs = self.icrs[n_step*self.num_envs:(n_step+1)*self.num_envs]
        
        trajectories = icrs2trajectories(icrs, self.initial_distance + self.translational_push_distance, self.push_speed, self.dt)
        
        # Draw trajectory lines
        if self.args.headless is False:
            for env_idx in range(self.num_envs):
                self.draw_viewer_trajectories(env_idx, trajectories[env_idx])
                
        # Move the pusher (all envs)
        if self.physics_engine == "PHYSX":
            # Use tensor API for faster simulation (only available for PhysX)
            # self.move_pusher_tensor(trajectories)
            self.move_pusher(trajectories)
        else:
            self.move_pusher(trajectories)
        
    def get_eef_slider_pose(self):
        
        ''' Get the pose of the slider in pusher's end-effector frame for all envs
    
        Output: 
        - poses (num_envs, 4x4)
    
        '''
        poses = []
        
        for env_idx in range(self.num_envs):
            
            # Get slider poses of all envs
            slider_handle = self.gym.get_actor_rigid_body_handle(self.envs[env_idx], self.slider_actor_handles[env_idx], 0)
            slider_pose_rigid_transform = self.gym.get_rigid_transform(self.envs[env_idx], slider_handle)
            slider_pose = tmat(slider_pose_rigid_transform)
            
            # Get pusher end-effector poses of all envs
            eef_handle = self.gym.find_actor_rigid_body_handle(self.envs[env_idx], self.pusher_actor_handles[env_idx], "eef")
            eef_pose_rigid_transform = self.gym.get_rigid_transform(self.envs[env_idx], eef_handle)
            eef_pose = tmat(eef_pose_rigid_transform)
            
            # Get slider pose in eef frame
            eef_slider_pose = np.linalg.inv(eef_pose) @ slider_pose
            
            poses.append(eef_slider_pose)
        
        poses = np.array(poses)
        return poses
    
    def move_pusher(self, joint_trajectories):
        ''' Move the pusher in a given trajectory with position PD control (default Isaac Gym control method)
        
        Input: joint_trajectories (num_envs, (n, 3))
        
        '''
        # Convert the trajectory format for "set_actor_dof_position_targets" function input type
        joint_trajectories = joint_trajectories.astype(np.float32)
        
        # Transpose the joint_trajectories to (n, num_envs, 3)
        # print(joint_trajectories.shape)
        waypoints = np.transpose(joint_trajectories, (1,0,2))
        
        for waypoint in waypoints:
            
            # is_close = False
            for env_idx in range(self.num_envs):
                self.gym.set_actor_dof_position_targets(self.envs[env_idx], self.pusher_actor_handles[env_idx], waypoint[env_idx])
                    
            # while is_close == False:
            # Step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # Get current states
            cur_dof_states = self.gym.get_vec_actor_dof_states(self.envs, self.pusher_actor_handles, gymapi.STATE_ALL)
            cur_dof_pos = cur_dof_states['pos']
            
            
            # print(cur_dof_pos)
            # Deploy actions
            # is_close = np.allclose(cur_dof_pos, waypoint, atol=0.1)
            
            # Step rendering
            self.gym.step_graphics(self.sim)
            if not self.args.headless:
                self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)
                
    def move_pusher_tensor(self, joint_trajectories):
        ## Something is wrong with this function
        ''' Move the pusher in a given trajectory with position PD control (default Isaac Gym control method)
        
        Input: joint_trajectories (num_envs, (n, 3))
        
        Note: Tensor operations are used to speed up the simulation (only applicable with PHYSX backend)
        '''
        paths_tensor = torch.Tensor(joint_trajectories).permute(1,0,2) # (n, num_envs, 3)
        
        # Move the pusher to all waypoints
        for pos_tensor in paths_tensor:
            is_close = False
            
            # Move the pusher to each waypoint (all envs simulatneously)
            while is_close == False:
                
                # Step the physics
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)

                # refresh tensors
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                self.gym.refresh_dof_state_tensor(self.sim)
                
                # Deploy actions
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_tensor.contiguous()))
                
                _cur_dof_states = self.gym.acquire_dof_state_tensor(self.sim)
                cur_dof_states = gymtorch.wrap_tensor(_cur_dof_states)
                cur_dof_pos = cur_dof_states[:, 0].view(self.num_envs, 3)
                
                # print("Current:  ",cur_dof_pos)
                # print("Target :  ",pos_tensor)
                is_close = torch.allclose(cur_dof_pos,pos_tensor, atol=0.1)
                
                # print(is_close)
                
                # Step rendering
                self.gym.step_graphics(self.sim)
                self.gym.sync_frame_time(self.sim)
                if not self.args.headless:
                    self.gym.draw_viewer(self.viewer, self.sim, False)

    def draw_viewer_trajectories(self,env_idx,waypoints):
        '''
        Draw trajectory for each env in viewer
        
        Input: 
            - env (gymapi.Env)
            - waypoints (n, 3)
        
        '''
        
        waypoints_local = waypoints.copy()
        
        # Get pusher end-effector poses of all envs
        eef_handle = self.gym.find_actor_rigid_body_handle(self.envs[env_idx], self.pusher_actor_handles[env_idx], "eef")
        eef_pose_rigid_transform = self.gym.get_rigid_transform(self.envs[env_idx], eef_handle)
        eef_pose = tmat(eef_pose_rigid_transform)
        
        # Transform waypoints to world frame
        waypoints_world = np.zeros_like(waypoints_local)
        waypoints_local_homogeneous = np.hstack([waypoints_local, np.ones((waypoints_local.shape[0],1))])
        waypoints_world_homogeneous = eef_pose @ waypoints_local_homogeneous.T
        waypoints_world = waypoints_world_homogeneous.T[:,0:3]
        
        # Unify the z-axis of waypoints
        waypoints_world[:,2] = 0.0
        
        # Make lines for gym add_lines
        waypoints_world_end = np.roll(waypoints_world, -1, axis=0)
        waypoints_world_end[-1] = waypoints_world[-1]
        # Make lines for gym add_lines
        lines = np.hstack([waypoints_world, waypoints_world_end]).reshape(-1,3)
        color = np.array([1.0,0.0,0.0])
        colors = np.tile(color, (lines.shape[0], 1))
        
        self.gym.add_lines(self.viewer,self.envs[env_idx], len(waypoints_world), lines.astype(np.float32), colors.astype(np.float32))
        
    def draw_viewer_push_contact(self,env_idx, push_contact):
        ''' 
        Draw contact points for each env in viewer
        
        Input: 
            - env (gymapi.Env)
            - push_contact (ContactPoint object)
        
        '''
        
        ## Draw contact points
        contact_points = push_contact.contact_points
        # Shift the contact points (2d) to 3d
        point1, point2 = np.append(contact_points[0],0), np.append(contact_points[1],0)
        
        lines = np.hstack([point1, point2]).reshape(-1,3)
        color = np.array([0.0,0.0,1.0])
        colors = np.tile(color, (lines.shape[0], 1))
        
        self.gym.add_lines(self.viewer,self.envs[env_idx], len(lines), lines.astype(np.float32), colors.astype(np.float32))
        
        ## Draw slider edge
        edge_xyz = push_contact.edge_xyz
        color = np.array([1.0,1.0,0.0])
        colors = np.tile(color, (edge_xyz.shape[0], 1))
        
        self.gym.add_lines(self.viewer, self.envs[env_idx], len(edge_xyz), edge_xyz.astype(np.float32), colors.astype(np.float32))
        
    def visualize_contact(self, push_contact):
        ''' 
        Visualize contact points in viewer
        
        Input: 
            - push_contact (ContactPoint object)
        
        '''
        
        fig = plt.figure()
        edge_xyz = push_contact.edge_xyz
        contacts_xy = push_contact.contact_points
        push_direction = push_contact.push_direction
        
        ax = fig.add_subplot(111)
        ax.scatter(edge_xyz[:,0], edge_xyz[:,1], color = np.array([0.0,0.0,1.0]))
        # left point
        ax.scatter(contacts_xy[0,0] - push_direction[0] , contacts_xy[0,1] - push_direction[1] , color = np.array([1.0,0.0,0.0]))
        ax.scatter(contacts_xy[0,0]  , contacts_xy[0,1]  , color = np.array([1.0,0.0,0.0]))
        
        # right point
        ax.scatter(contacts_xy[1,0] - push_direction[0] , contacts_xy[1,1] - push_direction[1] , color = np.array([1.0,0.0,1.0]))
        ax.scatter(contacts_xy[1,0]  , contacts_xy[1,1]  , color = np.array([1.0,0.0,1.0]))
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        
        plt.show()
        
    def visualize_push_results(self, network_input_image):
        '''
        Visualize ground truth results and network output
        
        '''
        # Get ground truth labels
        labels = np.array(self.labels).reshape(-1,self.num_samples)
        
        # Get network output
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # MODEL_NAME = '2023-04-29-0047_masked'
        MODEL_NAME = '2023-05-18-2230_masked'
        model = PushNet()
        model.to(DEVICE)
        model.load_state_dict(torch.load(F'../stable_pushnet/models/{MODEL_NAME}/model.pt'))
        
        # Preprocess image
        network_input_image_tensor = torch.from_numpy(network_input_image.astype(np.float32)).to(DEVICE)
        network_input_image_tensor = network_input_image_tensor.unsqueeze(0)
        network_input_image_tensors = torch.tile(network_input_image_tensor, (self.num_samples,1,1,1))
        network_input_image_tensors.to(DEVICE)
        
        # Preprocess velocity inputs
        velocity_inputs = torch.from_numpy(self.pushing_directions.astype(np.float32)).to(DEVICE)
        
        # Get network outputs
        network_output = model(network_input_image_tensors, velocity_inputs).cpu()
        predictions = torch.nn.Softmax()(network_output)[:,1].detach().numpy()
        
        
        fig = plt.figure()
        # 1. Draw depth image
        ax = fig.add_subplot(131)
        ax.imshow(network_input_image)
        
        # 2. Plot ground truth (simulation)
        # ax = fig.add_subplot(132,projection='3d')
        ax = fig.add_subplot(132)
        # ax.view_init(elev = 0,azim = 0)
        # ax.set_xlabel(r"$v_x$ [m/s]")
        # ax.set_ylabel(r"$v_y$ [m/s]")
        # ax.set_zlabel(r"$\omega$ [rad]", rotation=0)
        # ax.set_box_aspect((1,2,2))
        ax.set_aspect('equal')
        ax.grid(False)
        ax.legend()
        v = self.pushing_directions
        # ax.scatter(v[:,0], v[:,1], v[:,2], c = labels, cmap = 'jet', s=150)
        ax.scatter(v[:,1], v[:,2], c = labels, cmap = 'jet', s=30)
        
        # 3. Plot network output
        ax = fig.add_subplot(133)
        # ax = fig.add_subplot(133,projection='3d')
        # ax.view_init(elev = 0,azim = 0)
        # ax.set_xlabel(r"$v_x$ [m/s]")
        # ax.set_ylabel(r"$v_y$ [m/s]")
        # ax.set_zlabel(r"$\omega$ [rad]", rotation=0)
        # ax.set_box_aspect((1,2,2))
        ax.set_aspect('equal')
        ax.grid(False)
        ax.legend()
        v = self.pushing_directions
        discrepency = np.abs(predictions - labels)
        # p = ax.scatter(v[:,0], v[:,1], v[:,2], c = predictions, cmap = 'jet', s=150, vmin=0, vmax=1)
        p = ax.scatter( v[:,1], v[:,2], c = predictions, cmap = 'jet', s=30, vmin=0, vmax=1)
        fig.colorbar(p,ax=ax)
        
        plt.show()
        
        # label_indices = np.random.randint(0,self.init_file_idx,size=1000)
        # labels, velocities = [], []
        # for label_index in label_indices:
        #     label_name = ("%s_%0" + str(self.FILE_ZERO_PADDING_NUM) + 'd.npy')%('label',label_index)
        #     vel_name = ("%s_%0" + str(self.FILE_ZERO_PADDING_NUM) + 'd.npy')%('velocity',label_index)
        #     label = np.load(os.path.join(self.save_dir, label_name))
        #     vel = np.load(os.path.join(self.save_dir, vel_name))
        #     velocities.append(vel)
        #     labels.append(label)
        # labels = np.array(labels)
        # v = np.array(velocities)
        
        # # 4. Plot train data using network output   
        # ax = fig.add_subplot(144,projection='3d')
        # ax.view_init(elev = 0,azim = 0)
        # ax.set_xlabel(r"$v_x$ [m/s]")
        # ax.set_ylabel(r"$v_y$ [m/s]")
        # ax.set_zlabel(r"$\omega$ [rad]", rotation=0)
        # ax.set_box_aspect((1,2,2))
        # ax.grid(False)
        # ax.legend()
        # p = ax.scatter(v[:,0], v[:,1], v[:,2], c = labels, cmap = 'jet', s=150)
        # plt.show()
        
if __name__ == "__main__":
    
    objects = ["melamineware_g_0151", "melamineware_g_0130", "dish_small", "rice_bowl", "soup_bowl", "melamineware_g_0001"]
    
    # for obj in objects:
    env = PushSim()
    for i in tqdm(range(env.num_iters), desc = f'Object {"rice_bowl"}'):
        start_time = time.time()
        env.step(i)
        # print(f'Iteration {i} took {time.time() - start_time} seconds')
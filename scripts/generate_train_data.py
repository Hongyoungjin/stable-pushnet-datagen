# import python modules
import time
import os
import yaml
import argparse
import pickle
import trimesh

# import isaacgym modules
from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import *

# import 3rd party modules
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


# import local modules
from utils.crop_image_parallel import CropImageParallel
from utils.sample_push_contact_parallel import SamplePushContactParallel
from utils.push_dof_tools import *
from utils.stable_region_analytical import StableRegion

class PushSim(object):
    
    def __init__(self):
        
        # set default push velocities
        self.num_samples = 1000
        self.pushing_directions = fibonacci_sphere(self.num_samples * 2)
        self.icrs = velocities2icrs(self.pushing_directions)
        
        # initialize gym
        self.gym = gymapi.acquire_gym()
    
        # parse arguments
        parser = argparse.ArgumentParser(description="Push Sim: Push simulation of tableware for stable pushing network training data generation")
        parser.add_argument('--config', type=str, default="/home/hong/ws/twc-stable-pushnet/config/config_pushsim.yaml", help='Configuration file')
        parser.add_argument('--headless', type=bool, default=False, help='Turn on the viewer')
        parser.add_argument('--analytical', type=bool, default=False, help='Whether to use analytical stable region or not')
        parser.add_argument('--use_gpu_pipeline', type=bool, default=False, help='Whether to use gpu pipeline')
        parser.add_argument('--save_results', action='store_true', help='save results')
        # parser.add_argument('-d','--slider_dataset', type=str, default='dish_urdf', help='slider dataset name')
        parser.add_argument('-n','--slider_name', type=str, default='glass', help='slider name')
        self.args = parser.parse_args()
        
        self._load_configuration()
        self._create_simulation()
        self.gym.prepare_sim(self.sim) # Prepare simulation with buffer allocations
        self._create_ground()
        self._create_viewer()
        self._create_environments()
        
    def _load_configuration(self):
        ''' Configurate the entire simulation by conveying config data from configuration file '''
        config_file = self.args.config
        self.headless = self.args.headless
        self.save_results = self.args.save_results
        self.analytical = self.args.analytical
        self.slider_name = self.args.slider_name
        
        # Load configuation file
        with open(config_file,'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        
        sim_cfg = cfg["simulation"]
        cam_cfg = sim_cfg["camera"]["IntelRealSenseD415"]['color']
        # cam_cfg = sim_cfg["camera"]["ZividTwo"]
        
        # simulation setup (default)
        self.FILE_ZERO_PADDING_NUM = sim_cfg["FILE_ZERO_PADDING_NUM"]
        self.physics_engine = sim_cfg["physics_engine"]
        self.num_threads = sim_cfg["num_threads"]
        self.compute_device_id = sim_cfg["compute_device_id"]
        self.graphics_device_id = sim_cfg["graphics_device_id"]
        self.num_envs = sim_cfg['num_envs']
        self.use_gpu = sim_cfg['use_gpu']
        self.use_gpu_pipeline = sim_cfg['use_gpu_pipeline']
        self.dt = sim_cfg["dt"]
        self.render_freq = 1/sim_cfg["render_freq"]
        self.num_iters = sim_cfg["num_iters"]
        
        # Camera configuration
        fx, fy, cx, cy = cam_cfg["fx"], cam_cfg["fy"], cam_cfg["cx"], cam_cfg["cy"]
        self.camera_intrinsic = np.array([[fx,0,cx],
                                          [0,fy,cy],
                                          [0, 0, 1]])
        self.camera_rand_position_range = sim_cfg["camera_rand_position_range"]
        self.camera_rand_rotation_range = sim_cfg["camera_rand_rotation_range"]
        
        self.camera_default_length = sim_cfg["camera"]["IntelRealSenseD415"]['camera_pose']['x']
        self.camera_default_height = sim_cfg["camera"]["IntelRealSenseD415"]['camera_pose']['z']
        self.camera_default_angle = sim_cfg["camera"]["IntelRealSenseD415"]['camera_pose']['r']
        
        self.camera_poses = None # Camera poses will be set in reset_camera_poses()
        
        # Asset name setup
        self.slider_dataset_name = sim_cfg["slider_dataset"]
        
        # Set pusher name (slider name is already set in the constructor)
        self.pusher_dataset_name = sim_cfg["pusher_dataset"]
        self.pusher_name = sim_cfg["pusher_name"]
        
        # Pushing velocity
        self.push_speed = sim_cfg["push_speed"]
        
        # Slider initial contact perturbation
        self.rand_contact_position_range = sim_cfg["rand_contact_position_range"]
        rand_contact_rotation_range = sim_cfg["rand_contact_rotation_range"]
        self.rand_contact_rotation_range = np.deg2rad(rand_contact_rotation_range)
        
        
        # Slider friction coefficients
        max_friction = sim_cfg['max_friction_coefficient']
        min_friction = sim_cfg['min_friction_coefficient']
        
        # Pusher friction coefficient
        self.pusher_friction_coefficient = sim_cfg['pusher_friction_coefficient']
        self.friction_coefficients = [i for i in np.linspace(min_friction,max_friction,30)]
        
        # Initial approach distance of the pusher to the slider
        self.translational_push_distance = sim_cfg["translational_push_distance"]
        
        # Initial distance between pusher and the slider
        self.initial_distance = sim_cfg["initial_distance"]
        
        # Gripper width
        self.gripper_width = sim_cfg["gripper_width"]
        
        # Slider's stable pose
        stable_pose = np.load("assets/{}/{}/stable_poses.npy".format(self.slider_dataset_name, self.slider_name))
        self.slider_default_pose = stable_pose
        self.slider_default_pose[:2,3] = np.zeros(2)
        # For the assets that did not choose optimum stable pose.
        if len(stable_pose.shape) > 2:
            print("Error: Slider's stable pose must have only one stable pose. Aborting simulation.")
            exit()
        
        quat = R.from_matrix(stable_pose[:3,:3]).as_quat()
        self.slider_stable_pose = gymapi.Transform()
        self.slider_stable_pose.p = gymapi.Vec3(0.45, 0.1, stable_pose[2,3])
        self.slider_stable_pose.r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
        
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
        
        self.image_idx    = self.init_file_idx + 1 # Index of new file should start from the next index of the last file
        self.velocity_and_label_idx = self.init_file_idx + 1
        
    def _create_simulation(self):
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
        plane_params.static_friction = 0
        plane_params.dynamic_friction = 0
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
            
            cam_pos = gymapi.Vec3(1, 0.001, 3)
            cam_target = gymapi.Vec3(1, 0, 0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            
            # key callback
            self.gym.subscribe_viewer_keyboard_event(self.viewer,gymapi.KEY_ESCAPE,"QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer,gymapi.KEY_V,"toggle_viewer_sync")

            if self.viewer is None:
                print("*** Failed to create viewer")
                quit()
    
    def _create_environments(self):
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
            
            # Create camera sensors
            camera_handle = self.create_camera_sensors(env)
            
            # Set slider friction coefficient randomly
            rand_friction = self.friction_coefficients[np.random.randint(0, len(self.friction_coefficients))]
            # self.set_actor_rigid_body_friction_coefficient(env, slider_actor_handle, rand_friction)
            self.set_actor_rigid_body_friction_coefficient(env, slider_actor_handle, self.pusher_friction_coefficient)
            
            # Set pusher friction coefficient in a constant value
            # self.set_actor_rigid_body_friction_coefficient(env, pusher_actor_handle, self.pusher_friction_coefficient)
            self.set_actor_rigid_body_friction_coefficient(env, pusher_actor_handle, rand_friction)
            
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
        
    def set_actor_rigid_body_friction_coefficient(self, env, actor_handle, friction_coefficient):
        ''' Set actor rigid body's friction_cofficient
        
        Inputs:
        - env
        - actor_handle
        - friction_coefficient: friction coefficient of the actor
        
        '''
        
        shape_props = self.gym.get_actor_rigid_shape_properties(env, actor_handle)
        shape_props[0].friction = friction_coefficient
        shape_props[0].rolling_friction = 0
        shape_props[0].torsion_friction = 0
        shape_props[0].restitution = 0
        
        self.gym.set_actor_rigid_shape_properties(env, actor_handle, shape_props)
        
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
     
    def create_camera_sensors(self, env):
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
        # self.cam_pose.p = gymapi.Vec3(-0.67, -0.16, self.camera_default_height)
        self.cam_pose.p = gymapi.Vec3(self.camera_default_length, -0.16, self.camera_default_height)
        self.cam_pose.r = gymapi.Quat.from_euler_zyx(0, np.deg2rad(90 - self.camera_default_angle), 0) # top view
        # self.cam_pose.r = gymapi.Quat.from_euler_zyx(0, np.deg2rad(90+13.5), np.pi/2) # top view
        self.gym.set_camera_transform(camera_handle,env,self.cam_pose)
        
        return camera_handle
    
    def relocate_pusher(self, push_contact_list):
        '''
        For all envs:
            Relocate pusher pose to head to the pushing direction
        
        Input: 
        
        - push_contact_list: list of ContactPoint objects. 
            Each object contains:
            - edge_xyz (numpy.ndarray): (N, 3) array of edge points in world frame.
            - contact_points (numpy.ndarray): (2, 2) contact points in world frame.
            - contact_points_uv (numpy.ndarray): (2,2) contact points in image coordinates.
            - push_direction (numpy.ndarray): (2,) array of pushing direction in world frame.
            
        '''
        # step the physics
        self.gym.simulate(self.sim)
        
        # refresh results
        self.gym.fetch_results(self.sim, True)
        
        for env_idx in range(self.num_envs):
            
            # Get pusher pose reset parameters
            push_contact = push_contact_list[env_idx]
            
            if push_contact is None:
                continue
            
            push_direction_xy = push_contact.push_direction
            push_angl_z = np.arctan2(push_direction_xy[1], push_direction_xy[0])
            
            contact_points_xyz = push_contact.contact_points
            contact_center = contact_points_xyz.mean(0)[:2]
            
            
            # Randomly perturb the initial contact pose
            initial_contact_pose = [contact_center[0], contact_center[1], push_angl_z]
            perturbated_contact_pose = perturabte_initial_contact(initial_contact_pose,
                                                                  self.rand_contact_position_range,
                                                                  self.rand_contact_rotation_range)
            contact_center = perturbated_contact_pose[:2]
            push_angl_z = perturbated_contact_pose[2]
            push_direction_xy = np.array([np.cos(push_angl_z), np.sin(push_angl_z)])
            
            # Derive push offset
            push_offset_xy =  self.initial_distance * push_direction_xy / np.linalg.norm(push_direction_xy)
            
            # Reset pusher pose
            pusher_pos_xy = contact_center - push_offset_xy
            pusher_rot = R.from_euler('z', push_angl_z, degrees=False).as_quat()
            
            new_pusher_pose = gymapi.Transform()
            new_pusher_pose.p = gymapi.Vec3(pusher_pos_xy[0], pusher_pos_xy[1], 0)
            new_pusher_pose.r = gymapi.Quat(pusher_rot[0], pusher_rot[1], pusher_rot[2], pusher_rot[3])
            
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
            
            p_random = gymapi.Vec3(0, 0, np.random.uniform(-self.camera_rand_position_range, self.camera_rand_position_range))
            
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
    
    def reset_environments(self):
        ''' Reset the slider pose and pusher's DOF states '''
        self.slider_reset_pose = []
        # Set default slider pose
        default_slider_pose = self.slider_stable_pose
        
        for env_idx in range(self.num_envs):
            
            if not self.headless:
                self.gym.clear_lines(self.viewer)
            
            # Reset slider friction coefficient
            rand_friction = self.friction_coefficients[np.random.randint(0, len(self.friction_coefficients))]
            self.set_actor_rigid_body_friction_coefficient(self.envs[env_idx], self.slider_actor_handles[env_idx], rand_friction)
            
            # Reset camera poses
            self.reset_camera_poses()
            
            # Reset slider pose (Randomly)
            p_random = gymapi.Vec3(np.random.uniform(-self.slider_rand_position_range, self.slider_rand_position_range),
                                   np.random.uniform(-self.slider_rand_position_range, self.slider_rand_position_range), 0)
            
            q_random = R.from_euler('z', np.random.uniform(-self.slider_rand_rotation_range, -self.slider_rand_rotation_range), degrees=True).as_quat()
            q_random = gymapi.Quat(q_random[0], q_random[1], q_random[2], q_random[3])
            
            p = q_random.rotate(default_slider_pose.p) + p_random
            q = (q_random*default_slider_pose.r).normalize()
            slider_pose = gymapi.Transform(p, q)
            
            slider_rigid_body_handle = self.gym.find_actor_rigid_body_handle(self.envs[env_idx], self.slider_actor_handles[env_idx], self.slider_name)
            self.gym.set_rigid_transform(self.envs[env_idx], slider_rigid_body_handle, slider_pose)
            self.gym.set_rigid_linear_velocity(self.envs[env_idx], slider_rigid_body_handle, gymapi.Vec3(0,0,0))
            self.gym.set_rigid_angular_velocity(self.envs[env_idx], slider_rigid_body_handle, gymapi.Vec3(0,0,0))
            
            self.slider_reset_pose.append(slider_pose)
            
            # Reset pusher pose (to default)
            pusher_pose = gymapi.Transform()
            pusher_pose.p = gymapi.Vec3(0, 0, -2)
            pusher_pose.r = gymapi.Quat(0,0,0,1)
            
            pusher_rigid_body_handle = self.gym.find_actor_rigid_body_handle(self.envs[env_idx], self.pusher_actor_handles[env_idx], "base_link")
            self.gym.set_rigid_transform(self.envs[env_idx], pusher_rigid_body_handle, pusher_pose)
            self.gym.set_rigid_linear_velocity(self.envs[env_idx], pusher_rigid_body_handle, gymapi.Vec3(0,0,0))
            self.gym.set_rigid_angular_velocity(self.envs[env_idx], pusher_rigid_body_handle, gymapi.Vec3(0,0,0))
            
            # Reset pusher DOF states to zero
            self.gym.set_actor_dof_states(self.envs[env_idx], self.pusher_actor_handles[env_idx], self.default_pusher_dof_state, gymapi.STATE_ALL)
            
    def step_simulation(self):
        ''' Step the simulation '''
        
        # Sample push contacts and save network image inputs
        push_contact_list = self.sample_push_contacts() 
        
        if all([push_contact is None for push_contact in push_contact_list]):
            print("Error: No push contacts found for all environments. \nThere may be some problems regarding the size of the pusher. \nAborting simulation.")
            exit()
            
        # Get pushing results for the given push contacts (with or without simulation) and save results if needed
        self.get_push_results(push_contact_list)
        
        if self.image_idx != self.velocity_and_label_idx:
            print("Error: Data file mismatch. Data indexing is contaminated. \nAborting simulation.")
            exit()
        
    def sample_push_contacts(self):
        ''' 
            - Save the depth image for the network input
            - Get the push offset for all the environments
            
        Output:
            - push_offsets: Push offsets for all the environments (num_envs, (x,y,z))
                Used for pusher relocation in substep2
                
            - contact_angles: polar angles of contact points wrt object centroid (num_envs, )
        
        '''
        # Reset the environment
        self.reset_environments()
        
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
        depth_images, segmasks = self.get_camera_image()
        push_contact_list = []

        # fig = plt.figure(figsize=(10,10))
        # col = int(np.ceil(np.sqrt(self.num_envs)))
        # for i in range(self.num_envs):
        #     ax = fig.add_subplot(col,col,i+1)
        #     ax.imshow(depth_images[i])
        # plt.show()
        
        # sample a contact point
        spc = SamplePushContactParallel(self.num_envs, self.camera_intrinsic, self.gripper_width)
        ci = CropImageParallel(self.num_envs, self.camera_intrinsic, self.gripper_width)
        print("Sampling push contacts...")
        push_contact_list = spc.sample_push_contacts(depth_images, segmasks, self.camera_poses)
        
        push_contact = push_contact_list[0]
        edge_list_uv = push_contact.edge_uv
        contact_point = push_contact.contact_points_uv[0]
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(depth_images[0])
        # ax.scatter(contact_point[0], contact_point[1], color='r')
        # ax.scatter(edge_list_uv[:,0], edge_list_uv[:,1], color='b')
        # plt.show()
        
        # ################################
        # # Debugging for function speed #
        # ################################
        # times = []
        # for env_idx in range(self.num_envs):
        #     start_time = time.time()
            
        #     cps = ContactPointSampler(self.camera_intrinsic, self.camera_poses[env_idx], self.gripper_width)
        #     push_contacts = cps.sample(depth_images[env_idx], segmasks[env_idx])
        #     push_contact = push_contacts[1]
            
        #     cropped_depth_image = crop_image(depth_images[env_idx], push_contact)
            
        #     end_time = time.time()
        #     times.append(end_time - start_time)
            
        # times = np.array(times)
        # print("Average time: ", np.mean(times))
        # print("Std time: ", np.std(times))
        
        # plt.figure()
        # plt.hist(times, bins=int(np.ceil(self.num_envs/2)))
        # plt.title("Mean: %f, Std: %f"%(np.mean(times), np.std(times)))
        # plt.suptitle("Time for sampling and preprocessing depth image")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Frequency")
        # plt.show()
        
        # exit()
        
        # plt.figure()
        # plt.imshow(depth_images[0])
        # plt.show()
        
               
        cropped_depth_images = ci.crop_images_parallel(depth_images, push_contact_list)
        # fig = plt.figure(figsize=(10,10))
        # col = int(np.ceil(np.sqrt(self.num_envs)))
        # for i in range(self.num_envs):
        #     ax = fig.add_subplot(col,col,i+1)
        #     ax.imshow(cropped_depth_images[i])
        # plt.show()
        
        if self.save_results:
            
            print("Saving network inputs...")
            cropped_depth_images = ci.crop_images_parallel(depth_images, push_contact_list)
            cropped_segmasks = ci.crop_images_parallel(segmasks , push_contact_list)
            # cropped_masked_depth_images = ci.crop_images_parallel(np.multiply(depth_images, segmasks) , push_contact_list)
            
            for env_idx in range(self.num_envs):
                cropped_depth_img, cropped_segmask = cropped_depth_images[env_idx], cropped_segmasks[env_idx]
                
                ref_push_contact = push_contact_list[env_idx]
                if self.save_results:
                    
                        
                    if ref_push_contact is None:
                        continue
                    
                    else:
                        
                        name = ("_%0" + str(self.FILE_ZERO_PADDING_NUM) + 'd.npy')%(self.image_idx)
                        
                        with open(os.path.join(self.save_dir, 'image' + name), 'wb') as f:
                            np.save(f, cropped_depth_img)
                            
                        with open(os.path.join(self.save_dir, 'masked_image' + name), 'wb') as f:
                            np.save(f, cropped_segmask * cropped_depth_img)
                            
                            
                    self.image_idx += 1
                    
        return push_contact_list
    
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
   
    def get_analytical_results(self, push_contact_list, icrs, pushing_directions):
        
        if self.save_results:
            for env_idx in range(self.num_envs):
                    
                # Push contact properties
                push_contact = push_contact_list[env_idx]
                
                if push_contact is None:
                    continue
                
                edge_xyz = push_contact.edge_xyz
                contact_points = push_contact.contact_points
                contact_normals = push_contact.contact_normals
                
                # Get the slider friction coefficient
                shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_idx], self.slider_actor_handles[env_idx])
                slider_firction = shape_props[0].friction
                mu = (self.pusher_friction_coefficient + slider_firction) / 2 # Actual friction coefficient
                
                # Determine stability
                sr = StableRegion(edge_xyz[:,:2], contact_points, contact_normals, mu)
                stability_label = sr.is_stable_in_local_frame(icrs[env_idx])
                
                name = ("_%0" + str(self.FILE_ZERO_PADDING_NUM) + 'd.npy')%(self.velocity_and_label_idx)
                    
                # Save each pushing direction (network input)
                with open(os.path.join(self.save_dir, 'velocity' + name), 'wb') as f:
                    np.save(f, pushing_directions[env_idx])
                
                # Save each label (network output)
                with open(os.path.join(self.save_dir, 'label' + name), 'wb') as f:
                    np.save(f, stability_label)
                    
                self.velocity_and_label_idx += 1
        else:
            pass
   
    def get_simulation_results(self, push_contact_list, icrs, pushing_directions):
        
        # Relocate pushers' pose to the contact points
        self.relocate_pusher(push_contact_list)
        
        # Approach the pusher to the slider 
        approach_trajectories = get_approach_trajectories(self.initial_distance + self.translational_push_distance, self.push_speed, self.dt, self.num_envs)
        pushing_trajectories = icrs2trajectories(icrs, self.initial_distance + self.translational_push_distance, self.push_speed, self.dt)
        
        for env_idx in range(self.num_envs):
            
            if push_contact_list[env_idx] is None:
                continue
            
            self.draw_viewer_push_contact(env_idx, push_contact_list[env_idx])
            self.draw_viewer_trajectories(env_idx, np.concatenate((approach_trajectories[env_idx],pushing_trajectories[env_idx]), axis=0))
            
        self.move_pusher(approach_trajectories)
        
        # Get the slider pose before pushing
        eef_slider_poses_initial = self.get_eef_slider_pose()
        
        # Execute pushing
        
        for env_idx in range(self.num_envs):
            
            if push_contact_list[env_idx] is None:
                continue
            # self.draw_viewer_trajectories(env_idx, pushing_trajectories[env_idx])
            
            
        # self.move_pusher(pushing_trajectories, logging=True)
        self.move_pusher(pushing_trajectories)  
        
        # Get the slider pose after pushing
        eef_slider_poses_final = self.get_eef_slider_pose()
        
        # Evaluate the push
        labels = evaluate_push_stability(eef_slider_poses_initial, eef_slider_poses_final, self.threshold_pos, self.threshold_rot)
        self.gym.clear_lines(self.viewer)
        # save data
        if self.save_results:
            
            for env_idx in range(self.num_envs):
                
                if push_contact_list[env_idx] is None:
                    continue
                
                name = ("_%0" + str(self.FILE_ZERO_PADDING_NUM) + 'd.npy')%(self.velocity_and_label_idx)
                
                # Save each pushing direction (network input)
                with open(os.path.join(self.save_dir, 'velocity' + name), 'wb') as f:
                    np.save(f, pushing_directions[env_idx])
                
                # Save each label (network output)
                with open(os.path.join(self.save_dir, 'label' + name), 'wb') as f:
                    np.save(f, labels[env_idx])
                    
                self.velocity_and_label_idx += 1
    
    def get_camera_image(self):
        """Get images from camera

        Returns:
            depth_images (numpy.ndarray): image of shape (num_envs, H, W, 3)
            segmasks (numpy.ndarray): segmentation mask of shape (num_envs, H, W)
        """
        depth_images, segmasks = [], []
        
        self.gym.render_all_camera_sensors(self.sim)
        
        for i in range(self.num_envs):
            
            depth_image = self.gym.get_camera_image(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_DEPTH)
            segmask = self.gym.get_camera_image(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_SEGMENTATION)
            # Change data type for lighter storage
            depth_image = np.array(depth_image, dtype = np.float32)
            # segmask = np.array(segmask, dtype = np.bool8)
            segmask = np.array(segmask, dtype = np.uint8) # for line contoured image
            
            depth_images.append(depth_image)
            segmasks.append(segmask)
            
        depth_images = np.array(depth_images) * -1
        segmasks = np.array(segmasks)
        
        # fig = plt.figure(figsize=(10, 10))
        # columns = int(np.ceil(np.sqrt(depth_images.shape[0])))
        # for i in range(self.num_envs):
        #     ax = fig.add_subplot(columns, columns, i+1)
        #     ax.imshow(depth_images[i])
        # plt.show()
        
        return depth_images, segmasks
    
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
    
    def move_pusher(self, joint_trajectories, logging=False):
        ''' Move the pusher in a given trajectory with position PD control (default Isaac Gym control method)
        
        Input: joint_trajectories (num_envs, (n, 3))
        
        '''
        # Convert the trajectory format for "set_actor_dof_position_targets" function input type
        joint_trajectories = joint_trajectories.astype(np.float32)
        
        # Transpose the joint_trajectories to (n, num_envs, 3)
        waypoints = np.transpose(joint_trajectories, (1,0,2))
        
        # Log the dof states
        num_pusher_dofs = self.gym.get_actor_dof_count(self.envs[0], self.pusher_actor_handles[0])
        
        self.pos_dof_array = np.zeros((self.num_envs, waypoints.shape[0], num_pusher_dofs))
        self.vel_dof_array = np.zeros((self.num_envs, waypoints.shape[0], num_pusher_dofs))
        
        for waypoint_idx, waypoint in enumerate(waypoints):
            
            for env_idx in range(self.num_envs):
                self.gym.set_actor_dof_position_targets(self.envs[env_idx], self.pusher_actor_handles[env_idx], waypoint[env_idx])
                    
            # Step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # Step rendering
            self.gym.step_graphics(self.sim)
            self.gym.sync_frame_time(self.sim)
            
            if logging==True:
                # Log the dof states
                self.log_dof_states(waypoint_idx)
            
            if not self.args.headless:
                self.gym.draw_viewer(self.viewer, self.sim, False)
        
        if logging==True: 
            # Plot the dof states to check whether the pusher is moving as expected
            # Transpose the joint_trajectories to (num_envs, n, 3)
            waypoints = np.transpose(waypoints, (1,0,2))
            self.plot_dof_states(waypoints, self.pos_dof_array, self.vel_dof_array, self.dt) 
        
               
    def move_pusher_with_constant_velocity(self, velocity_waypoints, time):
        '''
        Move the pusher with constant velocity for a given time
        Inputs:
            - velocity (num_envs, 3)
            - time (float)
        '''
        # Set velocity targets for all envs
        for env_idx in range(self.num_envs):
            self.gym.set_actor_dof_velocity_targets(self.envs[env_idx], self.pusher_actor_handles[env_idx], velocity_waypoints[env_idx])
        
        # Step the physics

    def draw_viewer_trajectories(self, env_idx, waypoints):
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
        waypoints_world[:,2] = 0.01
        
        # Make lines for gym add_lines
        waypoints_world_end = np.roll(waypoints_world, -1, axis=0)
        waypoints_world_end[-1] = waypoints_world[-1]
        
        # Make lines for gym add_lines
        lines = np.hstack([waypoints_world, waypoints_world_end]).reshape(-1,3)
        color = np.array([1.0,0.0,0.0])
        colors = np.tile(color, (lines.shape[0], 1))
        
        self.gym.add_lines(self.viewer,self.envs[env_idx], len(waypoints_world), lines.astype(np.float32), colors.astype(np.float32))
        
    def draw_viewer_push_contact(self, env_idx, push_contact):
        ''' 
        Draw contact points for each env in viewer
        
        Input: 
            - env (gymapi.Env)
            - push_contact (ContactPoint object)
        
        '''
        
        ## Draw contact points
        contact_points = push_contact.contact_points
        # Shift the contact points (2d) to 3d
        contact_points[:,:2] = 0.01
        
        lines = np.hstack([contact_points[0], contact_points[1]]).reshape(-1,3)
        color = np.array([0.0,0.0,1.0])
        colors = np.tile(color, (lines.shape[0], 1))
        
        self.gym.add_lines(self.viewer,self.envs[env_idx], len(lines), lines.astype(np.float32), colors.astype(np.float32))
        
        ## Draw slider edge
        edge_xyz = push_contact.edge_xyz
        edge_xyz[:,2] = 0.01
        color = np.array([1.0,1.0,0.0])
        colors = np.tile(color, (edge_xyz.shape[0], 1))
        
        self.gym.add_lines(self.viewer, self.envs[env_idx], len(edge_xyz), edge_xyz.astype(np.float32), colors.astype(np.float32))
        
    def visualize_contacts(self, push_contact_list):
        ''' 
        Visualize contact points in viewer
        
        Input: 
            - push_contact_list (ContactPoint object)
        
        '''
        
        a = int(np.ceil(np.sqrt(len(push_contact_list))))
        fig = plt.figure()
        for idx, push_contact in enumerate(push_contact_list):
            edge_xyz = push_contact.edge_xyz
            contacts_xy = push_contact.contact_points
            push_direction = push_contact.push_direction
            contact_center = contacts_xy.mean(axis=0)
            
            ax = fig.add_subplot(a, a, idx+1)
            ax.scatter(edge_xyz[:,0], edge_xyz[:,1], color = np.array([0.0,0.0,1.0]))
            # left point
            ax.scatter(contacts_xy[0,0] - push_direction[0] , contacts_xy[0,1] - push_direction[1] , color = np.array([1.0,0.0,0.0]))
            ax.scatter(contacts_xy[0,0]  , contacts_xy[0,1]  , color = np.array([1.0,0.0,0.0]))
            
            # right point
            ax.scatter(contacts_xy[1,0] - push_direction[0] , contacts_xy[1,1] - push_direction[1] , color = np.array([1.0,0.0,1.0]))
            ax.scatter(contacts_xy[1,0]  , contacts_xy[1,1]  , color = np.array([1.0,0.0,1.0]))
            # ax.set_xlabel('X Label')
            # ax.set_ylabel('Y Label')
            ax.set_aspect('equal', 'box')
            ax.set_title(f'Contact {idx}')
        plt.show()
    
    def plot_dof_states(self, waypoints_array, pos_dof_array, vel_dof_array, dt):
        '''
        Plot the movement of the pusher and compare with the target movement in all envs
        Inputs:
            - waypoints_array (num_envs, (n, 3)) : Target trajectory of waypoints of the pusher in all envs 
            - pos_dof_array (num_envs, n_timesteps, n_dofs) : Trajectory of positional dof states of the pusher in all envs
            - vel_dof_array (num_envs, n_timesteps, n_dofs) : Trajectory of velocity dof states of the pusher in all envs
            - dt (float) : Timestep
        '''
        # Plot the dof states
        fig = plt.figure(figsize=(10,10))
        for env_idx in range(self.num_envs):
            ax_dist = fig.add_subplot(self.num_envs, 2, 2*env_idx+1)
            ax_vel  = fig.add_subplot(self.num_envs, 2, 2*env_idx+2)
            planar_length_diff = np.linalg.norm(pos_dof_array[env_idx,1:,:2] - pos_dof_array[env_idx,:-1,:2], axis=1)
            planar_length_array = np.cumsum(planar_length_diff)
            planar_speed_array = planar_length_diff / dt
            
            waypoints = waypoints_array[env_idx]
            target_velocity = waypoints[1:] - waypoints[:-1]
            target_speed = np.linalg.norm(target_velocity[:,:2], axis=1)  / dt
            
            target_length_diff = np.linalg.norm(waypoints[1:,:2] - waypoints[:-1,:2], axis=1)
            target_length_array = np.cumsum(target_length_diff)
            
            ax_dist.plot(np.arange(0, len(target_length_array)*dt, dt), target_length_array)
            ax_dist.plot(np.arange(0, len(planar_length_array)*dt, dt), planar_length_array)
            ax_dist.set_xlabel('Time (s)')
            ax_dist.set_ylabel('Movement (m)')
            ax_dist.legend(['target_dist', 'dist'])
            
            ax_vel.plot(np.arange(0, len(target_speed)*dt, dt), target_speed)
            ax_vel.plot(np.arange(0, len(planar_speed_array)*dt, dt), planar_speed_array)
            ax_vel.legend(['target_speed', 'speed'])
            ax_dist.set_xlabel('Time (s)')
            ax_dist.set_ylabel('Movement (m)')
            
        plt.show()
            
    def log_dof_states(self, waypoint_idx):
        '''
        Log DOF states of the pusher in all envs
        Input:
                - waypoint_idx (int) : Index of the push trajectory waypoint
        '''
        
        for env_idx in range(self.num_envs):
            pusher_dof_states = self.gym.get_actor_dof_states(self.envs[env_idx], self.pusher_actor_handles[env_idx], gymapi.STATE_ALL)
            pos_dof_states = pusher_dof_states['pos']
            vel_dof_states = pusher_dof_states['vel']
            
            self.pos_dof_array[env_idx, waypoint_idx] = pos_dof_states
            self.vel_dof_array[env_idx, waypoint_idx] = vel_dof_states

    
if __name__ == "__main__":
    
    env = PushSim()
    # for i in tqdm(range(env.num_iters), desc = f'Object {env.slider_name}'):
    for i in range(env.num_iters):
        start_time = time.time()
        env.step_simulation()
        # print(f'Iteration {i} took {time.time() - start_time} seconds')
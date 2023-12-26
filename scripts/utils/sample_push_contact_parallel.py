import numpy as np
from multiprocessing import Pool
import multiprocessing
# from isaacgym import gymapi, gymutil, gymtorch
# from isaacgym.torch_utils import *
import yaml
from .contact_point_sampler import ContactPointSampler
import time
import parmap
from itertools import repeat


import  matplotlib.pyplot as plt

class SamplePushContactParallel:
    
    def __init__(self, num_envs, camera_intrinsic, gripper_width):
        self.intrinsic = camera_intrinsic
        self.gripper_width = gripper_width
        self.num_cores = multiprocessing.cpu_count()
        self.num_envs = num_envs
        
    @staticmethod
    def sample_push_contact(env_idx, depth_image, segmask, camera_pose, camera_intrinsic, gripper_width):

        cps = ContactPointSampler(camera_intrinsic, camera_pose, gripper_width)
        push_contacts = cps.sample(depth_image, segmask)
        selected_idx = np.random.randint(0,len(push_contacts))
        push_contact = push_contacts[selected_idx]

        
        '''
        Outputs as a dictionary with key as env_idx and value as push_contact
        because multiprocessing does not run in increasing order of env_idx
        and we need to sort the output list by env_idx
        '''
        return {env_idx : push_contact}
    
    def sample_push_contacts(self, depth_images, segmasks, camera_poses):
        push_contact_with_idx_list= parmap.starmap(self.sample_push_contact, 
                                                   list(zip(range(self.num_envs), depth_images, segmasks, camera_poses, repeat(self.intrinsic), repeat(self.gripper_width))),
                                                  pm_processes=self.num_cores, pm_chunksize = self.num_cores) #pm_processes=NUM_CORES, pm_pbar={"desc":"Sampling contact points..."}
        push_contact_with_idx_list = sorted(push_contact_with_idx_list, key=lambda x: list(x.keys())[0])
        push_contact_list = [list(contact_dict.values())[0] for contact_dict in push_contact_with_idx_list]
        
        return push_contact_list

            
if __name__ == "__main__":
    # Load configuation file
    config_file = "/home/hong/ws/twc-stable-pushnet/config/config_pushsim.yaml"
    
    with open(config_file,'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    sim_cfg = cfg["simulation"]
    cam_cfg = sim_cfg["camera"]["ZividTwo"]
    fx, fy, cx, cy = cam_cfg["fx"], cam_cfg["fy"], cam_cfg["cx"], cam_cfg["cy"]
    camera_intrinsic = np.array([[fx,0,cx],
                                        [0,fy,cy],
                                        [0, 0, 1]])
    
    gripper_width = sim_cfg["gripper_width"]
    
    depth_images, segmasks, camera_poses = [], [], []
    
    for env_idx in range(180):
        name = ("_%0" + str(6) + 'd.npy')%(env_idx)
        depth_image = np.load("/home/hong/ws/twc-stable-pushnet/src/data/depth_images/" + "depth_image" + name)
        segmask = np.load("/home/hong/ws/twc-stable-pushnet/src/data/depth_images/" + "segmask" + name)
        camera_pose = np.load("/home/hong/ws/twc-stable-pushnet/src/data/depth_images/" + "camera_pose" + name)
        
        depth_images.append(depth_image)
        segmasks.append(segmask)
        camera_poses.append(camera_pose)
        
    depth_images = np.array(depth_images)
    segmasks = np.array(segmasks)
    camera_poses = np.array(camera_poses)
    

import numpy as np
from scipy import ndimage
import cv2
import parmap
import multiprocessing
import matplotlib.pyplot as plt

def crop_image(depth_image, push_contact):
        ''' Convert the given depth image to the network image input
        
        1. Set the contact point to the center of the image
        2. Rotate the image so that the push direction is aligned with the x-axis
        3. Crop the image so that the object can be fit into the entire image
        4. Resize the image to the network input size (96 x 96)
        
        Args:
            depth_image (np.array): depth image
            push_contact (PushContact): push contact information
        
        '''
        
        image_height, image_width = 96, 96

        H,W = depth_image.shape
        contact_points_uv = push_contact.contact_points_uv
        edge_uv = push_contact.edge_uv
        edge_center_uv = edge_uv.mean(axis=0)
        
        '''
        contact_points_uv, edge_uv: [row, col] = [u,v]
        Image coordinate:           [row, col] = [v,u]
        '''
        
        contact_center_uv = contact_points_uv.mean(0).astype(int)
        
        edge_center_vu = np.array([edge_center_uv[1], edge_center_uv[0]])
        contact_center_vu = np.array([contact_center_uv[1], contact_center_uv[0]])
        
        ########################################################
        # Modify pushing direction to head to the -v direction #
        ########################################################
        u1, v1 = contact_points_uv[0]
        u2, v2 = contact_points_uv[1]
        push_dir = np.array([u1-u2,v2-v1])
        
        # Center of the rotated edge center should be in -v direction
        rot_rad = np.pi - np.arctan2(push_dir[1],push_dir[0])  
        R = np.array([[np.cos(rot_rad), -np.sin(rot_rad)], [np.sin(rot_rad),  np.cos(rot_rad)]])
        rotated_edge_center = R @ (edge_center_vu - contact_center_vu)
        rot_angle = np.rad2deg(rot_rad)
        
        if rotated_edge_center[0] > 0:
            rot_angle = 180 + rot_angle
            
        ###################################
        # Rotate and crop the depth image #
        ###################################
        
        # Shift the image so that the contact point is at the center
        shifted_img = ndimage.shift(depth_image, (np.round(H/2-contact_center_vu[0]).astype(int), np.round(W/2-contact_center_vu[1]).astype(int)), mode='nearest')
        
        # Rotate the image so that the pushing direction heads to -v direction
        rotated_img = ndimage.rotate(shifted_img, rot_angle, mode='nearest', reshape=False)
        
        
        # Crop the image so that the object can be fit into the entire image
        center_y, center_x = np.round(H/2).astype(int), np.round(W/2).astype(int)
        
        
        
        crop_size_unit = int(H/2/3)
        cropped_img = rotated_img[center_y - 3*crop_size_unit : center_y + crop_size_unit, center_x  - 2*crop_size_unit : center_x  + 2*crop_size_unit]
        
        # Resize the image to the network input size (96 x 96)
        cropped_img = cv2.resize(cropped_img, (image_height,image_width))
        # fig = plt.figure()
        # ax = fig.add_subplot(131)
        # ax.scatter(edge_vu[:,0], edge_vu[:,1])
        # ax.scatter(contact_center[0], contact_center[1], c='g')
        # ax.scatter(edge_center_vu[0], edge_center_vu[1], c='r')
        # ax.set_aspect('equal')
        # ax = fig.add_subplot(132)
        # ax.scatter(rotated_edge_vu[:,0], rotated_edge_vu[:,1])
        # ax.scatter(rotated_edge_center[0], rotated_edge_center[1], c='r')
        # ax.scatter(rotated_contact_center[0], rotated_contact_center[1], c='g')
        # ax.set_aspect('equal')
        # ax = fig.add_subplot(133)
        # ax.imshow(cropped_img)
        # plt.show()
        
        return cropped_img   
class CropImageParallel:
    def __init__(self, num_envs, camera_intrinsic, gripper_width):
        self.num_envs = num_envs
        self.intrinsic = camera_intrinsic
        self.gripper_width = gripper_width
        self.num_cores = multiprocessing.cpu_count()
    
    @staticmethod
    def crop_image_parallel(env_idx, depth_image, push_contact):
        
        cropped_image = crop_image(depth_image, push_contact)
        
        return {env_idx: cropped_image}
        
    def crop_images_parallel(self, depth_images, push_contact_list):
        cropped_image_with_idx_list= parmap.starmap(self.crop_image_parallel, list(zip(range(self.num_envs), depth_images, push_contact_list)), pm_processes=self.num_cores, pm_chunksize = self.num_cores)
        #pm_pbar={"desc":"Cropping images..."}
        cropped_image_with_idx_list = sorted(cropped_image_with_idx_list, key=lambda x: list(x.keys())[0])
        cropped_images = [list(cropped_image_with_idx.values())[0] for cropped_image_with_idx in cropped_image_with_idx_list]
        return cropped_images
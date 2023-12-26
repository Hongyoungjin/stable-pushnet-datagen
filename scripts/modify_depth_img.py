import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml
import os
import re
from utils.dataloader_parallel import DataLoaderParallel
import parmap
import multiprocessing


# Data Directories
DATA_DIR = "/home/hong/ws/twc-stable-pushnet/src/data"
tensor_dir = DATA_DIR + '/tensors'
save_dir = DATA_DIR + '/depth_hole_test/tensors'


with open('/home/hong/ws/twc-stable-pushnet/config/config_pushsim.yaml','r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
FILE_ZERO_PADDING_NUM = cfg['simulation']['FILE_ZERO_PADDING_NUM']

def restore_defected_image(depth_im):
    restored_depth_image = depth_im.copy()
    restored_depth_image = np.expand_dims(restored_depth_image, axis=-1)
    restored_depth_image = restored_depth_image.astype('float32')
    inpaint_mask = np.zeros(depth_im.shape, dtype='uint8')
    inpaint_mask[depth_im == 0] = 255
    restored_depth_image = cv2.inpaint(
        restored_depth_image,
        inpaint_mask,
        inpaintRadius=5,
        flags=cv2.INPAINT_NS
        )
    return restored_depth_image
    
def defect_depth_img(depth_im, binary_im):
    '''
    Intentionally make random depth holes from a given depth image
    
            Returns:
            `numpy.ndarray`: (H, W) with `float32` depth image.
    '''
    defected_depth_im = depth_im.copy()
    binary_im = binary_im.astype(np.uint8)

    # get object inner random pixel
    inner_px = np.where(binary_im != 0)
    if len(inner_px[0]) != 0:
        rand_idx = np.random.choice(np.arange(len(inner_px[0])), 1)[0]
        center_px = [inner_px[1][rand_idx], inner_px[0][rand_idx]]

        # get random covariance matrix RSR^-1
        sigma1 = np.random.uniform(2, 6)
        sigma2 = np.random.uniform(2, 6)
        angle = np.random.uniform(0.0, 2*np.pi)
        rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        diag_mat = np.array([[sigma1, 0.0], [0.0, sigma2]])
        random_cov = rot_mat*diag_mat*np.transpose(rot_mat)

        # get noise
        noise = np.random.multivariate_normal(center_px, random_cov, size=1000)

        # make depth hole on the object
        ZD_px = noise.astype('int32')
        ZD_px[ZD_px > 95] = 95
        ZD_px[ZD_px < 0] = 0
        defected_depth_im[ZD_px[:, 1], ZD_px[:, 0]] = 0

        # make depth hole on edge
        contours, hierarchy = cv2.findContours(
            binary_im,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
            )
        contours = np.concatenate(contours)

        # if the number of edge pixel is less then 2, ignore.
        if len(contours) > 1:
            edge_px = contours.squeeze()
            edge_len = len(edge_px)
            edge_num = int(np.random.uniform(5, 10))
            if edge_num < edge_len:
                start_idx = np.random.randint(edge_len - edge_num)
                defected_depth_im[edge_px[start_idx:start_idx+edge_num, 1], edge_px[start_idx:start_idx+edge_num, 0]] = 0
            else:
                defected_depth_im[edge_px[:, 1], edge_px[:, 0]] = 0
    else:
        print("no binary mask")

    return defected_depth_im

def get_binary_mask(segmented_depth):
    '''
    Get the binary mask from a given segmented depth image
    
    Segmented depth image: A depth image that is "elementalizely multiplied" with binary mask.
    
    Returns:
        `numpy.ndarray`: (H, W) with `float32` depth image.
    '''
    
    binary_mask = segmented_depth > 0

    return binary_mask


def run(idx):
    ''' Modify images and save them '''
    image_name = ("%s_%0" + str(FILE_ZERO_PADDING_NUM) + "d.npy")%("image", idx)
    masked_image_name = ("%s_%0" + str(FILE_ZERO_PADDING_NUM) + "d.npy")%("masked_image", idx)
    velocity_name = ("%s_%0" + str(FILE_ZERO_PADDING_NUM) + "d.npy")%("velocity", idx)
    label_name = ("%s_%0" + str(FILE_ZERO_PADDING_NUM) + "d.npy")%("label", idx)
    image = np.load(os.path.join(tensor_dir, image_name), allow_pickle=True)
    masked_image = np.load(os.path.join(tensor_dir, masked_image_name), allow_pickle=True)
    velocity = np.load(os.path.join(tensor_dir, velocity_name), allow_pickle=True)
    label = np.load(os.path.join(tensor_dir, label_name), allow_pickle=True)

    binary_mask = get_binary_mask(masked_image)
    defected_img = defect_depth_img(image, binary_mask)
    restored_img = restore_defected_image(defected_img)

    defected_masked_image = np.multiply(defected_img,binary_mask)
    restored_masked_image = np.multiply(restored_img,binary_mask)
    
    save_name = ("_%0" + str(FILE_ZERO_PADDING_NUM) + 'd.npy')%(idx)
    
    # Visualize images
    # visualize(masked_image, defected_masked_image, restored_masked_image)
    
    # Save data
    save_data(save_dir, save_name, masked_image, defected_masked_image, restored_masked_image, velocity, label)
           
def visualize(masked_image, defected_masked_image, restored_masked_image):
    "Visualize the modified images"
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.imshow(masked_image)
    ax = fig.add_subplot(312)
    ax.imshow(defected_masked_image)
    ax = fig.add_subplot(313)
    ax.imshow(restored_masked_image)
    plt.show()   
    
def save_data(save_dir, name, masked_image, defected_masked_image, restored_masked_image, velocity, label):
    
    with open(os.path.join(save_dir, 'masked_image' + name), 'wb') as f:
                            np.save(f, masked_image)
    
    with open(os.path.join(save_dir, 'defected_masked_image' + name), 'wb') as f:
                            np.save(f, defected_masked_image)
                            
    with open(os.path.join(save_dir, 'restored_masked_image' + name), 'wb') as f:
                            np.save(f, restored_masked_image)
    
    with open(os.path.join(save_dir, 'velocity' + name), 'wb') as f:
                            np.save(f, velocity)
                            
    with open(os.path.join(save_dir, 'label' + name), 'wb') as f:
                            np.save(f, label)
           

# load indicies
indices_file = os.path.join(DATA_DIR,'split','test_indices.npy')
indices = np.load(indices_file)
num_cores = multiprocessing.cpu_count()
parmap.map(run, indices, pm_processes=num_cores, pm_chunksize = num_cores, pm_pbar = {'desc':'Modifying...'})

# for idx in indices:
#     run(idx)
    



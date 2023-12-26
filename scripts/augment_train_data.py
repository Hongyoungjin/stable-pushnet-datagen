import numpy as np
import os
import yaml
import argparse
from tqdm import tqdm
from utils.push_dof_tools import get_maximum_file_idx
from utils.dataloader_parallel import DataLoaderParallel
import parmap


# Get current file path
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
assets_dir = os.path.join(current_directory, "assets/dish_urdf/")
config_dir = os.path.join(current_directory, "../config/config_pushsim.yaml")
train_data_dir = os.path.join(current_directory, "../data/tensors")


# Open config file
with open(config_dir,'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

num_zero_padding = cfg["simulation"]["FILE_ZERO_PADDING_NUM"]

file_list = os.listdir(train_data_dir)
file_list = [file for file in file_list if file.endswith('.npy')]

data_max_idx = get_maximum_file_idx(train_data_dir)
dataloader = DataLoaderParallel(data_max_idx, train_data_dir, num_zero_padding)


# Load training data
image_list = dataloader.load_image_tensor_parallel()
masked_image_list = dataloader.load_masked_image_tensor_parallel()
velocity_list = dataloader.load_velocity_tensor_parallel()
label_list = dataloader.load_label_tensor_parallel()


# Get file list by each field
file_list_image = [file for file in file_list if file.startswith('image')]
file_list_masked_image = [file for file in file_list if file.startswith('masked_image')]
file_list_velocity = [file for file in file_list if file.startswith('velocity')]
file_list_labels = [file for file in file_list if file.startswith('label')]
    
    
images = np.squeeze(np.array(image_list), axis=1)
masked_images = np.squeeze(np.array(masked_image_list), axis=1)
velocities = np.array(velocity_list)
labels = np.array(label_list)

# Flip train data
flipped_images = np.flip(images, axis=2)
flipped_masked_images = np.flip(masked_images, axis=2)
flipped_velocities = -velocities # flip velocity
flipped_velocities[:,0] *= -1 # keep x velocity the same


def save_data(idx):
    name = ("_%0" + str(num_zero_padding) + 'd.npy')%(data_max_idx + idx + 1)
    
    with open(os.path.join(train_data_dir, 'image' + name), 'wb') as f:
        np.save(f, flipped_images[idx])
        
    with open(os.path.join(train_data_dir, 'masked_image' + name), 'wb') as f:
        np.save(f, flipped_masked_images[idx])
        
    with open(os.path.join(train_data_dir, 'velocity' + name), 'wb') as f:
        np.save(f, flipped_velocities[idx])
        
    with open(os.path.join(train_data_dir, 'label' + name), 'wb') as f:
        np.save(f, labels[idx])

parmap.map(save_data, range(len(flipped_images)), pm_pbar={'desc': 'Saving flipped data'}, pm_processes=16, pm_chunksize=16)
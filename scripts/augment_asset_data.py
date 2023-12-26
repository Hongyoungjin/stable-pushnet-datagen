import trimesh
import numpy as np
import os
import glob
import xml.etree.cElementTree as ET
from multiprocessing import Pool
import copy
import argparse



'''
Rescale the mesh in x-y-z direction to augment the slider dataset.
Randomly extend or shrink the mesh in x-y-z direction.
'''

# Parse arguments
parser = argparse.ArgumentParser(description='This script augments Isaac Gym asset.')
parser.add_argument('--num', required=True, help='Number of new assets to be created per each asset.')
args = parser.parse_args()

num_new_mesh = args.num

# Get current file path
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
assets_dir = os.path.join(current_directory, "assets/dish_urdf/")


# mesh rescale prameters
GRIPPER_WIDTH = 0.08
GRIPPER_FRAC = 0.8
gripper_target = GRIPPER_WIDTH * GRIPPER_FRAC
maximum_dish_size = 0.32 # in meters
minimum_dish_size = 0.1 # in meters

urdf_ext = '.urdf'
obj_ext = '.obj'



def indent(elem, level=0, more_sibs=False):
    # https://stackoverflow.com/questions/749796/pretty-printing-xml-in-python
    i = "\n"
    if level:
        i += (level-1) * '  '
    num_kids = len(elem)
    if num_kids:
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
            if level:
                elem.text += '  '
        count = 0
        for kid in elem:
            indent(kid, level+1, count < num_kids - 1)
            count += 1
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
            if more_sibs:
                elem.tail += '  '
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
            if more_sibs:
                elem.tail += '  '


def obj_to_urdf(asset_name):
    
    asset_path = os.path.join(assets_dir, asset_name)
    mesh_path = os.path.join(asset_path, asset_name + obj_ext)
    
    # if target exists, skip
    print('processing: ', asset_name)
    
    # Load mesh
    try:
        mesh = trimesh.load(mesh_path)
    except Exception as e:
        mesh_path = os.path.join(asset_path, asset_name + '.stl')
        mesh = trimesh.load(mesh_path)
    
    # Load stable pose
    stable_pose = np.load(os.path.join(asset_path, 'stable_poses.npy'))
    
    # Constraint the modified mesh size to be smaller than the maximum dish size
    # translation_matrix = trimesh.transformations.translation_matrix(stable_pose[:3, 3])
    # mesh.apply_transform(translation_matrix)
    
    # rotation_matrix = trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, 1])
    mesh.apply_transform(stable_pose)
    
    x_size, y_size, z_size = mesh.extents
    max_x_scale_factor = maximum_dish_size / x_size
    max_y_scale_factor = maximum_dish_size / y_size
    max_z_scale_factor = maximum_dish_size / z_size
    
    min_x_scale_factor = minimum_dish_size / x_size
    min_y_scale_factor = minimum_dish_size / y_size
    min_z_scale_factor = minimum_dish_size / z_size
    
    if not mesh.is_watertight:
        print('{} is not watertight.'.format(asset_name))
    
    for i in range(num_new_mesh):
        
        target_name = asset_name + '_' + str(i)
        
        # make directory
        os.makedirs(os.path.join(assets_dir, target_name), exist_ok=True)
        new_asset_path = os.path.join(assets_dir, target_name)
        
        # Create a scaling transformation matrix
        scale_matrix = np.diag([np.random.choice(np.linspace(min_x_scale_factor,max_x_scale_factor,1000)), 
                                np.random.choice(np.linspace(min_y_scale_factor,max_y_scale_factor,1000)), 
                                # np.random.choice(np.linspace(min_z_scale_factor,max_z_scale_factor,1000)), 
                                1.0, 
                                1.0])

        # Apply the scaling transformation to the mesh vertices
        new_mesh = copy.deepcopy(mesh)
        new_mesh = trimesh.transformations.transform_points(new_mesh.vertices, scale_matrix)
        new_mesh = trimesh.Trimesh(vertices=new_mesh, faces=mesh.faces)
        
        
        # Check 
        x_size, y_size, _ = new_mesh.extents
        if x_size > maximum_dish_size or y_size > maximum_dish_size:
            print('The modified mesh size is larger than the maximum dish size.')
            continue
        
        mass = 0.050
        new_mesh.density = mass / new_mesh.volume
        
        new_mesh.export(os.path.join(new_asset_path, target_name + obj_ext))

        # Apply the scaling transformation to the stable pose
        new_stable_pose = stable_pose.copy()
        try:
            new_stable_pose[:3, 3] = trimesh.transformations.transform_points(stable_pose[:3, 3].reshape(-1,3), scale_matrix)
        except:
            print(f"Stable pose of {target_name} is not singular. ")

        new_stable_pose = np.eye(4)
        
        # Save the stable pose
        np.save(os.path.join(new_asset_path, 'stable_poses.npy'), new_stable_pose)

        # create urdf file
        urdf = ET.Element('robot', name=target_name)
        link = ET.SubElement(urdf, 'link', name=target_name)
        inertial = ET.SubElement(link, 'inertial')
        mass = ET.SubElement(inertial, 'mass', value=str(new_mesh.mass))
        inertia_dict = {'ixx': str(new_mesh.moment_inertia[0, 0]),
                        'ixy': str(new_mesh.moment_inertia[0, 1]),
                        'ixz': str(new_mesh.moment_inertia[0, 2]),
                        'iyy': str(new_mesh.moment_inertia[1, 1]),
                        'iyz': str(new_mesh.moment_inertia[1, 2]),
                        'izz': str(new_mesh.moment_inertia[2, 2])}
        inertia = ET.SubElement(inertial, 'inertia', inertia_dict)

        visual = ET.SubElement(link, 'visual')
        origin = ET.SubElement(visual, 'origin', xyz='0 0 0', rpy='0 0 0')
        geometry = ET.SubElement(visual, 'geometry')
        _mesh = ET.SubElement(geometry, 'mesh', filename=os.path.join(new_asset_path, target_name + obj_ext), scale='1 1 1')

        collision = ET.SubElement(link, 'collision')
        origin = ET.SubElement(collision, 'origin', xyz='0 0 0', rpy='0 0 0')
        geometry = ET.SubElement(collision, 'geometry')
        _mesh = ET.SubElement(geometry, 'mesh', filename=os.path.join(new_asset_path, target_name + obj_ext), scale='1 1 1')

        # save urdf file
        indent(urdf)
        
        tree = ET.ElementTree(urdf)
        
        with open(os.path.join(new_asset_path, target_name + urdf_ext), 'wb') as f:
            tree.write(f, encoding='utf-8', xml_declaration=True)
        
        
if __name__ == '__main__':
    # get file list
    asset_names = os.listdir(assets_dir)
    asset_names.sort()
    
    with Pool(8) as pool:
        pool.map(obj_to_urdf, asset_names)
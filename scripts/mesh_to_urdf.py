import os
import glob
import argparse
from itertools import repeat
import xml.etree.cElementTree as ET
from multiprocessing import Pool
import numpy as np
import trimesh
import parmap
import multiprocessing


# mesh rescale prameters
GRIPPER_WIDTH = 0.08
GRIPPER_FRAC = 0.8
gripper_target = GRIPPER_WIDTH * GRIPPER_FRAC

# Parse arguments
parser = argparse.ArgumentParser(description='This script converts mesh data to Isaac Gym asset.')
parser.add_argument('--root', required=True, help='Path to mesh folder')
parser.add_argument('--target', required=True, help='Path to asset folder')
parser.add_argument('--extension', required=True, help='Mesh extension (e.g. .obj, .stl, etc.)')
args = parser.parse_args()

mesh_root_dir = args.root
target_root_dir = args.target
obj_ext = args.extension

if not os.path.exists(target_root_dir):
    os.makedirs(target_root_dir)

def indent(elem, level=0, more_sibs=False):
    ''' Add indent when making URDF file'''
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


def obj_to_urdf(mesh_file):
    # if target exists, skip
    target_name = os.path.basename(mesh_file).split('.')[0]  # A0, A1, ...
    if os.path.exists(os.path.join(target_root_dir, target_name)):
        print('overide existing file for: ', target_name)

    print('processing: ', mesh_file)
    # Load mesh
    mesh = trimesh.load(os.path.join(mesh_file))

    if not mesh.is_watertight:
        print('{} is not watertight.'.format(mesh_file))

    # make directory
    os.makedirs(os.path.join(target_root_dir, target_name), exist_ok=True)

    # rescale mesh based on gripper width(EGAD paper)
    exts = mesh.bounding_box_oriented.primitive.extents
    max_dim = np.max(exts)
    scale = GRIPPER_WIDTH / max_dim
    # mesh.apply_scale(0.001) # mm to m scale
    mesh.apply_scale(scale) # mm to m scale

    mass = 0.050
    mesh.vertices -= mesh.center_mass
    mesh.density = mass/mesh.volume

    # save mesh
    mesh.export(os.path.join(target_root_dir, target_name, target_name + obj_ext))

    # create urdf file
    urdf = ET.Element('robot', name=target_name)
    link = ET.SubElement(urdf, 'link', name=target_name)
    inertial = ET.SubElement(link, 'inertial')
    mass = ET.SubElement(inertial, 'mass', value=str(mesh.mass))
    inertia_dict = {'ixx': str(mesh.moment_inertia[0, 0]),
                    'ixy': str(mesh.moment_inertia[0, 1]),
                    'ixz': str(mesh.moment_inertia[0, 2]),
                    'iyy': str(mesh.moment_inertia[1, 1]),
                    'iyz': str(mesh.moment_inertia[1, 2]),
                    'izz': str(mesh.moment_inertia[2, 2])}
    inertia = ET.SubElement(inertial, 'inertia', inertia_dict)

    visual = ET.SubElement(link, 'visual')
    origin = ET.SubElement(visual, 'origin', xyz='0 0 0', rpy='0 0 0')
    geometry = ET.SubElement(visual, 'geometry')
    _mesh = ET.SubElement(geometry, 'mesh', filename=os.path.join(target_root_dir, target_name, target_name + obj_ext), scale='1 1 1')

    collision = ET.SubElement(link, 'collision')
    origin = ET.SubElement(collision, 'origin', xyz='0 0 0', rpy='0 0 0')
    geometry = ET.SubElement(collision, 'geometry')
    _mesh = ET.SubElement(geometry, 'mesh', filename=os.path.join(target_root_dir, target_name, target_name + obj_ext), scale='1 1 1')

    # save urdf file
    indent(urdf)
    tree = ET.ElementTree(urdf)
    with open(os.path.join(target_root_dir, target_name, target_name + '.urdf'), 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)

    # get stable poses
    mesh.apply_scale(1000.0)
    stable_poses, prob = mesh.compute_stable_poses(n_samples=10, sigma=0.1)
    for i in range(len(stable_poses)):
        stable_poses[i][0:3, 3] *= 0.001

    np.save(os.path.join(target_root_dir, target_name, 'stable_poses.npy'), stable_poses)
    np.save(os.path.join(target_root_dir, target_name, 'stable_prob.npy'), prob)

    # save log as txt
    with open(os.path.join(target_root_dir, target_name, 'log.txt'), 'w') as f:
        f.write('num stable poses: {}\n'.format(len(stable_poses)))
        s = 'prob: '
        for i, p in enumerate(prob):
            s += '{:.3f}'.format(p)
            if i < len(prob) - 1:
                s += ', '
        s += '\n'
        f.write(s)


if __name__ == '__main__':
    # get file list
    obj_files = glob.glob(os.path.join(mesh_root_dir, '*' + obj_ext))
    obj_files.sort()

    with Pool(8) as pool:
        pool.map(obj_to_urdf, obj_files)
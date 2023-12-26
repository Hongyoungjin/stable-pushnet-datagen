import numpy as np
from scipy.spatial.transform import Rotation as R
import re
import os
from copy import copy
from .tools import Tmat2D

def tmat(pose):
    ''' Pose datatype conversion
    
    gymapi.Transform -> Homogeneous transformation matrix (4 x 4)
    
    '''
    t = np.eye(4)
    t[0, 3], t[1, 3], t[2, 3] = pose.p.x, pose.p.y, pose.p.z
    quat = np.array([pose.r.x, pose.r.y, pose.r.z, pose.r.w])
    t[:3,:3] = R.from_quat(quat).as_matrix()
    return t
 
def fibonacci_sphere(samples=2000):

    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples//2):
        x = 1 - (i / float(samples - 1)) * 2  # x goes from 1 to -1
        radius = np.sqrt(1 - x * x)  # radius at x

        theta = phi * i  # golden angle increment

        y = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))
    velocity = np.array(points)
    # with open("/home/hong/ws/pushnet/data/velocity.npy", "wb") as f:
    #     np.save(f, velocity)
        
    return velocity

def velocities2icrs(velocity):
    """
    Calculate ICR (Instantaneous Center of Rotation) for each velocity.
    """
    vx, vy, w = velocity[:,0], velocity[:,1], velocity[:,2]
    ICRs = []
    for i in range(len(vx)):
        if w[i] == 0:
            w[i] = 1e-6
        icr= np.array([-vy[i] / w[i], vx[i] / w[i]])
        ICRs.append(icr)
        
    return np.array(ICRs)

def fit_2d_gaussian(icrs):
    """
    Fit a 2D Gaussian to the ICRs.
    """
    mu = np.mean(icrs, axis=0)
    sigma = np.cov(icrs.T)
    return mu, sigma    

velocities_fibonacci = fibonacci_sphere()
icrs_fibonacci = velocities2icrs(velocities_fibonacci)
# mu, sigma = fit_2d_gaussian(icrs)
   
def get_random_icrs_old(num_envs):
    ''' Get random ICR in the gripper frame for all envs
    
    Output: ICR {end-effector} (n_envs, (x, y))
    '''
    
    # ICR positions
    one_tenths = np.linspace(0.1, 1, 100)
    ones       = np.linspace(1  , 10, 100)
    tens       = np.linspace(10 , 100, 60)
    hundreds   = np.linspace(100, 1000, 40)
    
    # positions_positive = np.concatenate([one_tenths, ones, tens, hundreds])
    positions_positive = np.concatenate([one_tenths, ones])
    positions_negative = -positions_positive
    positions = np.concatenate([positions_positive, positions_negative])
    
    # Randomly sample an ICR for all envs
    x,y = np.copy(positions_positive), np.copy(positions)
    np.random.shuffle(x)
    np.random.shuffle(y)
    
    icrs = np.array([x[:num_envs], y[:num_envs]]).T
    
    return icrs

def get_random_icrs(num_envs):
    ''' Get random ICR in the gripper frame for all envs
    
    Output: ICR {end-effector} (n_envs, (x, y))
    '''
    
    # # ICR positions
    # velocities = fibonacci_sphere()
    # icrs = velocities2icrs(velocities)
    
    # 2d gaussian fit for icrs in fibonacci_sphere(2000)
    # mu = np.array([10.08706798, 10.20278399])
    # sigma = np.array([[100499.88346488, 100178.28872892], [100178.28872892, 100341.96510333]])

    # Randomly sample an ICR for all envs
    # selected_icrs = np.random.Generator.multivariate_normal(mean, cov, num_envs)
    # selected_icrs = np.random.multivariate_normal(mu, sigma, num_envs)
    
    
    velocities_fibonacci = fibonacci_sphere()
    icrs_fibonacci = velocities2icrs(velocities_fibonacci)
    np.random.shuffle(icrs_fibonacci)
    selected_icrs = icrs_fibonacci[:num_envs]
    
    # print(selected_icrs)
    return selected_icrs

def icrs2directions(num_envs, icrs):
    '''  ICRs -> Unit pushing directions {end-effector}

    Input: ICR in gripper frame (num_envs, (x, y))
    Output: Unit pushing direction in gripper frame (num_envs, (Vx, Vy, w))

    '''
    # Convert to unit pushing directions for all envs
    xs, ys = icrs[:,0], icrs[:,1]
    directions = np.vstack((ys, -xs, np.ones(num_envs))).T

    # Make sure the x component of each direction is positive
    directions_positive_x = np.where(directions[0] < 0, -directions, directions)
    directions_magnitude = np.linalg.norm(directions_positive_x, axis=1)
    directions_final = directions_positive_x / directions_magnitude[:, np.newaxis]

    return directions_final

def direction2icr(direction):
    ''' Convert unit pushing direction to ICR
    Input: Unit pushing direction in gripper frame (Vx, Vy, w)
    Output: ICR in gripper frame (x, y)
    '''
    
    Vx, Vy, w = direction[0], direction[1], direction[2]
    
    if w ==0:
        return np.array([np.inf, np.inf])
    
    else:
        x = -Vy/w
        y =  Vx/w
        return np.array([x, y])
    
def icrs2trajectories(icrs, approach_distance, push_speed, dt):

    ''' Convert ICRs to trajectories with constant ICR for all envs
    
    Input: 
    
        - ICRs in gripper frame (x, y)
        - approach_distance (float): Initial approach distance in meters (identical for all envs)
        - push_speed (float): Pushing speed in meters per second (identical for all envs)
        - dt (float): Time step in seconds (identical for all envs)
        
        
    Output: Trajectory in gripper frame (n, (x, y, theta))
    
    Note:
        Resultant trajectory will be connected with approach trajectory
        => start of this trajectory == end of approach trajectory
    '''
    
    push_distance = 0.05 # m, 0.05*pi
    # push_distance = 0.628 # m, 0.2*pi
    
    push_time = push_distance / push_speed
    push_timesteps = int(push_time / dt)
    
    
    joint_trajectories = []
    for icr in icrs:
        
        x0, y0 = icr[0], icr[1]
        alpha = np.arctan2(-y0,-x0)
        r = np.sqrt(x0**2 + y0**2)
        
        # Left - hand side ICR
        theta = np.linspace(alpha, alpha + push_distance / r, push_timesteps)

        # Right - hand side ICR
        if y0 < 0:
            theta = np.linspace(alpha, alpha - push_distance / r, push_timesteps)
            
        x, y = x0 + r*np.cos(theta), y0 + r*np.sin(theta)
        
        # Minimize initial waypoint position error
        init_err_x, init_err_y = x[0], y[0]
        x, y = x - init_err_x, y - init_err_y
        
        # Start of this trajectory == end of approach trajectory
        x += approach_distance
        
        joint_trajectory = np.vstack((x,y,theta-alpha)).T
        joint_trajectories.append(joint_trajectory)
        
    joint_trajectories = np.array(joint_trajectories)
    
    return joint_trajectories

def icrs2trajectories_vel(icrs, push_distance, push_speed, dt):
    ''' Convert ICRs to velocity trajectories with constant ICR for all envs
    
    Input: 
    
        - ICRs in gripper frame (x, y)
        - push_distance (float): Total push distance in meters (identical for all envs)
        - push_speed (float): Pushing speed in meters per second (identical for all envs)
        - dt (float): Time step in seconds (identical for all envs)
        
        
    Output: Trajectory in gripper frame (n, (x, y, theta))
    
    Note:
        Resultant trajectory will be connected with approach trajectory
        => start of this trajectory == end of approach trajectory
    '''
    
    push_time = push_distance / push_speed
    push_timesteps = int(push_time/dt)
    
    joint_trajectories = []
    
    for icr in icrs:
        
        x0, y0 = icr[0], icr[1]
        alpha = np.arctan2(-y0,-x0)
        r = np.sqrt(x0**2 + y0**2)
        
        theta = np.linspace(alpha, alpha + 0.05 * np.pi / r, push_timesteps)
        
        omega = push_speed / r
        
        
        
        
        # Right - hand side ICR
        if y0 < 0:
            theta = np.linspace(alpha, alpha - 0.05 * np.pi / r, push_timesteps)
            
        x, y = x0 + r*np.cos(theta), y0 + r*np.sin(theta)
        
        # Minimize initial waypoint position error
        init_err_x, init_err_y = x[0], y[0]
        x, y = x - init_err_x, y - init_err_y
        
        joint_trajectory = np.vstack((x,y,theta-alpha)).T
        joint_trajectories.append(joint_trajectory)
        
    joint_trajectories = np.array(joint_trajectories)
    
    return joint_trajectories

def evaluate_push_stability(eef_slider_pose_initial, eef_slider_pose_final, threshold_pos, threshold_rot):
    ''' Derive push results by comparing the slider poses in the pusher's end-effector frame before and after pushing. (for all envs)
        
    Input: 
        - eef_slider_pose_initial (num_envs, 4x4)
        - eef_slider_pose_final (num_envs, 4x4)
        - threshold_pos: Positional threshold for the push result.(default: 0.01 meters)
        - threshold_rot: Rotational threshold for the push result.(default: 5 degrees)
        
        If the difference between the initial and final slider pose is smaller than the threshold, the push is deemed successful.
            
    Output: labels (num_envs, )
        - 0: Fail
        - 1: Successful
        
    Explanation for each filter:
    
        - Positional Filtering: 
            - x, y, z: Deviation from the initial position
            
        - Rotational Filtering:
            - rx, ry: Stumping (tripping)
            - rz: Rolling
    
    '''
    
    poses_init, poses_final = eef_slider_pose_initial, eef_slider_pose_final
    poses_init_inv = np.linalg.inv(poses_init)
    
    # Get the relative poses between the initial and final slider poses
    rel_poses = np.einsum('...ij,...jk->...ik', poses_init_inv, poses_final)
    
    rel_positions = rel_poses[:,:3,3]
    rel_rotations_euler = R.from_matrix(rel_poses[:,:3,:3]).as_euler('xyz', degrees=True)
    
    # Label the results for all envs with the given threshold
    labels = []
    for i in range(len(rel_positions)):
        is_close_position = np.all(np.abs(rel_positions[i]) < threshold_pos)
        is_close_rotation = np.all(np.abs(rel_rotations_euler[i]) < threshold_rot)
        
        if is_close_position and is_close_rotation:
            labels.append(1)
        else:
            labels.append(0)
        
    return labels
    
def get_approach_trajectories(approach_distance, push_speed, dt, num_envs):
    ''' Get the approach joint trajectories for all envs 
    
    Input: 
        - push_offsets: the offsets of the pusher from the slider (start of the path)
    
    Output: 
        - trajectories: the approach trajectories for all envs (shape: (num_envs, approach_distance / dt, 3))
    '''
    approach_time = approach_distance / push_speed
    approach_timesteps = int(approach_time / dt)
    # Get approach joint trajectories (Assume approach in x-direction {end-effector frame})
    waypoints_x = np.linspace(0, approach_distance, approach_timesteps)
    others      = np.zeros_like(waypoints_x)
    trajectory  = np.vstack([waypoints_x, others, others]).T
    trajectories = np.tile(trajectory,(num_envs,1)).reshape(num_envs, -1, 3)
    
    return trajectories

def get_maximum_file_idx(path):
    '''
    Get maximum file index in a directory
    - path (str): directory path
    
    This function is used for saving train data files in a 'self.save_dir' directory.
    
    It assumes that all files in the directory are named as 'file_{index}.npy' with 5 zero paddings.
    
    '''
    
    file_list = os.listdir(path)
    
    if len(file_list) == 0:
        return -1
    
    file_list = [file for file in file_list if file.endswith('.npy')]
    file_list = [file for file in file_list if file.startswith('image')]
    
    # Extract numbers and delete zero padding from file names 
    numbers = [int(re.search(r'(\d+)(\.\d+)?', file).group()) for file in file_list]
    maximum_number = np.max(numbers)
    
    return maximum_number

if __name__ == '__main__':
    maximum_number = get_maximum_file_idx("/home/hong/ws/pushnet/data/tensors")
    print(maximum_number)

# def perturabte_initial_contact(initial_contact_pose, rand_position_range, rand_rotation_range):
#     ''' For uncertainty in camera calibration, we apply a small randomness to the initial contact pose.

#     Input: 
#         - initial_contact_pose: SE(2) contact pose (x,y,theta)
#         - rand_position_range: random positional range of the contact pose (float)
#         - rand_rotation_range: rnadom rotation range of the contact pose (float)
    
#     Returns: 
#         - perturbated_contact_pose: SE(2) contact pose (x,y,theta)

#     '''
    
#     # Get inital pose in homogeneous form
#     x,y,theta = initial_contact_pose
#     z=0 # Assuming SE(2) contact pose
    
#     init_trans = np.eye(4)
#     init_trans[:3,:3] = R.from_euler('z', theta, degrees=False).as_matrix()
#     init_trans[:3,3] = np.array([x,y,z])


#     # Get random variables
#     r_random = np.random.uniform(-rand_rotation_range, rand_rotation_range)
#     p_random_x = np.random.uniform(-rand_position_range, rand_position_range)
#     p_random_y = np.random.uniform(-rand_position_range, rand_position_range)
    
#     # Get transformation matrix
#     rand_t = np.eye(4)
#     rand_t[:3,:3] = R.from_euler('z', r_random, degrees=True).as_matrix()
#     rand_t[:3,3]  = np.array([p_random_x,p_random_y,z])
    
#     # Perturbate.
#     perturbated_trans = rand_t @ init_trans
#     perturbated_theta = R.from_matrix(perturbated_trans[:3,:3]).as_euler('zxy', degrees=False)[0]
#     perturbated_xy = perturbated_trans[:2,3]
    
#     perturbated_contact_pose = []
#     perturbated_contact_pose.append(perturbated_xy[0])
#     perturbated_contact_pose.append(perturbated_xy[1])
#     perturbated_contact_pose.append(perturbated_theta)
    
#     return np.array(perturbated_contact_pose)

def perturabte_initial_contact(initial_contact_pose, rand_position_range, rand_rotation_range):
    ''' For uncertainty in camera calibration, we apply a small randomness to the initial contact pose.

    Input: 
        - initial_contact_pose: SE(2) contact pose (x,y,theta, in meters and radians)
        - rand_position_range: random positional range of the contact pose (float, in meters)
        - rand_rotation_range: rnadom rotation range of the contact pose (float, in radians)
    
    Returns: 
        - perturbated_contact_pose: SE(2) contact pose (x,y,theta)

    '''
    
    # Get inital pose in homogeneous form
    x,y,theta = initial_contact_pose
    init_t = Tmat2D(x,y,theta)
    init_t = np.array([x,y,1]).reshape(-1,1)


    # Get random transformation matrix
    t_rand = np.random.uniform(-rand_rotation_range, rand_rotation_range)
    x_rand = np.random.uniform(-rand_position_range, rand_position_range)
    y_rand = np.random.uniform(-rand_position_range, rand_position_range)
    
    rand_t = Tmat2D(x_rand, y_rand, t_rand)
    
    # Perturbate.
    pert_t = rand_t @ init_t
    pert_theta = theta + t_rand
    # sin, cos = pert_t[1,0], pert_t[1,1]
    # pert_theta = np.arctan2(sin,cos)[np.newaxis]
    # pert_xy = pert_t[:2,2]
    
    # print(x_rand, y_rand, np.rad2deg(t_rand))
    # print(x - pert_t[0,0], y - pert_t[1,0], theta - pert_theta)
    
    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(111)
    # ax.arrow(x, y, np.cos(theta), np.sin(theta), width=0.01, color='r')
    # ax.arrow(x_rand, y_rand, np.cos(t_rand), np.sin(t_rand), width=0.01, color='b')
    # ax.arrow(pert_t[0], pert_t[1], np.cos(pert_theta), np.sin(pert_theta), width=0.01, color='g')
    # ax.set_aspect('equal')
    # plt.show()
    return np.array([pert_t[0,0], pert_t[1,0], pert_theta])
    
    
    
    
    

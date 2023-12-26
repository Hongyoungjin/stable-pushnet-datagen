import numpy as np
import cv2
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial.transform import Rotation as R
import trimesh
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d


class ContactPoint(object):
    def __init__(self, edge_xyz, edge_uv, contact_points, contact_points_uv, push_direction):
        """Push contact point.
        Args:
            edge_xyz (numpy.ndarray): (N, 3) array of edge points in world frame.
            edge_uv (numpy.ndarray): (N, 2) array of edge points in image coordinates.
            contact_points (numpy.ndarray): (N, 2, 3) contact points in world frame.
            contact_points_uv (numpy.ndarray): (N, 2,2) contact points in image coordinates.
            push_direction (numpy.ndarray): (2,) array of pushing direction in world frame.
        """
        self.edge_xyz = edge_xyz
        self.edge_uv = edge_uv
        self.contact_points = contact_points
        self.contact_points_uv = contact_points_uv
        self.push_direction = push_direction

    @property
    def contact_normals(self):
        """Set contact normals.
        Args:
            contact_normals (numpy.ndarray): (2,2) contact normals in world frame.
        """
        ch = ConvexHull(self.edge_xyz[:, :2])
        ch_xy = self.edge_xyz[ch.vertices][:, :2]
        surface_normal1 = self._get_surface_normal(ch_xy, self.contact_points[0])
        surface_normal2 = self._get_surface_normal(ch_xy, self.contact_points[1])
        return np.array([surface_normal1, surface_normal2])

    @property
    def pose(self):
        """Position in world frame.
        Returns:
            pose (numpy.ndarray): Position (x, y theta) in world frame.
        """
        position = self.contact_points.mean(0)
        orientation = np.arctan2(self.push_direction[1], self.push_direction[0])
        return np.array([position[0], position[1], orientation])

    @staticmethod
    def _get_surface_normal(convex_hull, contact_point):
        """
        Get a surface normal from convex hull and a contact point.
        
        Args:
            convex_hull (numpy.ndarray): shape (N, 2)
            contact_point (numpy.ndarray): shape (3, )
            
        Returns:
            surface_normal (numpy.ndarray): shape (2, )
        """
        dist = np.linalg.norm(contact_point[:2] - convex_hull, axis=-1)
        candidate_index = np.argsort(dist)[:2]
        
        if np.abs(candidate_index[0] - candidate_index[1]) < (len(convex_hull) - 1):
            edge_point1 = convex_hull[np.min(candidate_index)]
            edge_point2 = convex_hull[np.max(candidate_index)]
        else:
            edge_point1 = convex_hull[np.max(candidate_index)]
            edge_point2 = convex_hull[np.min(candidate_index)]

        surface_normal = edge_point2 - edge_point1
        surface_normal = surface_normal / np.linalg.norm(surface_normal)
        surface_normal = np.array([-surface_normal[1], surface_normal[0]])
        return surface_normal
    
    def visualize_on_image(self, image):
        contact_points_uv = self.contact_points_uv.reshape(-1,2)
        contact_point1 = contact_points_uv[0]
        contact_point2 = contact_points_uv[1]
        center = np.mean(contact_points_uv, axis=0)
        
        push_direcion = contact_point2 - contact_point1
        push_direcion = np.array([push_direcion[1], -push_direcion[0]], dtype=np.float64)
        push_direcion /= np.linalg.norm(push_direcion)

        # visualize contact point
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image, cmap='gray')
        ax.scatter(
            contact_point1[0],
            contact_point1[1], c='g', marker='o')
        ax.scatter(
            contact_point2[0],
            contact_point2[1],
            c='r', marker='o')
        ax.arrow(
            center[0], center[1],
            push_direcion[0] * 50,
            push_direcion[1] * 50,
            color='b', head_width=10, head_length=10)
        plt.show()

    def visualize_on_cartesian(self):
        # get random 1000 point on edge_xy
        rand_idx = np.random.randint(0, len(self.edge_xyz), 1000)
        
        # fig = plt.figure()
        # # plot edge points
        # plt.plot(self.edge_xy[rand_idx, 0], self.edge_xy[rand_idx, 1], 'ko')
        # # plot contact points
        # plt.plot(self.contact_points[0, 0], self.contact_points[0, 1], 'go')
        # plt.plot(self.contact_points[1, 0], self.contact_points[1, 1], 'ro')
        # # plot push direction
        # plt.arrow(
        #     self.pose[0], self.pose[1],
        #     self.push_direction[0] * 0.05, self.push_direction[1] * 0.05,
        #     color='b', head_width=0.01, head_length=0.01)
        # # plot surface normal
        # plt.arrow(
        #     self.contact_points[0, 0], self.contact_points[0, 1],
        #     self.contact_normals[0, 0] * 0.05, self.contact_normals[0, 1] * 0.05,
        #     color='r', head_width=0.01, head_length=0.01)   
        # plt.arrow(
        #     self.contact_points[1, 0], self.contact_points[1, 1],
        #     self.contact_normals[1, 0] * 0.05, self.contact_normals[1, 1] * 0.05,
        #     color='r', head_width=0.01, head_length=0.01)   
        # plt.axis('equal')
        # plt.show()   
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.edge_xyz[rand_idx, 0], self.edge_xyz[rand_idx, 1], self.edge_xyz[rand_idx, 2], c='k', marker='o')
        ax.scatter(self.contact_points[0, 0], self.contact_points[0, 1], self.contact_points[0, 2], c='g', marker='o')
        ax.scatter(self.contact_points[1, 0], self.contact_points[1, 1], self.contact_points[1, 2], c='r', marker='o')
        ax.set_box_aspect((1,1,1))
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()
        
class ContactPointSampler(object):
    '''
    Samples contact points from a depth image
    Depth image -> Contact points 
    
    '''
    def __init__(self, camera_intr, camera_extr, gripper_width=0.08, num_push_dirs=8):
        self.camera_intr = camera_intr
        self.camera_extr = camera_extr
        self.gripper_width = gripper_width
        self.num_push_dirs = num_push_dirs
        self._width_error_threshold = 1e-5
        
    def sample(self, depth_image, mask):
        edge_list_uv, edge_list_xyz = self.edge_list_using_pcd(depth_image, mask, self.camera_extr, self.camera_intr)
        contact_pair_uv, contact_pair_xyz = self.get_contact_points(edge_list_uv, edge_list_xyz)
        edge_center = edge_list_xyz.mean(0)
        contact_pair_centers = contact_pair_xyz.mean(1)
        
        # Calculate pushing directions
        pushing_directions = contact_pair_xyz[:,0] - contact_pair_xyz[:,1]
        pushing_directions = pushing_directions[:,:2]
        pushing_directions = np.roll(pushing_directions, 1, axis=1)
        pushing_directions[:,1] *= -1
        
        edge_list_xy = edge_list_xyz[:, :2]
        contact_pair_centers_xy = contact_pair_centers[:, :2]
        
        
        projections_contact_point_centers = np.einsum('ij,ij->i', pushing_directions, contact_pair_centers_xy).reshape(-1, 1)
        projections_edges = np.einsum('ij,kj -> ik', pushing_directions, edge_list_xy)
        projections_edges_min_max = np.vstack([projections_edges.min(1), projections_edges.max(1)]).T
        projections_edges_median = projections_edges_min_max.mean(1).reshape(-1,1)
        
        pushing_directions = np.where(projections_edges_median > projections_contact_point_centers, pushing_directions, -pushing_directions)
                
        # fig = plt.figure(figsize=(10,10))
        # ax = fig.add_subplot(111)
        # ax.scatter(edge_list_xyz[:,0], edge_list_xyz[:,1], c='k', marker='o')
        # ax.scatter(contact_pair_xyz[0,0,0], contact_pair_xyz[0,0,1], c='b', marker='o')
        # ax.scatter(contact_pair_xyz[0,1,0], contact_pair_xyz[0,1,1], c='g', marker='o')
        # ax.arrow(contact_pair_centers_xy[0,0], contact_pair_centers_xy[0,1], pushing_directions[0,0], pushing_directions[0,1], width=0.001, color='r')
        # ax.set_aspect('equal')
        # plt.show()
        
        # ################
        # # Original way #   
        # ################
        
        # reference_directions = edge_center - contact_pair_centers
        
        # for i in range(len(reference_directions)):
        #     if np.dot(reference_directions[i], pushing_directions[i]) < 0:
        #         pushing_directions[i] *= -1
                
        contact_pair_angles = np.rad2deg(np.arctan2(pushing_directions[:,1], pushing_directions[:,0]))
        # sorted_indices = np.where(np.abs(-150 - contact_pair_angles) < 10)[0]
        
        sorted_indices = np.argsort(contact_pair_angles)
        
        step = sorted_indices.shape[0] // self.num_push_dirs

        try:
            sorted_indices = sorted_indices[::step]
        except:
            print(f'Error: number of contact points is less than {self.num_push_dirs}')
            pass
            
        contact_points = []
        
        for idx in sorted_indices:
            
            contact_points.append(ContactPoint(edge_list_xyz, edge_list_uv, contact_pair_xyz[idx], contact_pair_uv[idx], pushing_directions[idx]))
            
        return contact_points
        
    def edge_detection(self, depth_image, segmask):
        """
        Detect edges in depth image and filter them with segmask
        Args:
            depth_image (numpy.ndarray): (H, W) depth image.
            segmask (numpy.ndarray): (H, W) segmentation mask.
        Returns:
            edge_image (numpy.ndarray): (H, W) edge image.
        """
        self.depth_image = depth_image
        self.mask = segmask
        # https://stackoverflow.com/questions/38094594/detect-approximately-objects-on-depth-map

        # Unify scene depth for easy scene removal
        depth_img = np.multiply(depth_image, segmask)
        depth_img -= np.min(depth_img)
        depth_img = depth_img * 255/np.max(depth_img)
        depth_img = depth_img.astype(np.uint8)

        # binary threshold
        _ ,th = cv2.threshold(depth_img,0,255,1)
        
        # morphological close operation to fill the gaps in the image
        kernel = np.ones((15,15),np.uint8)
        dilate = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, 3)
        
        smooth = cv2.medianBlur(dilate,3)
        
        contours, _ = cv2.findContours(smooth,2,1)
        
        # Derive the outer contour as the edge of the object
        edges = np.zeros(depth_img.shape,dtype=np.uint8)
        for contour in contours:
            contour = contour.squeeze(1)
        edges[contour.T[1],contour.T[0]] = 1
        
        
        return edges 
    
    def canny_edge_detection(self, depth_image,segmask):
        
        # Unify scene depth for easy scene removal
        self.depth_image = depth_image
        self.mask = segmask
        
        kernel = np.ones((5, 5), np.uint8)
        
        img_erosion = cv2.erode(segmask, kernel, iterations=1)
        img_dilation = cv2.dilate(segmask, kernel, iterations=1)
        
        
        # fig = plt.figure()
        # ax = fig.add_subplot(221)
        # ax.imshow(depth_image)
        # ax = fig.add_subplot(222)
        # ax.imshow(segmask)
        # ax = fig.add_subplot(223)
        # ax.imshow(img_erosion)
        # ax = fig.add_subplot(224)
        # ax.imshow(img_dilation)
        
        # np.save("/home/cloudrobot2/catkin_ws/src/push_planners/twc-stable-pushnet/src/data/image_depth", self.depth_image)
        # np.save("/home/cloudrobot2/catkin_ws/src/push_planners/twc-stable-pushnet/src/data/image_mask", self.mask)
        # np.save("/home/cloudrobot2/catkin_ws/src/push_planners/twc-stable-pushnet/src/data/image_erosion", img_erosion)
        # np.save("/home/cloudrobot2/catkin_ws/src/push_planners/twc-stable-pushnet/src/data/image_dilation", img_dilation)
        
            
        # plt.show()
        # exit()
        
        depth_image = np.multiply(depth_image, img_erosion)
        depth_image -= np.min(depth_image)
        depth_image = depth_image * 255/np.max(depth_image)
        depth_image = depth_image.astype(np.uint8)
        
        # Blur the image for better edge detection
        img_blur = cv2.GaussianBlur(depth_image, (15,15), 0)

        # Canny Edge Detection
        edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=200)
        edges = np.where(edges == 255, 1, 0)
        edges = edges.astype(np.uint8)
        
        return edges
    
    def edge_list(self, depth_image, edge_image):
        """Get surface normal using gradient
        Args:
            edge (numpy.ndarray): (H, W) edge image. 0-background 1-edge.
        Returns:
            Edge normals per edge pixel coordinates (y,x,ny,nx)  [y,x] == [v,u] == [row,col] == [height,width]
        """
        edge_pixel_y, edge_pixel_x = np.where(edge_image == 1)

        # edges uv -> xy coordinates
        pcd = self.depth_to_pcd(depth_image, self.camera_intr)
        pcd_w = (np.matmul(self.camera_extr[:3,:3], pcd[:,:3].T) + self.camera_extr[:3,3].reshape(3,1)).T
        
        # edge_list_xy = pcd_w[np.ravel_multi_index((edge_pixel_y, edge_pixel_x), self.depth_image.shape)][:,:2]
        edge_list_xyz = pcd_w[np.ravel_multi_index((edge_pixel_y, edge_pixel_x), self.depth_image.shape)]
        edge_list_uv = np.c_[edge_pixel_y, edge_pixel_x]
        
        
        
        
        #######################
        #  KMeans Clustering ##
        #######################
        
        
        # Group edge points into two clusters based on height
        pcd_heights = edge_list_xyz[:,2]
        db = KMeans(n_clusters=3).fit(pcd_heights.reshape(-1,1))
        labels = db.labels_
        
       
        groups = []
        for i in range(len(set(labels))):
            groups.append(np.mean(pcd_heights[np.where(labels == i)[0]]))
        groups = np.array(groups)
        index = np.argsort(groups)
        median_group = index[0]
        
        indices = np.where(labels == median_group)[0]
        
        edge_list_xyz = edge_list_xyz[indices]
        edge_list_uv = edge_list_uv[indices]
        
        trimesh_pcd_w = trimesh.points.PointCloud(self.pcd_w, colors=[0,0,0,50])
        trimesh_pcd = trimesh.points.PointCloud(edge_list_xyz, colors=[0,255,0,255])
        base_frame = trimesh.creation.axis(origin_size=0.01)
        
        scene = trimesh.Scene([trimesh_pcd_w, trimesh_pcd, base_frame])
        scene.show()
        
        ########################
        ##  DBSCAN Clustering ##
        ########################
        # edge_list_xyz = np.nan_to_num(edge_list_xyz)
        # eps = 0.01
        # num_increase, num_decrease = 0, 0
        # while True: 
        #     db = DBSCAN(eps = eps, min_samples=10).fit(edge_list_xyz)
        #     labels = db.labels_
        #     print(len(set(labels)))
        #     if len(set(labels)) < 2:
        #         eps-=0.001
        #         num_decrease += 1
        #     elif len(set(labels)) > 2:
        #         eps+=0.001
        #         num_increase += 1
        #     if num_decrease > 1 and num_increase > 1 :
        #         break
        #     if len(set(labels)) == 2:
        #         break
        
        
        # print(labels)
        # cluster_heights = []
        # heights = edge_list_xyz[:,:2]
        # for i in range(len(set(labels))):
        #     cluster_heights.append(np.mean(heights[np.where(labels == i)[0]]))
        # print(cluster_heights)
        # # Select the cluster with the highest mean height
        # index = np.argmax(cluster_heights)
        # cluster_indices = np.where(labels == index)[0]
        # print(cluster_indices)
        # edge_list_xyz = edge_list_xyz[cluster_indices]
        # print(edge_list_xyz)
        # edge_list_uv = edge_list_uv[cluster_indices]
        
        # fig  = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(edge_list_xyz[:,0], edge_list_xyz[:,1], edge_list_xyz[:,2], c='k', marker='o')
        # plt.show()
        
        
        return edge_list_uv, edge_list_xyz
    
    def get_contact_points(self, edge_list_uv, edge_list_xyz):
        '''
        Sample contact points from depth contour
        Args:
            edge_list_xy (N,2): Edge indices list in uv coordinates 
            edge_list_xy (N,2): Edge list in xy coordinates (world frame)
            
        Returns:
            contact_uv_coordinates: (N,2,2): Sampled contact points in depth image pixel format ([(v11,u11),(v12,u12)], [(v21,u21),(v22,u22)] ... [(vN1,uN1),(vN2,uN2)])
            contact_pair_pcd:       (N,2,3): Sampled contact points in 3D coordiantes {world} ([(x11,y11,z11),(x12,y12,z12)], [(x21,y21,z21),(x22,y22,z22)] ... [(xN1,yN1,zN1),(xN2,yN2,zN2)])
        
        '''
        edge_list_xy = edge_list_xyz[:,:2]
        distances = cdist(edge_list_xy, edge_list_xy)
        # Find the point index pairs of cetrain euclidean distance
        contact_pair_idx = np.where(np.abs(distances - self.gripper_width) <= self._width_error_threshold)
        contact_pair_idx = np.vstack((contact_pair_idx[0],contact_pair_idx[1])).T
        contact_pair_idx = np.unique(contact_pair_idx, axis=0) # remove duplicates
        
        contact_pair_xyz = np.hstack((edge_list_xyz[contact_pair_idx[:,0]], edge_list_xyz[contact_pair_idx[:,1]])).reshape(-1,2,3)
        contact_pair_uv = np.hstack((edge_list_uv[contact_pair_idx[:,0]], edge_list_uv[contact_pair_idx[:,1]])).reshape(-1,2,2)
        return contact_pair_uv, contact_pair_xyz
    
    # def get_contact_points(self, edge_list_uv, edge_list_xyz)
    
    @staticmethod
    def remove_outliers(array, threshold=3):
        # Calculate the mean and standard deviation of the array
        mean = np.mean(array, axis=0)
        std = np.std(array, axis=0)

        # Calculate the Z-scores for each data point
        z_scores = np.abs((array - mean) / std)

        # Filter out the outliers based on the threshold
        filtered_array = array[(z_scores < threshold).all(axis=1)]

        return filtered_array

    def edge_list_using_pcd(self, depth_image, segmask, camera_extr, camera_intrinsic):
        '''
        Reproject depth image in vertical view
        '''
        
        # Get point cloud of the object only
        depth_image = depth_image * segmask
        pcd = self.depth_to_pcd(depth_image, camera_intrinsic)
        pcd_object = pcd[np.where(pcd[:,2] > 0.1)[0]]
        
        # Transform point cloud to world frame
        pcd_w = (np.matmul(camera_extr[:3,:3], pcd_object[:,:3].T) + camera_extr[:3,3].reshape(3,1)).T
        
        #########################
        #  Height Thresholding ##
        #########################
        
        threshold_height = 0.01
        # Remove points that are too close to the ground
        pcd_w = pcd_w[np.where(pcd_w[:,2] > threshold_height)[0]]
        
        
        ##########################################
        # Edge Detection - alpha shape algorithm #
        ##########################################
        
        # Calculate the Delaunay triangulation of the point cloud
        pcd_w_2d = pcd_w[:,:2]

        # Define the alpha value (adjust according to your data)
        # alpha_value = 500

        # Calculate the alpha shape of the point cloud
        # alpha_shape = alphashape.alphashape(pcd_w_2d, alpha=alpha_value)
        # precise_contour_points = []
        # for poly in alpha_shape:
        #     point = np.array(poly.exterior.coords.xy)
        #     precise_contour_points.append(point)
        # # Get the points on the precise contour
        # outermost_points = np.array(alpha_shape.exterior.coords)
        
        # Find the convex hull of the point cloud
        hull = ConvexHull(pcd_w_2d)

        # Get the indices of the points on the outermost contour
        outermost_indices = hull.vertices
        num_interpolated_points = 1000
        outermost_indices = np.append(outermost_indices, outermost_indices[0])
        edge_list_xyz = self.interploate_with_even_distance(pcd_w[outermost_indices], num_interpolated_points)
        
        # # Get the points on the outermost contour
        # outermost_points = pcd_w[outermost_indices]
        
        # # Extract x and y coordinates from the contour points
        # x = outermost_points[:, 0]
        # y = outermost_points[:, 1]
        # z = outermost_points[:, 2]
        
        # # Create an interpolation function for x and y coordinates separately
        # interpolation_function_x = interp1d(np.arange(len(x)), x, kind='linear')
        # interpolation_function_y = interp1d(np.arange(len(y)), y, kind='linear')
        # interpolation_function_z = interp1d(np.arange(len(z)), z, kind='linear')

        # # Generate evenly spaced indices for interpolation
        # interpolation_indices = np.linspace(0, len(x)-1, num=num_interpolated_points)

        # # Interpolate x and y coordinates using the interpolation functions
        # x_interpolated = interpolation_function_x(interpolation_indices)
        # y_interpolated = interpolation_function_y(interpolation_indices)
        # z_interpolated = interpolation_function_z(interpolation_indices)

        # # Create the interpolated trajectory with m points (m, 2)
        # edge_list_xyz = np.column_stack((x_interpolated, y_interpolated, z_interpolated))
        # # edge_list_xyz = np.hstack([interpolated_contour_points, 0.005 * np.zeros(len(interpolated_contour_points)).reshape(-1,1)]).reshape(-1,3)
        
        # Get uv coordinates of the edge list
        edge_list_xyz_camera = (np.matmul(np.linalg.inv(camera_extr)[:3,:3], edge_list_xyz[:,:3].T) + np.linalg.inv(camera_extr)[:3,3].reshape(3,1)).T
        edge_list_uvd = edge_list_xyz_camera @ camera_intrinsic.T
        edge_list_uv = edge_list_uvd[:,:2] / edge_list_uvd[:,2].reshape(-1,1)
        edge_list_uv = edge_list_uv.astype(int)
        
        # ####################
        # #  Edge Detection ##
        # ####################
        
        
        # # Find the fartest points of the pcd in the xy plane
        
        # theta = np.linspace(0, np.pi, 100)
        # directions = np.vstack((np.cos(theta), np.sin(theta))).T
        # projections = directions @ pcd_w[:,:2].T
        # indices_sorted_by_distance = np.argsort(projections, axis=1)
        # max_indices, min_indices = indices_sorted_by_distance[:,-10:], indices_sorted_by_distance[:,:10]
        # max_points, min_points = pcd_w[max_indices], pcd_w[min_indices]
        # farthest_points = np.vstack((max_points, min_points))
        # farthest_points = np.unique(farthest_points.reshape(-1,3), axis=0)
        # edge_list_xyz = self.remove_outliers(farthest_points)
        
        
        # edge_list_xyz_camera = (np.matmul(np.linalg.inv(camera_extr)[:3,:3], edge_list_xyz[:,:3].T) + np.linalg.inv(camera_extr)[:3,3].reshape(3,1)).T
        # edge_list_uvd = edge_list_xyz_camera @ camera_intrinsic.T
        # edge_list_uv = edge_list_uvd[:,:2] / edge_list_uvd[:,2].reshape(-1,1)
        # edge_list_uv = edge_list_uv.astype(int)
        
        return edge_list_uv, edge_list_xyz
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # # Find the naive center of the object
        # object_center = np.mean(pcd_w, axis=0)
        
        # # Define the vertical view - camera pose
        # R_v = R.from_euler('xyz', [0, 180, 0], degrees=True).as_matrix()
        # t_v = np.array([object_center[0], object_center[1],  camera_extr[2,3]]) # Assume the camear is right above the object
        # # t_v = np.array([0, 0, camera_extr[2,3]]) # Assume the camear is right above the object
        
        # camera_pose_v = np.eye(4)
        # camera_pose_v[:3,:3] = R_v
        # camera_pose_v[:3,3] = t_v
        
        # # Compute camera projection matrix
        # K = camera_intrinsic
        # T = np.linalg.inv(camera_pose_v)
        # pcd_v = (np.matmul(T[:3,:3], pcd_w[:,:3].T) + T[:3,3].reshape(3,1)).T
        
        # point_uvd = pcd_v @ K.T
        # point_uvd = point_uvd / point_uvd[:,2].reshape(-1,1)
        # point_uv = point_uvd[:,:2]
        # point_uv = point_uv.astype(int)
        
        # # Get the depth image in vertical view
        # depth_image_v = np.zeros(depth_image.shape)
        # depth_image_v[point_uv[:,1], point_uv[:,0]] = pcd_v[:,2]
        
        # segmask = np.ones(depth_image.shape)
        
        # self.pcd_w = pcd_w
        
        # return depth_image_v, segmask, camera_pose_v
        
    @staticmethod
    def interploate_with_even_distance(trajectory, num_sample):
        '''
        From a trajectory, interpolate the points with even Euclidean distances (xy-plane).
        
        Args:
            trajectory (N,3): Trajectory points
            num_sample (int): Number of points to be sampled
        Returns:
            interpolated_trajectory (num_sample,3): Interpolated trajectory points
        '''
        # Extract the x and y coordinates from the trajectory
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        z = trajectory[:, 2]

        # Compute the cumulative distance along the trajectory
        distances = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        distances = np.insert(distances, 0, 0)  # Prepend a zero for the initial position

        # Create an interpolation function for x and y coordinates
        interp_func_x = interp1d(distances, x, kind='linear')
        interp_func_y = interp1d(distances, y, kind='linear')
        interp_func_z = interp1d(distances, z, kind='linear')

        # Generate evenly spaced distances for the interpolated points
        target_distances = np.linspace(0, distances[-1], num_sample)

        # Interpolate the x and y coordinates at the target distances
        interpolated_x = interp_func_x(target_distances)
        interpolated_y = interp_func_y(target_distances)
        interpolated_z = interp_func_z(target_distances)

        # Return the interpolated x and y coordinates as a (m, 2) trajectory
        interpolated_trajectory = np.column_stack((interpolated_x, interpolated_y, interpolated_z))
        return interpolated_trajectory
        
    @staticmethod
    def pcd_idx_to_uv_coordinates(contact_pair_idx, depth_img_shape):
        """Get xy pixel coordinates from point cloud indices
        Args:
            contact_pair_idx (N,2): PCD indices of contact pairs ([idx_11,idx_12], [idx_21,idx_22] ... [idx_N1,idx_N2])
        Returns:
            (N,(2,2)): Contact points in depth image pixel format ([(v_11,u_11),(v_12,u_12)], [(v_21,u_21),(v_22,u_22)] ... [(v_N1,u_N1),(v_N2,u_N2)])
            
            Note: 
            1. the order of the contact points is not guaranteed to be the same as the input
            2. Returned coordinates start with v, not u. (v,u) = (x,y)
            
        """
        first_indices, second_indices = contact_pair_idx[:,0], contact_pair_idx[:,1]
        H, W = depth_img_shape[0], depth_img_shape[1]
        y1, x1= np.unravel_index(first_indices,  (H,W))
        y2, x2= np.unravel_index(second_indices, (H,W))
        contact_uv_coordinates = np.hstack((y1.reshape(-1,1), x1.reshape(-1,1), y2.reshape(-1,1), x2.reshape(-1,1))).reshape(-1,2,2)
        
        return contact_uv_coordinates
    
    @staticmethod
    def depth_to_pcd(depth_image, camera_intr):
        height, width = depth_image.shape
        row_indices = np.arange(height)
        col_indices = np.arange(width)
        pixel_grid = np.meshgrid(col_indices, row_indices)
        pixels = np.c_[pixel_grid[0].flatten(), pixel_grid[1].flatten()].T
        pixels_homog = np.r_[pixels, np.ones([1, pixels.shape[1]])]
        depth_arr = np.tile(depth_image.flatten(), [3, 1])
        point_cloud = depth_arr * np.linalg.inv(camera_intr).dot(pixels_homog)
        return point_cloud.transpose()
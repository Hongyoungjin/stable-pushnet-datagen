import numpy as np
import open3d as o3d
import copy
import os
import warnings


# 현재 스크립트 파일의 절대 경로를 얻습니다.
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
assets_dir = os.path.join(current_directory, "assets/dish_urdf/")

# Define arguments to pass to the Python script
objects = os.listdir(assets_dir)
objects.sort()

class SelectStablePose(object):
    
    def __init__(self, asset_dir):
        
        self.is_modified = None
        self.current_idx = 0
        self.current_pose = None
        self.asset_dir = asset_dir
        self.objects = os.listdir(asset_dir)
        self.objects.sort()
        
        self.check_for_all_objects()
        
    def check_for_all_objects(self):
        
        for object in self.objects:
            
            self.current_object = object
            self.check_for_object()

    def check_for_object(self):
        self.is_modified = False
        self.stable_poses = np.load(self.asset_dir + self.current_object + '/stable_poses.npy', allow_pickle=True)
        # Normal case where there are multiple stable poses
        if len(self.stable_poses.shape) == 3:
            
            print("Current object: ", self.current_object)
            
            for idx in range(self.stable_poses.shape[0]):
                
                if self.is_modified:
                    break
                
                print("Idx: ", idx)
                self.current_idx = idx
                self.current_pose = self.stable_poses[idx]
                self.visualize_in_given_pose()
            
        # Case where there is only one stable pose        
        if len(self.stable_poses.shape) == 2:
            
            print("Current object: ", self.current_object)
            self.current_pose = self.stable_poses
            self.visualize_in_given_pose()
                
    def visualize_in_given_pose(self) :
        
        # Load the mesh
        
        # # try:
        # #     mesh = o3d.io.read_triangle_mesh(self.asset_dir + self.current_object + '/' + self.current_object + '.stl')
        # # except:
        mesh = o3d.io.read_triangle_mesh(self.asset_dir + self.current_object + '/' + self.current_object + '.obj')
            
        mesh1 = copy.deepcopy(mesh).transform(self.current_pose)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06, origin=[0, 0, 0])
        
        # Load the text to visualize
        text_object = 'Object: ' + self.current_object
        text_index = 'Pose index: ' + str(self.current_idx)
        
        text_object_pcd = self.text_3d(text_object, pos=[-0.05, 0.07, 0], direction=[0, 0, 1], degree=-90.0, density = 20, font='DejaVu Sans Mono for Powerline.ttf', font_size=1)
        text_index_pcd = self.text_3d(text_index, pos=[-0.05, 0.06, 0], direction=[0, 0, 1], degree=-90.0, density = 20, font='DejaVu Sans Mono for Powerline.ttf', font_size=1)
        
        
        text_notice = 'Press S to save the pose, C to close the window, E to exit the program'
        text_notice_pcd = self.text_3d(text_notice, pos=[-0.1, -0.07, 0], direction=[0, 0, 1], degree=-90.0, density = 20, font='DejaVu Sans Mono for Powerline.ttf', font_size=1)
        
        
        # Create the visualizer
        vis = o3d.visualization.VisualizerWithKeyCallback()
        
        # Register the callback functions
        vis.register_key_callback(ord('S'), self.modify_stable_pose)
        vis.register_key_callback(ord('C'), self.close_window)
        vis.register_key_callback(ord('E'), self.__del__)
        vis.create_window()
        
        # Add the geometry to the visualizer
        vis.add_geometry(mesh1)
        vis.add_geometry(mesh_frame)
        
        # Add the text to the visualizer
        vis.add_geometry(text_object_pcd)
        vis.add_geometry(text_index_pcd)
        vis.add_geometry(text_notice_pcd)
        
        # Run the visualizer
        vis.run()
    
    @staticmethod
    def text_3d(text, pos, direction=None, degree=0.0, density = 10, font='/usr/share/fonts/truetype/ttf-bitstream-vera/VeraMoBd.ttf', font_size=16):
        # https://github.com/isl-org/Open3D/issues/2#issuecomment-610683341
        """
        Generate a 3D text point cloud used for visualization.
        :param text: content of the text
        :param pos: 3D xyz position of the text upper left corner
        :param direction: 3D normalized direction of where the text faces
        :param degree: in plane rotation of text
        :param font: Name of the font - change it according to your system
        :param font_size: size of the font
        :return: o3d.geoemtry.PointCloud object
        """
        if direction is None:
            direction = (0., 0., 1.)

        from PIL import Image, ImageFont, ImageDraw
        from pyquaternion import Quaternion

        try:
            font_obj = ImageFont.truetype(font, font_size*density)
        except IOError:
            font_obj = ImageFont.load_default()
            
        font_dim = font_obj.getsize(text)

        img = Image.new('RGB', font_dim, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
        img = np.asarray(img)
        img_mask = img[:, :, 0] < 128
        indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

        pcd = o3d.geometry.PointCloud()
        pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
        pcd.points = o3d.utility.Vector3dVector(indices / 100.0 / density)

        raxis = np.cross([0.0, 0.0, 1.0], direction)
        
        if np.linalg.norm(raxis) < 1e-6:
            raxis = (0.0, 0.0, 1.0)
            
        trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
                Quaternion(axis=direction, degrees=degree)).transformation_matrix
        
        trans[0:3, 3] = np.asarray(pos)
        
        pcd.transform(trans)
        
        return pcd
    
    ## Callback functions
    def __del__(self, vis):
        vis.destroy_window()
        exit()
        
    def close_window(self, vis):
        vis.destroy_window()
        
    def modify_stable_pose(self, vis):
        with open(self.asset_dir + self.current_object + '/stable_poses.npy', 'wb') as f:
            np.save(f, self.current_pose)
        vis.destroy_window()
        print("Modified stable pose for object: ", self.current_object)
        print("Idx: ", self.current_idx)
        self.is_modified = True


if __name__ == '__main__':
    select = SelectStablePose(assets_dir)
    
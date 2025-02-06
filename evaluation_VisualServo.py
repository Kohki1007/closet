import numpy as np
import os
from typing import List, Optional

from PIL import Image
import cv2
import numpy as np
import math
import random
import os
import time
from os.path import join

from isaacgym import gymutil, gymtorch, gymapi
import isaacgym.torch_utils as torch_utils
from predict_opening_door.dataset.predict_opening_wall.rgbd_attention_variouscolor.utils_another.build_yolo import YOLO
# from plane_estimation import Depth
# from make_urdf import edit_urdf_solo, edit_urdf

from predict_targetimage_action import Autoencoder

from pix2pix_network import define_G

from torchvision import transforms

import torch
import time


class rendar():
    def __init__(self, num_envs = 2):
        # self.closet_id = 2
        self.num_envs = num_envs
        self.test_num_envs = self.num_envs//2
        self.device = 'cuda'
        self.num_dofs = 8
        self.physics_engine = gymapi.SIM_PHYSX

        self.yolo = YOLO()


        self.closet_position = torch.load("/home/engawa/py_ws/visual_servo/src/network/predict_action/dataset/predict_action/closet_props/closet_position.pth")
        self.closet_rot = torch.load("/home/engawa/py_ws/visual_servo/src/network/predict_action/dataset/predict_action/closet_props/closet_rot.pth")

        # self.input_base = torch.load("/home/engawa/py_ws/visual_servo/src/network/predict_action/dataset/predict_action/dataset_base/input_with_attention.pth")
        # self.output_base = torch.load("/home/engawa/py_ws/visual_servo/src/network/predict_action/dataset/predict_action/dataset_base/output_with_attention.pth")

        self.network = Autoencoder(train = True)
        targetG_weight = torch.load("/home/engawa/py_ws/visual_servo/src/network/pytorch-CycleGAN-and-pix2pix/checkpoints/rgbd_d_0807/latest_net_G.pth", map_location='cuda')
        deleteG_weight = torch.load("/home/engawa/py_ws/visual_servo/src/network/pytorch-CycleGAN-and-pix2pix/checkpoints/delete_ur_many/latest_net_G.pth", map_location='cuda')
        # deleteG_weight = torch.load("src/network/pytorch-CycleGAN-and-pix2pix/checkpoints/delete_ur_many/latest_net_G.pth", map_location='cuda')
        action_weight = torch.load("src/network/log/0825/weight/weights_last.pth", map_location='cuda')

        self.generate_target = self.network.generate_target
        # self.delete_ur = self.network.delete_ur
        self.delete_ur = define_G(4, 1, 64, 'unet_256')
        self.predict_action = self.network.predict_action

        self.generate_target.module.load_state_dict(targetG_weight)
        self.delete_ur.module.load_state_dict(deleteG_weight)
        self.predict_action.load_state_dict(action_weight)

        self.generate_target.to('cuda')
        self.delete_ur.to('cuda')
        self.predict_action.to('cuda')

        self.asset_path_list = []

        self.transform = transforms.Resize((256, 256))
        self.transform_reducation = transforms.Resize((48, 64))
        self.transform_attention = transforms.Resize((90, 160))

        self.handle_trajectory = torch.load("/home/engawa/py_ws/visual_servo/src/network/predict_action/dataset/predict_action/dataset_base/handle_trajectory.pth")
        self.ur_default_dof_pos = torch.tensor([0.5, 0, 1.0, 0, 0, 0])
        self.handle_trajectory *= -1
        # self.handle_trajectory[:, :, 0] += 0.3
        # self.handle_trajectory[:, :, 1] -= 0.1
        # self.handle_trajectory[:, 1

        print("good")

        # self.closet_position = torch.load("/home/engawa/py_ws/visual_servo/src/network/predict_opening_door/dataset/delete_UR/rgbd_rgbd/base/closet_position.pth")

        self.trajectory_list = []



    def create_sim(self, compute_device_id = 0, graphics_device_id = 0):
        self.gym = gymapi.acquire_gym()
        # self.num_envs= 1
        

        self.args = gymutil.parse_arguments(description="Joint control Methods Example")
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity.x = 0
        sim_params.gravity.y = 0
        sim_params.gravity.z = -9.81
        sim_params.gravity.z = 0
        sim_params.dt = 1.0 / 3
        sim_params.use_gpu_pipeline = True
        # create a simulator
        # sim_params = gymapi.SimParams()
        sim_params.substeps = 2
        # sim_params.dt = 1.0 / 60.0

        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1

        sim_params.physx.num_threads = self.args.num_threads
        sim_params.physx.use_gpu = self.args.use_gpu

        sim_params.use_gpu_pipeline = False
        self.sim = self.gym.create_sim(
        compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)

        

    def create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        
    

    ##closet list : include closet information
    ###[handle width, handle height, handle, closet_x, closet_y]
    def create_envs(self, i = 1, spacing = 10, num_per_row = 10):
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        self.cam_pos = gymapi.Vec3(0.9, 2.0, 1.0)
        self.cam_target = gymapi.Vec3(-0.5, 0, 0.5)
        # self.cam_pos = gymapi.Vec3(0.5, 0, 1.5)
        # self.cam_target = gymapi.Vec3(-0.8, 0, 1.0)
        # self.cam_pos = gymapi.Vec3(0.5, 0.6, 1.0)
        # self.cam_target = gymapi.Vec3(-0.2, 0, 1.0)
        self.cam_pos = gymapi.Vec3(0.5, -0.6, 1.8)
        self.cam_target = gymapi.Vec3(-0.8, 0, 1.0)

        self.cam_pos = gymapi.Vec3(2., -1.7, 2.)
        self.cam_target = gymapi.Vec3(-0.8, -0.3, 1.0)

        self.gym.viewer_camera_look_at(self.viewer, None, self.cam_pos, self.cam_target)
        # # if i == 1:
        #     self.gym.viewer_camera_look_at(self.viewer, None, self.cam_pos, self.cam_target)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets")
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "/home/engawa/py_ws/visual_servo/src/network/predict_opening_door/dataset/predict_opening_wall/assets"
        # asset_root = "../../assets"
        ur_asset_file = "urdf/ur3/robots/ur3.urdf"
        # ur_asset_file = "urdf/ur3/robots/ur3_agv.urdf"
        ur_asset_file = "urdf/ur3/robots/hook_test.urdf"
        # ur_asset_file = "urdf/ur3/robots/.urdf"
        # camera_asset_file = "urdf/camera/urdf/camera.urdf"
        handle_prop_list = torch.load("/home/engawa/py_ws/visual_servo/src/network/predict_action/dataset/predict_action/closet_props/handle_position.pth")
        for i in range(handle_prop_list.shape[0]):
            file_path = f"urdf/closet_for_sdf/urdf_varius_size/{handle_prop_list[i, 0].to(torch.int64)}_{handle_prop_list[i, 1].to(torch.int64)}_{handle_prop_list[i, 2].to(torch.int64)}.urdf"
            self.asset_path_list.append(file_path)


        # for a in range(5):
        #     scale = scale_list[a]*10
        #     scale_number = math.trunc(scale)
        #     half = 0.290 * scale_list[a] / 2
        #     for b in range(12):
        #         wide = random.uniform(0.05, half)
        #         round_wide = wide*10000
        #         wide_number = math.trunc(round_wide)

        #         height = random.uniform(0.8, 1.2)
        #         round_height = height*1000
        #         height_number = math.trunc(round_height)
                
        #         closet_id = a * 12 + b

        #         handle_prop_list[closet_id] = torch.tensor([scale_number, wide_number, height_number])
                

        #         file_path = f"/home/engawa/py_ws/IsaacGymEnvs_yolo/assets/urdf/closet_for_sdf/urdf_varius_size/{scale_number}_{wide_number}_{height_number}.urdf"

        #         if not self.check_file_exists(file_path):
        #             edit_urdf(scale_number, wide_number, height_number)

        #         file_path = f"urdf/closet_for_sdf/urdf_varius_size/{scale_number}_{wide_number}_{height_number}.urdf"
        #         self.asset_path_list.append(file_path)

        # load ur asset
        asset_options = gymapi.AssetOptions()

        asset_box = self.gym.create_box(self.sim, 0.8, 4.0, 2.5, asset_options)
        # asset_camera = self.gym.create_box(self.sim, 0.05, 0.03, 0.03, asset_options)
        # target_box_color = gymapi.vec3(1.0, 1.0, 1.0)

        vh_options = gymapi.VhacdParams()
   
        vh_options.max_convex_hulls = 10000
        asset_options.convex_decomposition_from_submeshes = True
        asset_options.vhacd_enabled = True
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        # asset_options.max_angular_velocity = 0.001
        ur_asset = self.gym.load_asset(self.sim, asset_root, ur_asset_file, asset_options)

        # load closet asset
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.use_mesh_materials = True

        closet_asset = []
        for i in range(len(self.asset_path_list)):
            closet_asset.append(self.gym.load_asset(self.sim, asset_root, self.asset_path_list[i], asset_options))

        # # camera_asset = self.gym.load_asset(self.sim, asset_root, camera_asset_file, asset_options)
        # closet_asset = self.gym.load_asset(self.sim, asset_root, closet_asset_file, asset_options)

        # wall_asset = self.gym.load_asset(self.sim, asset_root, wall_asset_file, asset_options)

        pi = 3.14 * 30 /180
        camera_start_pose_right= gymapi.Transform()
        camera_start_pose_right.p = gymapi.Vec3(0,0.15,1.5)
        camera_start_pose_right.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1, 0), np.radians(-pi))

        camera_start_pose_left = gymapi.Transform()
        camera_start_pose_left.p = gymapi.Vec3(0,-0.15,1.5)
        camera_start_pose_left.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(-pi))

        self.num_ur_bodies = self.gym.get_asset_rigid_body_count(ur_asset)
        self.num_ur_dofs = self.gym.get_asset_dof_count(ur_asset)
        self.num_closet_bodies = self.gym.get_asset_rigid_body_count(closet_asset[0])
        self.num_closet_dofs = self.gym.get_asset_dof_count(closet_asset[0])
        # self.num_camera_bodies = self.gym.get_asset_rigid_body_count(camera_asset)
        # self.num_camera_dofs = self.gym.get_asset_dof_count(camera_asset)

        print("num ur bodies: ", self.num_ur_bodies)
        print("num ur dofs: ", self.num_ur_dofs)
        print("num closet bodies: ", self.num_closet_bodies)
        print("num closet dofs: ", self.num_closet_dofs)

        self.num_dofs = self.num_ur_dofs + self.num_closet_dofs
        self.ur_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device="cpu")
        # self.ur_dof_targets = torch.zeros((self.num_envs, 38), dtype=torch.float, device="cpu")

        # set ur dof properties
        ur_dof_props = self.gym.get_asset_dof_properties(ur_asset)
        self.ur_dof_lower_limits = []
        self.ur_dof_upper_limits = []
        for i in range(self.num_ur_dofs):
            ur_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                ur_dof_props['stiffness'][i] = 10000000
                ur_dof_props['damping'][i] = 50
                ur_dof_props['effort'][i] = 0.0000000001
            else:
                ur_dof_props['stiffness'][i] = 7000.0
                ur_dof_props['damping'][i] = 50.0

            self.ur_dof_lower_limits.append(ur_dof_props['lower'][i])
            self.ur_dof_upper_limits.append(ur_dof_props['upper'][i])

        self.ur_dof_lower_limits = self.to_torch(self.ur_dof_lower_limits, device=self.device)
        self.ur_dof_upper_limits = self.to_torch(self.ur_dof_upper_limits, device=self.device)
        self.ur_dof_speed_scales = torch.ones_like(self.ur_dof_lower_limits)/2

        closet_dof_props = self.gym.get_asset_dof_properties(closet_asset[0])

        for j in range(self.num_closet_dofs):
            closet_dof_props['driveMode'][j] = gymapi.DOF_MODE_POS
            closet_dof_props['stiffness'][j] = 7000
            closet_dof_props['damping'][j] = 0
            closet_dof_props['effort'][j] = 0



        # wall_dof_props = self.gym.get_asset_dof_properties(wall_asset)

        # for j in range(self.num_closet_dofs):
        #     wall_dof_props['driveMode'][j] = gymapi.DOF_MODE_NONE
        #     wall_dof_props['stiffness'][j] = 7000
        #     wall_dof_props['damping'][j] = 0
        #     wall_dof_props['effort'][j] = 0


        ur_start_pose = gymapi.Transform()
        ur_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        ur_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0, 1), np.radians(0))
        ur_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1,0, 0), np.radians(0))

        closet_start_pose = gymapi.Transform()
        closet_start_pose.p = gymapi.Vec3(-0.85, 0, 0)
        closet_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0, 1), np.radians(0))

        # num_ur_bodies = self.gym.get_asset_rigid_body_count(ur_asset)
        # num_ur_shapes = self.gym.get_asset_rigid_shape_count(ur_asset)
        # num_closet_bodies = self.gym.get_asset_rigid_body_count(closet_asset[0])
        # num_closet_shapes = self.gym.get_asset_rigid_shape_count(closet_asset[0])
        # # num_prop_bodies = self.gym.get_asset_rigid_body_count(prop_asset)
        # # num_prop_shapes = self.gym.get_asset_rigid_shape_count(prop_asset)

        self.camera_props1 = gymapi.CameraProperties()
        # self.camera_props.horizontal_fov = 75.0
        self.camera_props1.width = 640
        self.camera_props1.height = 480 
        self.camera_props1.horizontal_fov = 100
        self.camera_props1.enable_tensors = True
        # self.camera_props1.near_plane = 0.1
        # self.camera_props1.far_plane = 0.50

        self.camera_props2 = gymapi.CameraProperties()
        # self.camera_props.horizontal_fov = 75.0
        self.camera_props2.width = 640
        self.camera_props2.height = 480
        self.camera_props2.horizontal_fov = 100
        self.camera_props2.enable_tensors = True
        # self.camera_props1.near_plane = 0.1
        # self.camera_props1.far_plane = 0.50

        self.camera_props3 = gymapi.CameraProperties()
        self.camera_props3.width = 160
        self.camera_props3.height = 90 
        self.camera_props3.horizontal_fov = 120
        self.camera_props3.enable_tensors = True

        self.camera_props4 = gymapi.CameraProperties()
        # self.camera_props.horizontal_fov = 75.0
        self.camera_props4.width = 160
        self.camera_props4.height = 90
        self.camera_props4.horizontal_fov = 120
        self.camera_props4.enable_tensors = True

        self.camera_pos_right2 = gymapi.Vec3(0.,0.2,1.5)
        self.camera_pos_left2  = gymapi.Vec3(0.,-0.2,1.5)

        self.camera_rot_right2 = gymapi.Vec3(-0.4,0.2, 1.)
        self.camera_rot_left2 = gymapi.Vec3(-0.4,-0.2, 1.)


        self.camera_pos_right = gymapi.Vec3(0.3,0.1,1)
        self.camera_pos_left  = gymapi.Vec3(0.3,-0.1,1)

        self.camera_rot_right = gymapi.Vec3(-0.4,0.1, 1.)
        self.camera_rot_left = gymapi.Vec3(-0.4, -0.1, 1)
        # self.camera_rot_left = gymapi.Vec3(-0.4, 0, 1)

        self.camera_pos_fix = gymapi.Vec3(-0.1,0,0.6)

        self.camera_rot_fix = gymapi.Vec3(-0.6,0,0.491)
   
        self.urs = []
        self.closets = []
        self.default_prop_states = []
        self.prop_start = []
        self.envs = []
        self.hand_handle = []
        self.drawer_handle = []
        self.lfinger_handle = []
        self.rfinger_handle = []
        self.camera_handle = []
        self.closet_list = []
        self.asset_list = []

        asset_id = random.sample(range(1, 40), self.test_num_envs)
        asset_id = [15]
        position_id = random.sample(range(1, 50), self.test_num_envs)

        self.asset_list.append(asset_id)

        self.closet_position = self.closet_position[asset_id[0]]

        for i in range(self.test_num_envs):
            self.env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            
            # quotient = i //5
            # remainder = i % 5
            # asset_number = quotient * 12 + remainder
            # asset_number = remainder * 12 + quotient

            ur_pose = ur_start_pose
            ur_pose.p.x = 0
            ur_pose.p.y = 0
            # ur_pose.p.z = 0.4
            ur_pose.p.z = 0
            self.ur_actor = self.gym.create_actor(self.env_ptr, ur_asset, ur_start_pose, "ur", i, 0, 0)
            self.gym.set_actor_dof_properties(self.env_ptr, self.ur_actor, ur_dof_props)

            closet_pose = closet_start_pose
            closet_position = torch.tensor([random.uniform(-0.7, -0.5), random.uniform(-0.15, 0.15)]).to("cuda")
            closet_pose.p.x = self.closet_position[0]
            closet_pose.p.y = self.closet_position[1]
            closet_pose.p.z = 0

            self.closet_list.append(self.closet_position)
            self.closet_actor = self.gym.create_actor(self.env_ptr, closet_asset[self.asset_list[i][0]], closet_pose, "closet", i, 0, 0)
            self.gym.set_actor_dof_properties(self.env_ptr, self.closet_actor, closet_dof_props)

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(-1.4, 0, 1.25)
            # pose.p = gymapi.Vec3(-1.4, 0, 1.25)
            pose.r = gymapi.Quat(0, 0, 0, 1)
            box_handle = self.gym.create_actor(self.env_ptr, asset_box, pose, "actor1", i, 0)
            shape_props = self.gym.get_actor_rigid_shape_properties(self.env_ptr, box_handle)
            shape_props[0].restitution = 1
            shape_props[0].compliance = 0.5

            self.gym.set_actor_rigid_shape_properties(self.env_ptr, box_handle, shape_props)
            self.gym.set_rigid_body_color(self.env_ptr, box_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1., 1., 1.))

            # ############camera
            # pose_camera = gymapi.Transform()
            # pose_camera.p = gymapi.Vec3(0.375, -0.1, 5.)
            # # pose.p = gymapi.Vec3(-1.4, 0, 1.25)
            # pose_camera.r = gymapi.Quat(0, 0, 0, 1)
            # camera_handle = self.gym.create_actor(self.env_ptr, asset_camera, pose_camera, "camera", i, 0)
            # shape_props_camera = self.gym.get_actor_rigid_shape_properties(self.env_ptr, camera_handle)
            # shape_props_camera[0].restitution = 1
            # shape_props_camera[0].compliance = 0.5

            # self.gym.set_actor_rigid_shape_properties(self.env_ptr, camera_handle, shape_props_camera)
            # self.gym.set_rigid_body_color(self.env_ptr, camera_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1., 1., 1.))
            # ###################
            
            self.envs.append(self.env_ptr)
            # self.urs.append(self.ur_actor)
            # self.closets.append(self.closet_actor)
            self.camera_handle.append(self.gym.create_camera_sensor(self.envs[i], self.camera_props1))
            self.camera_handle.append(self.gym.create_camera_sensor(self.envs[i], self.camera_props2))
            
            self.gym.set_camera_location(self.camera_handle[0], self.env_ptr, self.camera_pos_right, self.camera_rot_right)
            self.gym.set_camera_location(self.camera_handle[1], self.env_ptr, self.camera_pos_left, self.camera_rot_left)

        for i in range(self.test_num_envs):
            self.env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            ur_pose = ur_start_pose
            ur_pose.p.x = -5
            ur_pose.p.y = 0
            # ur_pose.p.z = 0.4
            ur_pose.p.z = 0
            self.ur_actor = self.gym.create_actor(self.env_ptr, ur_asset, ur_start_pose, "ur", i, 0, 0)
            self.gym.set_actor_dof_properties(self.env_ptr, self.ur_actor, ur_dof_props)

            closet_pose = closet_start_pose
            closet_position = self.closet_list[i]
            closet_pose.p.x = self.closet_position[0]
            closet_pose.p.y = self.closet_position[1]
            closet_pose.p.z = 0

            # self.closet_list.append(closet_position)
            self.closet_actor = self.gym.create_actor(self.env_ptr, closet_asset[self.asset_list[i][0]], closet_pose, "closet", i, 0, 0)
            self.gym.set_actor_dof_properties(self.env_ptr, self.closet_actor, closet_dof_props)

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(-1.4, 0, 1.25)
            # pose.p = gymapi.Vec3(-1.4, 0, 1.25)
            pose.r = gymapi.Quat(0, 0, 0, 1)
            box_handle = self.gym.create_actor(self.env_ptr, asset_box, pose, "actor1", i, 0)
            shape_props = self.gym.get_actor_rigid_shape_properties(self.env_ptr, box_handle)
            shape_props[0].restitution = 1
            shape_props[0].compliance = 0.5

            # ############camera
            # pose_camera = gymapi.Transform()
            # pose_camera.p = gymapi.Vec3(0.375, -0.1, 1.0)
            # # pose.p = gymapi.Vec3(-1.4, 0, 1.25)
            # pose_camera.r = gymapi.Quat(0, 0, 0, 1)
            # camera_handle = self.gym.create_actor(self.env_ptr, asset_camera, pose_camera, "camera", i, 0)
            # shape_props_camera = self.gym.get_actor_rigid_shape_properties(self.env_ptr, camera_handle)
            # shape_props_camera[0].restitution = 1
            # shape_props_camera[0].compliance = 0.5

            # self.gym.set_actor_rigid_shape_properties(self.env_ptr, camera_handle, shape_props_camera)
            # self.gym.set_rigid_body_color(self.env_ptr, camera_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1., 1., 1.))
            # ###################

            self.gym.set_actor_rigid_shape_properties(self.env_ptr, box_handle, shape_props)
            self.gym.set_rigid_body_color(self.env_ptr, box_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0., 0., 0.))
            self.envs.append(self.env_ptr)

            self.camera_handle.append(self.gym.create_camera_sensor(self.envs[i + self.test_num_envs], self.camera_props1))
            self.camera_handle.append(self.gym.create_camera_sensor(self.envs[i + self.test_num_envs], self.camera_props2))

            self.camera_handle.append(self.gym.create_camera_sensor(self.envs[i + self.test_num_envs], self.camera_props3))
            self.camera_handle.append(self.gym.create_camera_sensor(self.envs[i + self.test_num_envs], self.camera_props4))

            self.gym.set_camera_location(self.camera_handle[0], self.env_ptr, self.camera_pos_right, self.camera_rot_right)
            self.gym.set_camera_location(self.camera_handle[1], self.env_ptr, self.camera_pos_left, self.camera_rot_left)

            self.gym.set_camera_location(self.camera_handle[4], self.env_ptr, self.camera_pos_right2, self.camera_rot_right2)
            self.gym.set_camera_location(self.camera_handle[5], self.env_ptr, self.camera_pos_left2, self.camera_rot_left2)
            
            

        

        self.hand_handle = self.gym.find_actor_rigid_body_handle(self.env_ptr, self.ur_actor, "hook")
        self.closet_door = self.gym.find_actor_rigid_body_handle(self.env_ptr, self.closet_actor, "closet_door")
        self.closet_handle = self.gym.find_actor_rigid_body_handle(self.env_ptr, self.closet_actor, "handle")

        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)

        
        self.ini()

    def ini(self):
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # self.gym.refresh_dof_state_tensor(sim)
        # self.gym.refresh_dof_state_tensor(sim)
        self.dof_state = gymtorch.wrap_tensor(self.dof_state_tensor)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.ur_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_ur_dofs]
        self.ur_dof_pos = self.ur_dof_state[..., 0]
        self.ur_dof_vel = self.ur_dof_state[..., 1]
        self.closet_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_ur_dofs:]
        self.closet_dof_pos = self.closet_dof_state[..., 0]
        self.closet_dof_vel = self.closet_dof_state[..., 1]

        self.global_indices = torch.arange(self.num_envs * (2 + 0), dtype=torch.int32, device="cuda").view(self.num_envs, -1)

        self.closet_dof_pole0 = self.gym.find_actor_dof_handle(self.envs[0], self.closet_actor, 'door_right_joint')
        self.closet_dof_pole1 = self.gym.find_actor_dof_handle(self.envs[0], self.closet_actor, 'door_left_joint')
        self.closet_dof_handle = self.gym.find_actor_dof_handle(self.envs[0], self.closet_actor, 'handle')

        self.ur_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_ur_dofs]
        self.ur_dof_pos = self.ur_dof_state[..., 0]
        self.ur_dof_vel = self.ur_dof_state[..., 1]
        self.closet_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_ur_dofs:self.num_ur_dofs+2]
        self.closet_dof_pos = self.closet_dof_state[..., 0]
        self.closet_dof_vel = self.closet_dof_state[..., 1]


        _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_tensor = gymtorch.wrap_tensor(_root_tensor)
        self.root_tensor = self.root_tensor.view(self.num_envs, -1, 13)
        # self.box_root_state = self.root_tensor.view(self.num_envs, -1, 13)[:, 2]


        self.ur_dof_state[:self.test_num_envs, :self.num_ur_dofs, 0] = self.ur_default_dof_pos.to("cpu")

        # self.ur_dof_state[:self.test_num_envs, 0, 0] = 0.5


        multi_env_ids_int32 = self.global_indices[:, :2].flatten().to('cpu')

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                            gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        # self.render()

        self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        self._jacobian = self.gym.acquire_jacobian_tensor(self.sim, "ur")
        self.jacobian = gymtorch.wrap_tensor(self._jacobian)
        self.gym.refresh_dof_state_tensor(self.sim)

        now = time.time() 
        self.gym.write_viewer_image_to_file(self.viewer, f"/home/engawa/py_ws/visual_servo/src/network/images_sice/result/{now}.png")
    
    def pre_physics_step(self, k):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        # degree = self.closet_angle_list[:, i]

        for i in range(self.num_envs):

            id = i * 3 + 2

            # self.root_tensor[id, 0] = -1.25
            # self.root_tensor[id, 1] = 0
            self.root_tensor[id, 3:7] = torch.tensor([0, 0, 0, 1])
            self.root_tensor[id, 7:] = 0

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_tensor))

        target_angle = self.closet_rot[:, k]

        right_rad = target_angle * 1.57/180
        left_rad = right_rad * -2
        angle = torch.stack([right_rad, left_rad], dim = 1)


        # self.ur_dof_targets[:self.test_num_envs, :2] = self.actions.to("cpu")

        self.closet_dof_state[:, :2, 0] = angle
        multi_env_ids_int32 = self.global_indices[:, :2].flatten().to('cpu')

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                            gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
        
        self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.hand_pos = self.rigid_body_states[:self.test_num_envs, self.closet_handle][:, 0:3].to("cuda")
        self.hand_rot = self.rigid_body_states[:self.test_num_envs, self.closet_handle][:, 3:7]

        self.get_image(k)
    
    def pre_physics_step_brank(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        self.root_tensor[:, 2, 3:7] = torch.tensor([0, 0, 0, 1])
        self.root_tensor[:, 2, 7:] = 0

        # self.root_tensor[:, 3, :3] = torch.tensor([0.375, -0.1, 5.])
        # self.root_tensor[:, 3, 3:7] = torch.tensor([0, 0, 0, 1])
        # self.root_tensor[:, 3, 7:] = 0

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_tensor))


    def check_file_exists(self, file_path):
        if os.path.exists(file_path):
            # print(f"ファイルが存在します: {file_path}")
            return True
        else:
            # print(f"ファイルが存在しません: {file_path}")
            return False
        
    def destroy(self):
        self.gym.destroy_sim(self.sim)

    def get_target_image(self):
        image = self.get_image()
        with torch.no_grad():
            self.target_image = self.generate_target(image)

            # check_image = self.target_image / 1.6

            image = check_image[:, 0]*200

            depth_predict = image[0].to('cpu').detach().numpy().copy()
            depth_predict = Image.fromarray((depth_predict).astype(np.uint8))
            depth_predict.save(f"/home/engawa/py_ws/visual_servo/src/network/target.png")


            self.generate_target = self.generate_target.to('cpu')

            self.target_image = self.target_image[:, 0].unsqueeze(1)

        # return target_image
    
    def stereo_reaching(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        handle_position = torch.stack([self.stereo(x) for x in range(self.test_num_envs)]).to("cuda")

        handle_position = self.rigid_body_states[:, self.closet_handle][:, :3].clone().to("cuda")

        goal_position = handle_position[:self.test_num_envs, :] - 0.001

        # self.rendaring_sphere(handle_position, 0)
        # self.gym.write_viewer_image_to_file(self.viewer, join(self.record_frames_dir, f"reaching/caluculate_position.png"))


        # goal_position = handle_position.clone()
        # goal_position[:, 0] += 0.03
        # goal_position[:, 1] += -0.015
        # # goal_position[:, 2] += 0.03
        # # goal_position[:, 3] = 1.57

        half_position = goal_position.clone()
        half_position[:, 1] += -0.08

        current_position = self.rigid_body_states[:self.test_num_envs, self.hand_handle][:, 0:3].clone().to('cuda')
        hand_position = self.rigid_body_states[:self.test_num_envs, self.hand_handle][:, 0:3]

        # dt = (half_position - current_position)/50 
        dt_half = (half_position - current_position)/20
        dt_goal = (goal_position - half_position)/5        

        # for i in range(100):
        #     self.gym.refresh_dof_state_tensor(self.sim)
        #     # # d = dt_half * i
        #     target = self.ur_dof_state[:self.test_num_envs, :3, 0].clone() + dt_half.to("cpu")
        #     current = self.ur_dof_state[:self.test_num_envs, :3, 0].clone()
        #     diff = torch.sqrt(torch.sum((target - current)**2))

        #     while diff > 0.0001:
        #         self.ur_dof_state[:self.test_num_envs, :3, 0] += dt_half.to("cpu")
        #         multi_env_ids_int32 = self.global_indices[:, :2].flatten().to('cpu')

        #         self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                             gymtorch.unwrap_tensor(self.dof_state),
        #                                             gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
                
        #         self.gym.simulate(self.sim)
        #         self.gym.step_graphics(self.sim)
        #         self.gym.draw_viewer(self.viewer, self.sim, True)

        #         self.gym.refresh_dof_state_tensor(self.sim)

        #         current = self.ur_dof_state[:self.test_num_envs, :3, 0].clone()
        #         diff = torch.sqrt(torch.sum((target - current)**2))

        #     self.ur_dof_state[:self.test_num_envs, :3, 0] += dt_half.to("cpu")
        #     multi_env_ids_int32 = self.global_indices[:, :2].flatten().to('cpu')

        #     self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                         gymtorch.unwrap_tensor(self.dof_state),
        #                                         gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
            
        #     self.gym.simulate(self.sim)
        #     self.gym.step_graphics(self.sim)
        #     self.gym.draw_viewer(self.viewer, self.sim, True)
        #     self.gym.refresh_dof_state_tensor(self.sim)
        #     hand_position = self.rigid_body_states[:self.test_num_envs, self.hand_handle][:, 0:3]

        #     # self.gym.write_viewer_image_to_file(self.viewer, join(self.record_frames_dir, f"reaching/{i}.png"))

        #     # self.gym.simulate(self.sim)
        #     # self.gym.step_graphics(self.sim)
        #     # self.gym.draw_viewer(self.viewer, self.sim, True)

        for i in range(21):
            # d = dt_half * i
            self.ur_dof_state[:self.test_num_envs, :3, 0] += dt_half.to("cpu")
            multi_env_ids_int32 = self.global_indices[:, :2].flatten().to('cpu')

            self.gym.set_dof_state_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.dof_state))
            
            self.gym.simulate(self.sim)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            time.sleep(0.1)
            now = time.time() 
            self.gym.write_viewer_image_to_file(self.viewer, f"/home/engawa/py_ws/visual_servo/src/network/images_sice/result/{now}.png")
    
            # self.gym.write_viewer_image_to_file(self.viewer, join(self.record_frames_dir, f"reaching/{i}.png"))

        for i in range(5):
            # d = dt_goal * (i+1)
            self.ur_dof_state[:self.test_num_envs, :3, 0] += dt_goal.to("cpu")
            multi_env_ids_int32 = self.global_indices[:, :2].flatten().to('cpu')

            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.dof_state),
                                                gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
            
            
            self.gym.simulate(self.sim)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            time.sleep(0.1)

            now = time.time() 
            self.gym.write_viewer_image_to_file(self.viewer, f"/home/engawa/py_ws/visual_servo/src/network/images_sice/result/{now}.png")
        
        self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        # # self.gym.write_viewer_image_to_file(self.viewer, join(self.record_frames_dir, f"reaching/half_position.png"))

        self.gym.simulate(self.sim)

        # self.ur_dof_targets[:, :3] = goal_position
        # self.ur_dof_targets[self.test_num_envs:, :] = torch.zeros_like(self.ur_dof_targets[self.test_num_envs:])
        self.ur_dof_targets[:self.test_num_envs, :3] =  goal_position
        self.gym.set_dof_position_target_tensor(self.sim,
                                        gymtorch.unwrap_tensor(self.ur_dof_targets))
        
        # self.gym.write_viewer_image_to_file(self.viewer, join(self.record_frames_dir, f"reaching/goaL.png"))
        
        self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        # self.gym.write_viewer_image_to_file(self.viewer, join(self.record_frames_dir, f"reaching/targetposition.png"))

        # self.gym.simulate(self.sim)
        # self.gym.step_graphics(self.sim)
        # self.gym.draw_viewer(self.viewer, self.sim, True)

        # self.gym.simulate(self.sim)

        print("aa")

    def stereo(self, envs):
        envs += self.test_num_envs
        self.gym.render_all_camera_sensors(self.sim)
        color_image_right = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[envs], 2, gymapi.IMAGE_COLOR)).to(torch.float32).to(torch.float32)
        color_image_left = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[envs], 3, gymapi.IMAGE_COLOR)).to(torch.float32)

        color_image_right = color_image_right.view(color_image_right .shape[0],-1,4)[...,:3]
        color_image_left = color_image_left.view(color_image_left .shape[0],-1,4)[...,:3]

        # image = color_image_right.to('cpu').detach().numpy().copy()
        # # color_image /= 255
        # # color_image = color_image.permute(2, 0, 1)
        

        # # depth_unit = (depth).astype(np.uint8)
        # image = Image.fromarray((image).astype(np.uint8))
        # color_path = f"/home/engawa/py_ws/visual_servo/src/network/test.png"

        # image.save(color_path)

        color_image_right = color_image_right.permute(2, 0, 1)
        color_image_right = self.transform_attention(color_image_right)
        color_image_right = color_image_right.permute(1, 2, 0).to('cpu').detach().numpy().copy()

        color_image_left = color_image_left.permute(2, 0, 1)
        color_image_left = self.transform_attention(color_image_left)
        color_image_left = color_image_left.permute(1, 2, 0).to('cpu').detach().numpy().copy()


        pred = self.yolo.estimate_Bounding_box(color_image_left, color_image_right)

        target_position = self.yolo.stereo(pred)[0]

        self.hand_pos = self.rigid_body_states[:self.test_num_envs, self.hand_handle][:, 0:3].to("cuda")




        return target_position
    

    def get_delete_image(self):
        image = self.get_image()
        with torch.no_grad():
            # self.delete_ur = self.delete_ur.to('cuda')
            self.delete_image = self.delete_ur(image[0, :4].unsqueeze(0).to(torch.float32))
            # self.delete_image = torch.cat((self.delete_image, image[0, 4].unsqueeze(0).unsqueeze(1).to("cuda")), 1)
            check_image = self.delete_image 

            image = check_image[:, 0]*255

            depth_predict = image[0].to('cpu').detach().numpy().copy()
            depth_predict = Image.fromarray((depth_predict).astype(np.uint8))
            depth_predict.save(f"/home/engawa/py_ws/visual_servo/src/network/delete.png")

            image = cv2.imread("/home/engawa/py_ws/visual_servo/src/network/delete.png")
            cv2.imshow('camera', image)
            cv2.waitKey(10)

            self.delete_image *= 1.32

            #####################################cpu 切り替え
            # self.delete_ur = self.delete_ur.to('cpu')

        # return target_image
    

    def visual_servo(self, angle, i):

        handle_trajectory = self.handle_trajectory.clone()
        handle_trajectory = handle_trajectory[self.asset_list[0][0],:].squeeze()
        handle = handle_trajectory



        dif = torch.abs((handle[angle + 2, 1] - handle[angle, 1]) / handle[angle + 2, 0] - handle[angle, 0])
        if dif < 0.2:
            base = "x"
        else:
            base = "y"
        # # self.predict_action = self.predict_action.to('cuda')
        with torch.no_grad():
            action = self.predict_action(self.delete_image, self.target_image).clone()
        # self.predict_action = self.predict_action.to('cpu')
        self.delete_image = self.delete_image.to("cpu")
        action = action.to("cpu")
        target = self.ur_dof_state[:self.test_num_envs, :2, 0].clone() - action[:2]
        # self.ur_dof_state[:self.test_num_envs, :2, 0] -= action[:2]
        hand_position = self.ur_dof_state[:self.test_num_envs, :2, 0].clone()

        # difference = torch.sqrt(torch.sum((handle_trajectory[angle:, :2] - hand_position)**2, dim = 1))
        # target_angle = torch.argmin(difference)

        # # # if i < 23:
        if base == "x":
            hand_x = target[:self.test_num_envs, 0].clone()
            handle_x = (handle_trajectory[angle:, 0] - 0.01) - hand_x
            handle_id = torch.where(handle_x < 0, 100, handle_x)
            target_angle = torch.argmin(handle_id) + angle
        # target_angle = torch.tensor([i*4])
        # else:
        #     hand_x = self.ur_dof_state[:self.test_num_envs, 0, 1].clone()
        #     handle_x = (handle_trajectory[angle-10:, 1] - 0.01) - hand_x
        #     handle_id = torch.where(handle_x < 0, 100, handle_x)
        #     target_angle = torch.argmin(handle_id) + angle
        #     target_angle = torch.tensor([angle +4])
        else:
            hand_x = target[:self.test_num_envs, 1,].clone()
            handle_x = (handle_trajectory[angle:, 1] - 0.01) - hand_x
            handle_id = torch.where(handle_x < 0, 100, handle_x)
            target_angle = torch.argmin(handle_id) + angle

        # target_angle = torch.ones(1) * i * 2
        right_rad = target_angle * 1.57/180
        left_rad = right_rad * -2
        angle = torch.stack([right_rad, left_rad], dim = 0)
        self.closet_dof_state[:, :2, 0] = angle
        

        # current_handle_position = self.ur_dof_state[:self.test_num_envs, :3, 0]
        # trajectory_position =         

        
        # multi_env_ids_int32 = self.global_indices[:, :2].flatten().to('cpu')

        ########################closet の開閉角度設定


        ########################ur を設定
        dif = 1
        while dif > 0.001:
            self.gym.refresh_dof_state_tensor(self.sim)
            self.ur_dof_state[:self.test_num_envs, :2, 0] = target
            self.closet_dof_state[:, :2, 0] = angle
            self.gym.set_dof_state_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.dof_state))
            
            # self.trajectory_list.append(target_angle)
            self.gym.simulate(self.sim)
            # self.gym.step_graphics(self.sim)
            # self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            hand_pose = self.ur_dof_state[:self.test_num_envs, :2, 0].clone()

            dif = torch.sqrt(torch.sum((target - hand_pose)**2, dim = 1))



        self.gym.refresh_dof_state_tensor(self.sim)
        self.hand_pos = self.rigid_body_states[:self.test_num_envs, self.hand_handle][:, 0:3].to("cuda")
        self.gym.refresh_dof_state_tensor(self.sim)
        self.ur_dof_state[:self.test_num_envs, 3:6, 0] = torch.tensor([0, 0, 0])

        self.gym.set_dof_state_tensor(self.sim,
                                            gymtorch.unwrap_tensor(self.dof_state))
        
        # self.trajectory_list.append(target_angle)
        self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        now = time.time() 
        self.gym.write_viewer_image_to_file(self.viewer, f"/home/engawa/py_ws/visual_servo/src/network/images_sice/result/{now}.png")

        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.hand_pos = self.rigid_body_states[:self.test_num_envs, self.hand_handle][:, 0:3].to("cuda") 
        self.handle_pos = self.rigid_body_states[:self.test_num_envs, self.closet_handle][:, 0:3].to("cuda")

        # hand_x = self.ur_dof_state[:self.test_num_envs, 0, 0].clone()
        # handle_x = (handle_trajectory - 0.01) - hand_x
        # handle_id = torch.where(handle_x < 0, 100, handle_x)
        # target_angle = torch.argmin(handle_id)
        # # target_angle = torch.tensor([i*4])

        # right_rad = target_angle * 1.57/180
        # left_rad = right_rad * -2
        # angle = torch.stack([right_rad, left_rad], dim = 0)
        # self.closet_dof_state[:, :2, 0] = angle

        # self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                     gymtorch.unwrap_tensor(self.dof_state),
        #                                     gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
        
        # # self.trajectory_list.append(target_angle)
        # self.gym.simulate(self.sim)
        # self.gym.step_graphics(self.sim)
        # self.gym.draw_viewer(self.viewer, self.sim, True)
        step = True

        if target_angle > 170:
            step = False

        print(target_angle)
        
        return step, target_angle
        









        
    def get_image(self):
        self.gym.render_all_camera_sensors(self.sim)
        init_image = torch.zeros([self.test_num_envs, 5, 256, 256])
        # test = self.num_envs // 2
        # directly_list = ["door_with_hook", "door_only"]
        for l in range(self.test_num_envs):
            # attention = torch.zeros([480, 640])
            color_image_left = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[l], 1, gymapi.IMAGE_DEPTH)).to(torch.float32)
            color_image = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[l], 1, gymapi.IMAGE_COLOR)).to(torch.float32)
            color_image = color_image.view(color_image .shape[0],-1,4)[...,:3]
            # image = color_image.to('cpu').detach().numpy().copy()

            # # image = color_image.to('cpu').detach().numpy().copy()
            # # image /= 255
            # image = Image.fromarray((image).astype(np.uint8))
            # color_path = f"/home/engawa/py_ws/visual_servo/src/network/test.png"

            # image.save(color_path)
            color_image /= 255
            color_image = color_image.permute(2, 0, 1)
            

            # depth_unit = (depth).astype(np.uint8)
            # image = Image.fromarray((image).astype(np.uint8))
            # color_path = f"/home/engawa/py_ws/visual_servo/src/network/test.png"

            # image.save(color_path)

            # color.save(color_path)

            color_image_left *= -1

            init_image[l, :3] = self.transform(color_image)
            init_image[l, 3] = self.transform(color_image_left.unsqueeze(0))

        for l in range(self.test_num_envs):
            self.gym.render_all_camera_sensors(self.sim)
            color_image = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[l + self.test_num_envs], 1, gymapi.IMAGE_COLOR)).to(torch.float32)
            color_image = color_image.view(color_image .shape[0],-1,4)[...,:3]
            image = color_image.to('cpu').detach().numpy().copy()
            # image /= 255
            image = Image.fromarray((image).astype(np.uint8))
            color_path = f"/home/engawa/py_ws/visual_servo/src/network/test_black.png"
            image.save(color_path)
            

            # image.save(color_path)
            color_image = color_image.permute(2, 0, 1)
            color_image = self.transform_attention(color_image)
            color_image = color_image.permute(1, 2, 0).to('cpu').detach().numpy().copy()

            init_image[l, 4] = self.yolo.make_attension(color_image)

        return init_image
    
    def get_image_target(self):

        self.gym.render_all_camera_sensors(self.sim)
        init_image = torch.zeros([self.test_num_envs, 5, 256, 256])
        # test = self.num_envs // 2
        # directly_list = ["door_with_hook", "door_only"]
        for l in range(self.test_num_envs):
            # attention = torch.zeros([480, 640])
            color_image_left = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[l], 1, gymapi.IMAGE_DEPTH)).to(torch.float32)
            color_image = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[l], 1, gymapi.IMAGE_COLOR)).to(torch.float32)
            color_image = color_image.view(color_image .shape[0],-1,4)[...,:3]
            image = color_image.to('cpu').detach().numpy().copy()

            # image = color_image.to('cpu').detach().numpy().copy()
            # image /= 255
            image = Image.fromarray((image).astype(np.uint8))
            color_path = f"/home/engawa/py_ws/visual_servo/src/network/test.png"

            image.save(color_path)
            color_image /= 255
            color_image = color_image.permute(2, 0, 1)
            

            # depth_unit = (depth).astype(np.uint8)
            # image = Image.fromarray((image).astype(np.uint8))
            # color_path = f"/home/engawa/py_ws/visual_servo/src/network/test.png"

            # image.save(color_path)

            # color.save(color_path)

            color_image_left *= -1

            init_image[l, :3] = self.transform(color_image)
            init_image[l, 3] = self.transform(color_image_left.unsqueeze(0))

        for l in range(self.test_num_envs):
            self.gym.render_all_camera_sensors(self.sim)
            color_image = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[l + self.test_num_envs], 1, gymapi.IMAGE_COLOR)).to(torch.float32)
            color_image = color_image.view(color_image .shape[0],-1,4)[...,:3]
            # image = color_image.to('cpu').detach().numpy().copy()
            # # image /= 255
            # image = Image.fromarray((image).astype(np.uint8))
            # color_path = f"/home/engawa/py_ws/visual_servo/src/network/test.png"

            # image.save(color_path)
            color_image = color_image.permute(2, 0, 1)
            color_image = self.transform_attention(color_image)
            color_image = color_image.permute(1, 2, 0).to('cpu').detach().numpy().copy()

            init_image[l, 4] = self.yolo.make_attension(color_image)

        return init_image


    def save_image(self, image_array, path):
        if not isinstance(image_array, np.ndarray):
            image_array = image_array.to('cpu').clone().numpy().copy()
        image = Image.fromarray((image_array).astype(np.uint8))
        path_name = f"/home/engawa/py_ws/visual_servo/src/network/predict_opening_door/dataset/predict_opening/images/{path}.png"
        image.save(path_name)

    def check_image(self, path):
        path_name = f"/home/engawa/py_ws/visual_servo/src/network/predict_opening_door/dataset/predict_opening/images/{path}.png"
        image = cv2.imread(path_name)
        cv2.imshow('color', image) 
        cv2.waitKey(0)

    def simulate(self):
        self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)
        # # self.gym.draw_viewer(self.viewer, self.sim, True)

    def to_torch(self, x, dtype=torch.float, device='cuda:0', requires_grad=False):
        return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


if __name__ == "__main__": 
    rend = rendar()
    # closet_angle_list = torch.zeros([rend.num_envs, 51])
    # for i in range(closet_angle_list.shape[0]):
    #         # li = random.sample(range(180), k=36)
    #         # closet_angle_list[i, 1:] = torch.tensor(sorted(random.sample(range(10, 180), k=50)))
    #         # closet_angle_list[i, 0] = 0
    #         closet_angle_list[i, :-1] = torch.tensor(sorted(random.sample(range(170), k=50)))
    #         closet_angle_list[i, -1] = 180

    rend.create_sim()
    rend.create_ground_plane()
    rend.create_envs()
    rend.get_target_image()
    step = True
    # for i in range(10000):
    #     rend.pre_physics_step_brank()

    for i in range(100):
        rend.pre_physics_step_brank()
    rend.stereo_reaching()
    target_angle = 0
    i = 0
    while step:
        # i += 1
        delete_image = rend.get_delete_image()
        step, target_angle = rend.visual_servo(target_angle, i)
        i += 1
    # rend.get_feature_and_depth(0)
    # for i in range(100000000000000):
    #     rend.pre_physics_step_brank()





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
from pyquaternion import Quaternion

from isaacgym import gymutil, gymtorch, gymapi
import isaacgym.torch_utils as torch_utils
# import isaacgym.torch_utils as torch_utils
# from utils_another.build_yolo import YOLO
# from plane_estimation import Depth
from make_urdf import edit_urdf_real, edit_urdf_structure, edit_urdf_handle_position
from model import ivis, goal_image_generator, manipulator_remover, action_generator

import torch
from torchvision import transforms
import sys

class rendar():
    def __init__(self, num_envs = 1):
        # self.closet_id = 2
        
        self.num_envs = num_envs
        self.test_num_envs = self.num_envs
        self.device = 'cuda'
        self.num_dofs = 8
        self.physics_engine = gymapi.SIM_PHYSX
        self.ur_default_dof_pos = self.to_torch([-0.2, 0, 0.8, 0, 0, 0], device=self.device)
        self.ur_default_dof_pos = self.to_torch([0.05, -1.8, 1.2, -2.4, -1.57, 0], device=self.device)
        self.asset_path_list = []
        self.closet_asset_list = [] 

        self.transform = transforms.Resize((224, 224))
        self.transform_256 = transforms.Resize((256, 256))

        # print("good")
        self.object_count_list = []
        self.handle_number_list = []

        self.repeat = 0
        self.number = self.repeat * self.num_envs 

        # self.closet_position = torch.load("/home/engawa/py_ws/visual_servo/src/network/predict_opening_door/dataset/delete_UR/rgbd_rgbd/base/closet_position.pth")

        self.goal_generator = goal_image_generator()
        self.remover = manipulator_remover()
        self.action_generator = action_generator()

        self.goal_generator.load_state_dict(self.load_weight("log/generator_many_1214/weight/weights_last.pth"))
        self.remover.load_state_dict(self.load_weight("log/remover_sim_ur_2/weight/weights_last.pth"))
        self.action_generator.load_state_dict(self.load_weight("log//weight/transfer.pth"))

        self.goal_generator = self.goal_generator.to("cuda")
        self.remover = self.remover.to("cuda")
        self.action_generator = self.action_generator.to("cuda")

        self.handle_trajectory = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/action_1217/handle_trajectory.pth")
        
        self.handle_trajectory = self.handle_trajectory.view(800, 181, 6)
        self.closet_size = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/generator_1214/closet_size.pt")
        
        self.closet_size = self.closet_size.unsqueeze(1).repeat(1, 5, 1)
        self.closet_size = self.closet_size.view(800, 2)
        self.closet_position = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/generator_1214/closet_posistion.pt")
        # print(f"-------------------{self.closet_position.shape}")
        self.closet_position = self.closet_position.view(800, -1)
        # print(f"-------------------{self.closet_position.shape}")
        self.wall_position = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/generator_1214/wall_position.pt")
        self.wall_position = self.wall_position.unsqueeze(1).repeat(1, 5).view(800)

        self.number = random.randint(0, 800)
        self.number = torch.argmax(self.closet_size[:, 0])
        print(torch.max(self.closet_size[:, 0]))

        # self.number = 759

        self.handle_trajectory = self.handle_trajectory[self.number]

        self.repeat_number = 5


        # self.define_closetprops()

    # def define_closetprops(self):
        # self.closet_position = torch.zeros(self.num_envs, 10, 2)
        # # closet_rot = torch.zeros(160, 5, 41)
        # self.closet_size = torch.zeros(self.num_envs, 2)
        # self.wall_position = torch.zeros(self.num_envs)
        # y_max = [0.28, 0.2, 0.1, 0.02, -0.07]
        # y_min = [-0.45, -0.4, -0.3, -0.25, -0.18]
        # y_base = -0.1
        # x_base = -0.95

        # for i in range(self.closet_position.shape[0]):
        #     id = i%4
        #     self.wall_position[i] = random.uniform(-1.8, -1.55) 
        #     if id == 0:
        #         self.closet_size[i, 1] = random.uniform(0.8, 1.0)
        #     elif id == 1:
        #         self.closet_size[i, 1] = random.uniform(1.0, 1.2)
        #     elif id == 2:
        #         self.closet_size[i, 1] = random.uniform(1.2, 1.4)
        #     elif id == 3:
        #         self.closet_size[i, 1] = random.uniform(1.4, 1.6)
            
        #     self.closet_size[i, 0] = random.uniform(0.75, 1.15)


        #     for j in range(self.closet_position.shape[1]):
        #         self.closet_position[i, j, 0] = random.uniform(self.wall_position[i] + 0.6, -0.8)

        #         dy_max = (y_max[id+1] - 0.1) * (self.closet_position[i, j, 0]/-0.95)
        #         dy_min = (0.1 - y_min[id+1]) * (self.closet_position[i, j, 0]/-0.95)

        #         self.loset_position[i, j, 1] = random.uniform(0.1 - dy_min, dy_max - 0.1)

        # self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # self.out = cv2.VideoWriter('result.mp4', self.fourcc, 30.0, (640, 480))

        

    def load_weight(self, path):
        weight = torch.load(path, map_location='cuda')

        return weight


    def create_sim(self, compute_device_id = 0, graphics_device_id = 0):
        self.gym = gymapi.acquire_gym()
        # self.num_envs= 1
        

        self.args = gymutil.parse_arguments(description="Joint control Methods Example")
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity.x = 0
        sim_params.gravity.y = 0
        sim_params.gravity.z = -9.81
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

    def create_envs(self, i = 1, spacing = 2, num_per_row = 5):
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        self.cam_pos = gymapi.Vec3(0.9, 2.0, 1.0)
        self.cam_target = gymapi.Vec3(-0.5, 0, 0.5)
        # self.cam_pos = gymapi.Vec3(0.5, 0, 1.5)
        # self.cam_target = gymapi.Vec3(-0.8, 0, 1.0)
        # self.cam_pos = gymapi.Vec3(0.5, 0.6, 1.0)
        # self.cam_target = gymapi.Vec3(-0.2, 0, 1.0)
        self.cam_pos = gymapi.Vec3(0.9, -0.6, 1.8)
        self.cam_target = gymapi.Vec3(-0.8, -0.2, 1.0)

        self.gym.viewer_camera_look_at(self.viewer, None, self.cam_pos, self.cam_target)
        # # if i == 1:
        #     self.gym.viewer_camera_look_at(self.viewer, None, self.cam_pos, self.cam_target)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets")
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "/home/engawa/py_ws/visual_servo/src/network/predict_opening_door/dataset/predict_opening_wall/assets"
        ur_asset_file = "urdf/ur_description/urdf/ur5.urdf"



        for a in range(self.num_envs):
            # number = a + self.number
            handle_position = self.closet_size[self.number, 0]
            # handle_position = 1.15
            scale = self.closet_size[self.number, 1]
            edit_urdf_handle_position(scale, a, handle_position)

            file_path = f'urdf/closet_for_real/urdf_env/{a}.urdf'
            self.asset_path_list.append(file_path)

        asset_options = gymapi.AssetOptions()

        asset_box = self.gym.create_box(self.sim, 0.8, 6.0, 5, asset_options)
        asset_base = self.gym.create_box(self.sim, 0.4, 0.4, 0.2, asset_options)

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
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.use_mesh_materials = True

        closet_asset = []
        for i in range(len(self.asset_path_list)):
            closet_asset.append(self.gym.load_asset(self.sim, asset_root, self.asset_path_list[i], asset_options))

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
                ur_dof_props['effort'][i] = 1000000000
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

        ur_start_pose = gymapi.Transform()
        ur_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        ur_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0, 1), np.radians(0))
        ur_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0, 1), np.radians(180))

        closet_start_pose = gymapi.Transform()
        closet_start_pose.p = gymapi.Vec3(-0.85, 0, 0)
        closet_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0, 1), np.radians(0))

        self.camera_props1 = gymapi.CameraProperties()
        # self.camera_props.horizontal_fov = 75.0
        self.camera_props1.width = 1280
        self.camera_props1.height = 720
        self.camera_props1.horizontal_fov = 80.796
        self.camera_props1.enable_tensors = True

        self.camera_pos_left  = gymapi.Vec3(0.,-0.1,1)
        # self.camera_pos_left  = gymapi.Vec3(0.3,-0.,1)

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

        for i in range(self.num_envs):
            self.env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            asset_id = i

            ur_pose = ur_start_pose
            ur_pose.p.x = 1
            ur_pose.p.y = -0.3
            # ur_pose.p.z = 0.4
            ur_pose.p.z = 0.3
            self.ur_actor = self.gym.create_actor(self.env_ptr, ur_asset, ur_start_pose, "ur", i, 0, 0)
            self.gym.set_actor_dof_properties(self.env_ptr, self.ur_actor, ur_dof_props)

            closet_pose = closet_start_pose
            # closet_position = torch.tensor([random.uniform(-0.7, -0.5), random.uniform(-0.15, 0.15)]).to("cuda")
            closet_pose.p.x = self.closet_position[self.number, 0]
            # print(self.closet_position[asset_id + self.number, 0, 0])
            # closet_pose.p.x = -0.95
            closet_pose.p.y = self.closet_position[self.number, 1]
            # closet_pose.p.x = -1.2
            # closet_pose.p.y = -0.
            closet_pose.p.z = 0

            # self.closet_list.append(closet_position)
            self.closet_actor = self.gym.create_actor(self.env_ptr, closet_asset[asset_id], closet_pose, "closet", i, 0, 0)
            self.gym.set_actor_dof_properties(self.env_ptr, self.closet_actor, closet_dof_props)

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(self.wall_position[self.number] - 0.4, 0, 1.25)
            # pose.p = gymapi.Vec3(-1.95, 0, 1.25)
            # pose.p = gymapi.Vec3(-1.4, 0, 1.25)
            pose.r = gymapi.Quat(0, 0, 0, 1)
            box_handle = self.gym.create_actor(self.env_ptr, asset_box, pose, "actor1", i, 0)
            shape_props = self.gym.get_actor_rigid_shape_properties(self.env_ptr, box_handle)
            shape_props[0].restitution = 1
            shape_props[0].compliance = 0.5

            # pose_base = gymapi.Transform()
            # pose_base.p = gymapi.Vec3(0, 0, 0.15)
            # # pose.p = gymapi.Vec3(-1.95, 0, 1.25)
            # # pose.p = gymapi.Vec3(-1.4, 0, 1.25)
            # pose_base.r = gymapi.Quat(0, 0, 0, 1)
            # box_handle_base = self.gym.create_actor(self.env_ptr, asset_base, pose_base, "base", i, 0)
            # shape_props_base = self.gym.get_actor_rigid_shape_properties(self.env_ptr, box_handle_base)
            # shape_props_base[0].restitution = 1
            # shape_props_base[0].compliance = 0.5

            self.gym.set_actor_rigid_shape_properties(self.env_ptr, box_handle, shape_props)
            self.gym.set_rigid_body_color(self.env_ptr, box_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1., 1., 1.))
            
            self.envs.append(self.env_ptr)
            # self.urs.append(self.ur_actor)
            # self.closets.append(self.closet_actor)
            self.camera_handle.append(self.gym.create_camera_sensor(self.envs[i], self.camera_props1))
            # self.camera_handle.append(self.gym.create_camera_sensor(self.envs[i], self.camera_props2))
            
            # self.gym.set_camera_location(self.camera_handle[0], self.env_ptr, self.camera_pos_right, self.camera_rot_right)
            self.gym.set_camera_location(self.camera_handle[i], self.env_ptr, self.camera_pos_left, self.camera_rot_left)

            self.object_count_list.append(self.gym.get_asset_rigid_body_count(closet_asset[asset_id]) + self.gym.get_asset_rigid_body_count(ur_asset) + 1)
            self.handle_number_list.append(self.gym.find_actor_rigid_body_handle(self.env_ptr, self.closet_actor, "handle1"))
        

        # self.hand_handle = self.gym.find_actor_rigid_body_handle(self.env_ptr, self.ur_actor, "hook")
        self.closet_door = self.gym.find_actor_rigid_body_handle(self.env_ptr, self.closet_actor, "closet_door")
        self.closet_handle = self.gym.find_actor_rigid_body_handle(self.env_ptr, self.closet_actor, "handle1")

        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor)

        # print(self.ur_dof_targets.shape)

        self.ur_dof_targets[:self.test_num_envs, :self.num_ur_dofs] = self.ur_default_dof_pos.to("cpu")

        self.gym.set_dof_position_target_tensor(self.sim,
                                        gymtorch.unwrap_tensor(self.ur_dof_targets))

        self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        # self.render()

        self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)


        self.ini()

    def ini(self):
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # self.gym.refresh_dof_state_tensor(sim)
        # self.gym.refresh_dof_state_tensor(sim)
        self.dof_state = gymtorch.wrap_tensor(self.dof_state_tensor)
        self.gym.refresh_dof_state_tensor(self.sim)
        # self.ur_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_ur_dofs]
        # self.ur_dof_pos = self.ur_dof_state[..., 0]
        # self.ur_dof_vel = self.ur_dof_state[..., 1]
        self.closet_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_ur_dofs:]
        self.closet_dof_pos = self.closet_dof_state[..., 0]
        self.closet_dof_vel = self.closet_dof_state[..., 1]

        self.global_indices = torch.arange(self.num_envs * (2 + 0), dtype=torch.int32, device="cuda").view(self.num_envs, -1)

        self.ur_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_ur_dofs]
        self.ur_dof_pos = self.ur_dof_state[..., 0]
        self.ur_dof_vel = self.ur_dof_state[..., 1]
        self.closet_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_ur_dofs:self.num_ur_dofs+2]
        self.closet_dof_pos = self.closet_dof_state[..., 0]
        self.closet_dof_vel = self.closet_dof_state[..., 1]


        _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_tensor = gymtorch.wrap_tensor(_root_tensor).view(self.num_envs, -1, 13)

        # self.box_root_state = self.root_tensor.view(self.num_envs, -1, 13)[:, 2]

        self._jacobian = self.gym.acquire_jacobian_tensor(self.sim, "ur")
        self.jacobian = gymtorch.wrap_tensor(self._jacobian)
        # print(self.jacobian.shape)
        self.gym.refresh_dof_state_tensor(self.sim)

    def set_closet_position(self, i):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        self.root_tensor[:, 1, :2] = self.closet_position[self.number, :]
        # self.root_tensor[:, 1, 1] = 0
        # self.root_tensor[:, 1, 0] = -1.2
        # self.root_tensor[:, 2, 0] = self.wall_position[:, i] - 0.4
 
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_tensor))
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

    def set_ur_position(self, i):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        # self.root_tensor[:, 0, 0] = -0.3
        self.root_tensor[:, 0, 0] = self.closet_position[self.number, 0] + 0.7
        self.root_tensor[:, 0, 1] = self.closet_position[self.number, 1] - 0.05
        
        # self.root_tensor[:, 2, 0] = self.wall_position[:, i] - 0.4
 
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_tensor))
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        # self.gym.write_viewer_image_to_file(self.viewer, "/home/engawa/py_ws/visual_servo/src/network/sim2real/image/videot{self.repeat_number}est.png")

##############################################################
##########################################################3####変える
    def pre_physics_step(self, step, position):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        # degree = self.closet_angle_list[:, i]

        # for i in range(self.num_envs):

        #     id = i * 3 + 2

        #     # self.root_tensor[id, 0] = -1.25
        #     # self.root_tensor[id, 1] = 0
        #     self.root_tensor[id, 3:7] = torch.tensor([0, 0, 0, 1])
        #     self.root_tensor[id, 7:] = 0

        self.root_tensor[:, 2, 3:7] = torch.tensor([0, 0, 0, 1])
        self.root_tensor[:, 2, 7:] = 0

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_tensor))

        # if step == 40:
        #     target_angle = torch.ones([50]) * 180
        # else:
        target_angle = self.closet_rot[self.number:(self.number + self.num_envs), position, step] 
        # target_angle += 10

        right_rad = target_angle * 1.57/180
        left_rad = right_rad * -2
        angle = torch.stack([right_rad, left_rad], dim = 1)


        # self.ur_dof_targets[:self.test_num_envs, :2] = self.actions.to("cpu")

        self.closet_dof_state[:, :2, 0] = angle
        # multi_env_ids_int32 = self.global_indices[:40, :2].flatten().to('cpu')

        # self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                     gymtorch.unwrap_tensor(self.dof_state),
        #                                     gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor(self.sim,
                                            gymtorch.unwrap_tensor(self.dof_state))
        
        self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        

        # self.get_image(step, position)

    def set_ur(self):

        self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        ###handle_position の設定
        hand_pos = torch.zeros([self.num_envs, 3])
        ee_pos = torch.zeros([self.num_envs, 3])

        for j in range(self.num_envs):
            hand_pos[j] = self.rigid_body_states[sum(self.object_count_list[:j]) + self.handle_number_list[j], 0:3].to("cuda")
            ee_pos[j] = self.rigid_body_states[sum(self.object_count_list[:j]) + 9, 0:3].to("cuda") ######## enc = 0 以外は9かも
        # hand_rot = self.rigid_body_states[:self.test_num_envs, self.closet_handle][:, 3:7]

        

        hand_pos[:, 0] -= 0.01
        hand_pos[:, 2] += 0.012
        
        step_number = 10
        dd = (hand_pos - ee_pos)/10

        for i in range(step_number):
            goal_pose = ee_pos + (i + 1) * dd
            self.step_ur(ee_pos, goal_pose, i)

            self.gym.step_graphics(self.sim)
        # self.get_image(i, 0)
            self.gym.draw_viewer(self.viewer, self.sim, True)


        # self.gym.step_graphics(self.sim)
        # # self.get_image(i, 0)
        # self.gym.draw_viewer(self.viewer, self.sim, True)

    def step_ur(self, current_position, goal_position, i):
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        difference = torch.sqrt(torch.sum((current_position-goal_position)**2))
        goal_position =  goal_position.to("cuda")
        while difference > 0.001:
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            ja = self.jacobian[:self.test_num_envs, 8, :, :].to("cuda")
            # print(ja.shape)
            current_position = torch.zeros([self.num_envs, 3]).to("cuda")
            current_orientation = torch.zeros([self.num_envs, 4]).to("cuda")

            # print(goal_position)
            for j in range(self.num_envs):
                current_position[j] = self.rigid_body_states[sum(self.object_count_list[:j]) + 9, 0:3].to("cuda")
                current_orientation[j] = self.rigid_body_states[sum(self.object_count_list[:j]) + 9, 3:7].to("cuda")

            d_dof = self.ik(ja, current_position, current_orientation, goal_position).to("cpu")
            self.ur_dof_targets[:self.test_num_envs, :self.num_ur_dofs] += d_dof[:, 3:]
            
            self.root_tensor[:, 0, :3] += d_dof[:, :3]
        
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_tensor))
            self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.ur_dof_targets))
        
            self.gym.simulate(self.sim)

            difference = torch.sqrt(torch.sum((current_position-goal_position)**2))
            # print(difference)
        
        # self.gym.step_graphics(self.sim)
        # # self.get_image(i, 0)
        # self.gym.draw_viewer(self.viewer, self.sim, True)
        # time.sleep(0.01)



    def ik(self, jacobian_end_effector: torch.Tensor,
        current_position: torch.Tensor,
        current_orientation: torch.Tensor,
        goal_position: torch.Tensor,
        goal_orientation: Optional[torch.Tensor] = None,
        damping_factor: float = 0.05,
        squeeze_output: bool = True) -> torch.Tensor:

        if goal_orientation is None:
            goal_orientation = torch.zeros([self.num_envs, 4]).to("cuda")
            goal_orientation[:] = torch.tensor([0, 0, 0, 1])

        jacco = torch.zeros([self.num_envs, 6, 2]).to("cuda")
        jacco[:, 0, 0] = 0.02
        jacco[:, 1, 1] = 0.02

        jacobian_end_effector = torch.cat([jacco, jacobian_end_effector], dim = 2)

        current_position = current_position.to("cuda")

        # compute error
        q = torch_utils.quat_mul(goal_orientation, torch_utils.quat_conjugate(current_orientation)).to("cuda")
        # print(q)
        error = torch.cat([goal_position - current_position,  # position error
                        q[:, 0:3] * torch.sign(q[:, 3]).unsqueeze(-1)],  # orientation error
                        dim=-1).unsqueeze(-1)

        transpose = torch.transpose(jacobian_end_effector, 1, 2)
        lmbda = torch.eye(6, device=jacobian_end_effector.device) * (damping_factor ** 2)
        if squeeze_output:
            return (transpose @ torch.inverse(jacobian_end_effector @ transpose + lmbda) @ error).squeeze(dim=2)
        else:
            return transpose @ torch.inverse(jacobian_end_effector @ transpose + lmbda) @ error
        

    def generate_goal_image(self):
        image = self.get_image()
        with torch.no_grad():
            image = image.repeat(1, 3, 1, 1).to("cuda")
            self.goal_image = self.transform_256(self.goal_generator(image.to("cuda")))

            self.goal_image = self.goal_image.squeeze()

    def generate_current_image(self, i):
        image = self.get_image(i)


        with torch.no_grad():
            image = image.repeat(1, 3, 1, 1).to("cuda")
            self.remover_image = self.transform_256(self.remover(image.to("cuda")))
            test_depth = self.remover_image.squeeze().to('cpu').detach().numpy().copy()
            test_depth *= 255
            # color_image = color_image.permute(2, 0, 1)
            

            # depth_unit = (depth).astype(np.uint8)
            test_depth = Image.fromarray((test_depth).astype(np.uint8))
            
            # test_depth.save(f"/home/engawa/py_ws/visual_servo/src/network/sim2real/image/video/{self.repeat_number}/current_{str(i).zfill(2)}.png")


            another = self.transform(self.remover_image)
            self.goal_image = self.transform_256(self.goal_generator(another.repeat(1, 3, 1, 1).to("cuda")))

            test_depth = self.goal_image.squeeze().to('cpu').detach().numpy().copy()
            test_depth *= 255
            # color_image = color_image.permute(2, 0, 1)
            

            # depth_unit = (depth).astype(np.uint8)
            test_depth = Image.fromarray((test_depth).astype(np.uint8))
            
            # test_depth.save(f"/home/engawa/py_ws/visual_servo/src/network/sim2real/image/video/{self.repeat_number}/target_{str(i).zfill(2)}.png")


    def generate_action(self):
        with torch.no_grad():
            self.action = self.action_generator(self.remover_image.to("cuda"), self.goal_image.to("cuda"))
            # print(self.action.shape)
        
        # for i in range(self.action.shape[0]):
            # self.action[i, :3] /= torch.sqrt(torch.sum(self.action[i, :3]**2))

        self.action[:3] /= torch.sqrt(torch.sum(self.action[:3]**2))
        self.action[:3] *= 0.01

        # print(self.action[:3])

        # print(self.action)


    def opening(self, angle, i, old_base):
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.write_viewer_image_to_file(self.viewer, f"/home/engawa/py_ws/visual_servo/src/network/sim2real/image/video/{self.repeat_number}/env_{str(i).zfill(2)}.png")

        print(self.viewer)

        if angle < 40:
            base = "x"
            self.action[1] = 0
        else:
            base = "y"
        
        current_position = torch.zeros([self.num_envs, 2])
        # for i in range(10000):
        #     self.pre_physics_step_brank()

        for j in range(self.num_envs):
            current_position[j] = self.rigid_body_states[sum(self.object_count_list[:j]) + 9, 0:2].to("cuda")

        goal_position = current_position + self.action[:2].to("cpu")
        # goal_position[:, :2] += self.action[:2].to("cpu")
        # print(current_position)
        # print(self.action)
        # print(goal_position)
        # # for i in range(10000):
        # #     self.pre_physics_step_brank()
        # print(self.handle_trajectory)
        # for i in range(10000):
        #     self.pre_physics_step_brank()


        self.gym.refresh_rigid_body_state_tensor(self.sim)

        difference = torch.sqrt(torch.sum((current_position-goal_position)**2))
        goal_position =  goal_position.to("cuda")

        dif = torch.abs((self.handle_trajectory[angle + 2, 1] - self.handle_trajectory[angle, 1]) / self.handle_trajectory[angle + 2, 0] - self.handle_trajectory[angle, 0])
        

        if base == "x":
            hand_x = goal_position[:self.test_num_envs, 0].clone().to("cpu")
            handle_x = (self.handle_trajectory[angle:, 0] - 0.01) - hand_x
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
            hand_x = goal_position[:self.test_num_envs, 1,].clone().to("cpu")
            handle_x = (self.handle_trajectory[angle:, 1] + 0.01) - hand_x
            handle_id = torch.where(handle_x < 0, 100, handle_x)
            target_angle = torch.argmin(handle_id) + angle

        if old_base == "x" and base == "y":
            df = torch.tensor([0.03, 0.01])
        else:
            df  = torch.tensor([0.0, 0.0])

        # print(self.handle_trajectory[torch.argmin(handle_id) + angle])
        # print(self.handle_trajectory)
        # print(goal_position)

        # target_angle = torch.ones(1) * i * 2
        # right_rad = target_angle * 1.57/180
        # left_rad = right_rad * -2
        # angle = torch.stack([right_rad, left_rad], dim = 0)
        # self.closet_dof_state[:, :2, 0] = angle
        # self.gym.set_dof_state_tensor(self.sim,
        #                                         gymtorch.unwrap_tensor(self.dof_state))
        
        # self.gym.simulate(self.sim)
        # self.gym.step_graphics(self.sim)
        # self.gym.draw_viewer(self.viewer, self.sim, True)
        
        current_position = torch.zeros([self.num_envs, 3]).to("cuda")
        current_orientation = torch.zeros([self.num_envs, 4]).to("cuda")

        # print(goal_position)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        for j in range(self.num_envs):
            current_position[j] = self.rigid_body_states[sum(self.object_count_list[:j]) + 9, 0:3].to("cuda")
            current_orientation[j] = self.rigid_body_states[sum(self.object_count_list[:j]) + 9, 3:7].to("cuda")

        
        # dd = 

        # print(current_position)

        # d = (self.action[:2] / 5).to("cuda")

        # for p in range(5):
        #     if p == 0:
        #         goal_position = current_position
        #     goal_position[:, :2] += d

        #     goal_position[:, 2] = self.hand_z

        #     if torch.isnan(goal_position).any().item():
        #         for _ in range(10000):
        #             self.pre_physics_step_brank()
        #     # print(f"before ========== {goal_position}")

        #     # print(f"goal ============ {goal_position}")
        
        goal_position = current_position.clone()
        goal_position[:, :2] += self.action[:2]

        goal_position[:, 2] = self.hand_z
        goal_position[:, 1] += 0.005

        goal_position[:, :2] += df.to("cuda")


        # print(current_position)
        # if i == 1:
        #     for _ in range(10000):
        #         self.pre_physics_step_brank()
        
        # print(self.action)
        # print(goal_position)
        if torch.isnan(current_position).any().item():
            for _ in range(10000):
                self.pre_physics_step_brank()

        i = 0
        while difference > 0.000001:
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            ja = self.jacobian[:self.test_num_envs, 8, :, :].to("cuda")
            # print(ja.shape)
            current_position = torch.zeros([self.num_envs, 3]).to("cuda")
            current_orientation = torch.zeros([self.num_envs, 4]).to("cuda")

            # print(goal_position)
            for j in range(self.num_envs):
                current_position[j] = self.rigid_body_states[sum(self.object_count_list[:j]) + 9, 0:3].to("cuda")
                current_orientation[j] = self.rigid_body_states[sum(self.object_count_list[:j]) + 9, 3:7].to("cuda")

            d_dof = self.ik(ja, current_position, current_orientation, goal_position).to("cpu")
            self.ur_dof_targets[:self.test_num_envs, :self.num_ur_dofs] += d_dof[:, 3:]
            
            self.root_tensor[:, 0, :3] += d_dof[:, :3]
        
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_tensor))
            # self.gym.set_dof_position_target_tensor(self.sim,
            #                                     gymtorch.unwrap_tensor(self.ur_dof_targets))
            # self.closet_dof_state[:, :2, 0] = angle
            
            self.gym.set_dof_position_target_tensor(self.sim,
                                            gymtorch.unwrap_tensor(self.ur_dof_targets))
            
            right_rad = target_angle * 1.57/180
            left_rad = right_rad * -2
            angle = torch.stack([right_rad, left_rad], dim = 0)
            self.closet_dof_state[:, :2, 0] = angle
            self.gym.set_dof_state_tensor(self.sim,
                                                    gymtorch.unwrap_tensor(self.dof_state))
        
            self.gym.simulate(self.sim)

            self.gym.refresh_rigid_body_state_tensor(self.sim)
            for j in range(self.num_envs):
                current_position[j] = self.rigid_body_states[sum(self.object_count_list[:j]) + 9, 0:3].to("cuda")
                current_orientation[j] = self.rigid_body_states[sum(self.object_count_list[:j]) + 9, 3:7].to("cuda")

            difference = torch.sqrt(torch.sum((current_position-goal_position)**2))

            i += 1

        # print(f"reach ========= {current_position}")
        


            
            # print(f"after ========== {goal_position}")



        # self.gym.simulate(self.sim)
        # self.gym.step_graphics(self.sim)
        # self.gym.draw_viewer(self.viewer, self.sim, True)
        
        # for i in range(10000):
        #     self.pre_physics_step_brank()
        # print('OK')
        # time.sleep(5)
        
        # self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        hand_pos = torch.zeros([self.num_envs, 3])
        ee_pos = torch.zeros([self.num_envs, 3])
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        for j in range(self.num_envs):
            hand_pos[j] = self.rigid_body_states[sum(self.object_count_list[:j]) + self.handle_number_list[j], 0:3].to("cuda")
            ee_pos[j] = self.rigid_body_states[sum(self.object_count_list[:j]) + 9, 0:3].to("cuda")

        # print('OK2')

        # for i in range(10000):
        #     self.pre_physics_step_brank()

        # time.sleep(5)

        step = True

        if target_angle > 170:
            step = False

        if hand_pos[0, 0]  < ee_pos[0, 0] and hand_pos[0, 1]  > ee_pos[0, 1]:
            step = False

        if i > 500:
            step = False

        if torch.isnan(current_position).any().item():
            step = False

        # if angle > target_angle:
        #     step = False
        return step, target_angle, base
        

    def reaching(self):
        self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        ###handle_position の設定
        hand_pos = torch.zeros([self.num_envs, 3])
        ee_pos = torch.zeros([self.num_envs, 3])

        for j in range(self.num_envs):
            hand_pos[j] = self.rigid_body_states[sum(self.object_count_list[:j]) + self.handle_number_list[j], 0:3].to("cuda")
            ee_pos[j] = self.rigid_body_states[sum(self.object_count_list[:j]) + 9, 0:3].to("cuda")

        # print(hand_pos)
        # print(ee_pos)
        # print(self.handle_trajectory)

        

        hand_pos[:, 0] -= 0.01
        hand_pos[:, 2] += 0.012

        self.hand_z = hand_pos[:, 2]

        step_number = 10
        dd = (hand_pos - ee_pos)/10
        # print(ee_pos)
        # print(hand_pos)

        for i in range(step_number):
            goal_pose = ee_pos + (i + 1) * dd
            self.step_ur(ee_pos, goal_pose, i)

        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        for j in range(self.num_envs):
            hand_pos[j] = self.rigid_body_states[sum(self.object_count_list[:j]) + self.handle_number_list[j], 0:3].to("cuda")
            ee_pos[j] = self.rigid_body_states[sum(self.object_count_list[:j]) + 9, 0:3].to("cuda")

        # print(hand_pos)
        # print(ee_pos)
        # for i in range(10000):
        #     self.pre_physics_step_brank()

        # self.gym.write_viewer_image_to_file(self.viewer, f"/home/engawa/py_ws/visual_servo/src/network/images_sice/result/0.png")

    def step_ur(self, current_position, goal_position, i):
         #### 7でよいか確認
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # current_position = self.rigid_body_states[sum(self.object_count_list[:j]) + , 0:3].to("cuda")

        difference = torch.sqrt(torch.sum((current_position-goal_position)**2))
        goal_position =  goal_position.to("cuda")
        while difference > 0.001:
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            ja = self.jacobian[:self.test_num_envs, 8, :, :].to("cuda")
            # print(ja.shape)
            current_position = torch.zeros([self.num_envs, 3]).to("cuda")
            current_orientation = torch.zeros([self.num_envs, 4]).to("cuda")

            # print(goal_position)
            for j in range(self.num_envs):
                current_position[j] = self.rigid_body_states[sum(self.object_count_list[:j]) + 9, 0:3].to("cuda")
                current_orientation[j] = self.rigid_body_states[sum(self.object_count_list[:j]) + 9, 3:7].to("cuda")

            d_dof = self.ik(ja, current_position, current_orientation, goal_position).to("cpu")
            self.ur_dof_targets[:self.test_num_envs, :self.num_ur_dofs] += d_dof[:, 3:]
            
            self.root_tensor[:, 0, :3] += d_dof[:, :3]
        
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_tensor))
            # print(self.ik(ja, current_position, current_orientation, goal_position).to("cpu"))
            self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.ur_dof_targets))
        
            self.gym.simulate(self.sim)
            difference = torch.sqrt(torch.sum((current_position-goal_position)**2))



    def ik(self, jacobian_end_effector: torch.Tensor,
        current_position: torch.Tensor,
        current_orientation: torch.Tensor,
        goal_position: torch.Tensor,
        goal_orientation: Optional[torch.Tensor] = None,
        damping_factor: float = 0.05,
        squeeze_output: bool = True) -> torch.Tensor:

        if goal_orientation is None:
            goal_orientation = torch.zeros([self.num_envs, 4]).to("cuda")
            goal_orientation[:] = torch.tensor([0, 0, 0, 1])

        jacco = torch.zeros([self.num_envs, 6, 3]).to("cuda")
        jacco[:, 0, 0] = 0.02
        jacco[:, 1, 1] = 0.02
        jacco[:, 2, 2] = 0.025

        jacobian_end_effector = torch.cat([jacco, jacobian_end_effector], dim = 2)
        current_position = current_position.to("cuda")

        # compute error
        q = torch_utils.quat_mul(goal_orientation, torch_utils.quat_conjugate(current_orientation)).to("cuda")
        # print(q)
        error = torch.cat([goal_position - current_position,  # position error
                        q[:, 0:3] * torch.sign(q[:, 3]).unsqueeze(-1)],  # orientation error
                        dim=-1).unsqueeze(-1)

        # solve damped least squares (dO = J.T * V)
        transpose = torch.transpose(jacobian_end_effector, 1, 2)
        lmbda = torch.eye(6, device=jacobian_end_effector.device) * (damping_factor ** 2)
        if squeeze_output:
            return (transpose @ torch.inverse(jacobian_end_effector @ transpose + lmbda) @ error).squeeze(dim=2)
        else:
            return transpose @ torch.inverse(jacobian_end_effector @ transpose + lmbda) @ error

    def get_image(self, i):
        self.gym.render_all_camera_sensors(self.sim)
        image_data = torch.zeros([self.num_envs, 1, 224, 224])
        for l in range(self.num_envs):
            # print(l)
            # attention = torch.zeros([480, 640])
            color_image_left = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[l], 0, gymapi.IMAGE_DEPTH)).to(torch.float32)
            color_image = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[l], 0, gymapi.IMAGE_COLOR)).to(torch.float32)

            color_image = color_image.view(color_image .shape[0],-1,4)[...,:3]
            color_image = color_image.permute(2, 0, 1)
            color_image = self.transform(color_image)
            color_image = color_image.permute(1, 2, 0)
            image = color_image.to('cpu').detach().numpy().copy()
            color_image /= 255
            color_image = color_image.permute(2, 0, 1)
            

            # depth_unit = (depth).astype(np.uint8)
            image = Image.fromarray((image).astype(np.uint8))

            # color.save(color_path)

            color_image_left *= -1

            input_depth = color_image_left.clone()
            input_depth = input_depth.unsqueeze(0)
            input_depth /= torch.max(input_depth)/255
            input_depth = self.transform(input_depth)
            # print(depth_image)
            input_depth = input_depth.squeeze().to('cpu').detach().numpy().copy()
            input_depth = Image.fromarray((input_depth).astype(np.uint8))

            color_path = f"/home/engawa/py_ws/visual_servo/src/network/sim2real/image/video/{self.repeat_number}/collor_{str(i).zfill(2)}.png"
            input_path = f"/home/engawa/py_ws/visual_servo/src/network/sim2real/image/video/{self.repeat_number}/get_depth_{str(i).zfill(2)}.png"
            # test_depth = Image.fromarray((test_depth).astype(np.uint8))
            # image.save(color_path)
            # input_depth.save(input_path)
            
            color_image_left = self.transform(color_image_left.unsqueeze(0))
            color_image_left /= torch.max(color_image_left)
            # print(color_image_left)

            image_data[l] = color_image_left

            # self.current_image[l, position, step] = self.transform(color_image_left.unsqueeze(0)).clone()

            # color_path = f"/home/engawa/py_ws/visual_servo/src/network/sim2real/image/ur/{l}_{position}_{step}.png"

            # image.save(color_path)
            # depth.save(depth_path)

        return image_data
    
    def to_torch(self, x, dtype=torch.float, device='cuda:0', requires_grad=False):
        return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)
    
    def pre_physics_step_brank(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        

if __name__ == "__main__": 
    # arg = sys.argv[1]
    rend = rendar()

    rend.create_sim()
    rend.create_ground_plane()
    rend.create_envs()

    for i in range(200):
            rend.pre_physics_step_brank()

    # for position in range(5):
    rend.set_closet_position(0)
    rend.set_ur_position(0)
    # rend.generate_goal_image()
    rend.reaching()

    # for i in range(10000):
    #     rend.pre_physics_step_brank()

    # for step in range(40):
    #     rend.generate_current_image()
    #     rend.generate_action()
    #     rend.opening()

    i = 0
    step = True
    target_angle = 0
    base = 0

    while step:
        rend.generate_current_image(i)
        rend.generate_action()
        step, target_angle, base = rend.opening(target_angle, i, base)
        i += 1

    tensor = torch.tensor([target_angle])

    current = time.time()

    # torch.save(tensor, f"result_si_2/{current}.pt")
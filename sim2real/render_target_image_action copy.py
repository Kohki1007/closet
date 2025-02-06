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
# import isaacgym.torch_utils as torch_utils
# from utils_another.build_yolo import YOLO
# from plane_estimation import Depth
from make_urdf import edit_urdf_real, edit_urdf_structure, edit_urdf_handle_position

import torch
from torchvision import transforms

class rendar():
    def __init__(self, num_envs = 80):
        # self.closet_id = 2
        self.num_envs = num_envs
        self.test_num_envs = self.num_envs
        self.device = 'cuda'
        self.num_dofs = 8
        self.physics_engine = gymapi.SIM_PHYSX
        self.ur_default_dof_pos = self.to_torch([-0.2, 0, 0.8, 0, 0, 0], device=self.device)
        self.closet_default_dof_pos = self.to_torch([1.57, -3.14], device=self.device)
        self.asset_path_list = []
        self.closet_asset_list = [] 
        # self.max = torch.zeros([50, 15, 40])
        # self.closet_position = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/closet_position.pt")
        # self.closet_rot = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/closet_rot.pt")
        # self.wall_position = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/wall_position.pt")

        # self.closet_position = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/closet_posistion_structure.pt")
        self.closet_position = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/real_近い/closet_posistion.pt")
        self.closet_rot = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/real_近い/closet_rot.pt")
        self.wall_position = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/real_近い/wall_position.pt")
        # self.closet_rot = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/closet_rot_structure.pt")
        # self.closet_rot = torch.cat([self.closet_rot_st.clone(), self.closet_rot_ob.clone()], dim = 0)
        # self.closet_position = torch.cat([self.closet_position_st.clone(), self.closet_position_ob.clone()], dim = 0)
        # self.wall_position = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/wall_position_object.pt")

        # self.closet_position_st = []
        # self.closet_position_ob = []
        # self.closet_rot_ob = []
        # self.closet_rot_st = []

        self.transform = transforms.Resize((224, 224))

        # print("good")
        # self.number = 0
        self.current_image = torch.zeros([self.num_envs, 5, 40, 224, 224])
        self.target_image = torch.zeros([self.num_envs, 5, 40, 224, 224])
        # self.current_image = torch.zeros([20, 5, 40, 256, 256])
        # self.action = torch.zeros([self.num_envs, 5, 181, 6])

        self.repeat = 1
        self.number = 100 + self.repeat * self.num_envs 
        self.number = 80

        

        # self.closet_position = torch.load("/home/engawa/py_ws/visual_servo/src/network/predict_opening_door/dataset/delete_UR/rgbd_rgbd/base/closet_position.pth")



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
        
    

    ##closet list : include closet information
    ###[handle width, handle height, handle, closet_x, closet_y]
    def create_envs(self, i = 1, spacing = 2, num_per_row = 5):
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        self.cam_pos = gymapi.Vec3(0.9, 2.0, 1.0)
        self.cam_target = gymapi.Vec3(-0.5, 0, 0.5)
        # self.cam_pos = gymapi.Vec3(0.5, 0, 1.5)
        # self.cam_target = gymapi.Vec3(-0.8, 0, 1.0)
        # self.cam_pos = gymapi.Vec3(0.5, 0.6, 1.0)
        # self.cam_target = gymapi.Vec3(-0.2, 0, 1.0)
        self.cam_pos = gymapi.Vec3(0.5, -0.6, 1.8)
        self.cam_target = gymapi.Vec3(-0.8, 0, 1.0)

        self.gym.viewer_camera_look_at(self.viewer, None, self.cam_pos, self.cam_target)
        # # if i == 1:
        #     self.gym.viewer_camera_look_at(self.viewer, None, self.cam_pos, self.cam_target)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets")
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "/home/engawa/py_ws/visual_servo/src/network/predict_opening_door/dataset/predict_opening_wall/assets"
        # print(asset_root)
        # asset_root = "../../assets"
        ur_asset_file = "urdf/ur3/robots/ur3.urdf"
        # ur_asset_file = "urdf/ur3/robots/ur3_agv.urdf"
        ur_asset_file = "urdf/ur3/robots/hook_test.urdf"
        # closet_asset_file = "urdf/sektion_closet_model/urdf/sektion_closet_2.urdf"
        # closet_asset_file = "urdf/closet_for_sdf/urdf/closet_mimic.urdf"
        closet_asset_file = "urdf/closet_for_sdf/urdf_varius_size/10_900_1059.urdf"
        scale_list = [0.9, 1.0, 1.2, 1.4, 1.6]

        # handle_prop_list = torch.zeros([self.num_envs, 3])

        handle_prop_list = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/real_近い/closet_size.pt")
        # handle_prop_list = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/closet_size_structure.pt")

        # handle_prop_list = torch.cat([handle_prop_list_st, handle_prop_list_ob], dim = 0)


        # for i in range(handle_prop_list.shape[0]):
        #     file_path = f"urdf/closet_for_sdf/urdf_varius_size/{handle_prop_list[i, 0].to(torch.int64)}_{handle_prop_list[i, 1].to(torch.int64)}_{handle_prop_list[i, 2].to(torch.int64)}.urdf"
        #     print(file_path)
        #     self.asset_path_list.append(file_path)


        for a in range(self.num_envs):
            # number = a + self.number
            number = a + self.number
            handle_position = handle_prop_list[number, 0]
            scale = handle_prop_list[number, 1]

            edit_urdf_handle_position(scale, a, handle_position)

            # edit_urdf_object(scale_id, a, handle_position)
            # /home/engawa/py_ws/visual_servo/src/network/predict_opening_door/dataset/predict_opening_wall/assets/urdf/closet_for_real/urdf_real_object/{a}.urdf

            file_path = f'urdf/closet_for_real/urdf_env/{a}.urdf'
            self.asset_path_list.append(file_path)

        # torch.save(handle_prop_list, "/home/engawa/py_ws/visual_servo/src/network/dataset/closet_props/urdf_data.pth")

        # load ur asset
        asset_options = gymapi.AssetOptions()

        asset_box = self.gym.create_box(self.sim, 0.8, 6.0, 8, asset_options)
        # asset_box = self.gym.create_box(self.sim, 0.8, 4.0, 0.1, asset_options)
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
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.use_mesh_materials = True

        closet_asset = []
        for i in range(len(self.asset_path_list)):
            closet_asset.append(self.gym.load_asset(self.sim, asset_root, self.asset_path_list[i], asset_options))
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

        self.camera_props1 = gymapi.CameraProperties()
        # self.camera_props.horizontal_fov = 5.0
        self.camera_props1.width = 1280
        self.camera_props1.height = 720
        self.camera_props1.horizontal_fov = 1.9531
        self.camera_props1.horizontal_fov = 80.796
        self.camera_props1.enable_tensors = True

        self.camera_pos_right = gymapi.Vec3(0.2,0.2,0.7)
        self.camera_pos_left  = gymapi.Vec3(0.3,-0.1,1)

        self.camera_rot_right = gymapi.Vec3(-0.4,0.2, 1.)
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
            ur_pose.p.x = -5
            ur_pose.p.y = 0
            # ur_pose.p.z = 0.4
            ur_pose.p.z = 10
            self.ur_actor = self.gym.create_actor(self.env_ptr, ur_asset, ur_start_pose, "ur", i, 0, 0)
            self.gym.set_actor_dof_properties(self.env_ptr, self.ur_actor, ur_dof_props)

            closet_pose = closet_start_pose
            # closet_position = torch.tensor([random.uniform(-0.7, -0.5), random.uniform(-0.15, 0.15)]).to("cuda")
            closet_pose.p.x = self.closet_position[asset_id + self.number, 0, 0]
            closet_pose.p.y = self.closet_position[asset_id + self.number, 0, 1]
            closet_pose.p.z = 0

            # self.closet_list.append(closet_position)
            self.closet_actor = self.gym.create_actor(self.env_ptr, closet_asset[asset_id], closet_pose, "closet", i, 0, 0)
            self.gym.set_actor_dof_properties(self.env_ptr, self.closet_actor, closet_dof_props)

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(self.wall_position[i + self.number], 0, 1.25)
            # pose.p = gymapi.Vec3(-1.4, 0, 1.25)
            pose.r = gymapi.Quat(0, 0, 0, 1)
            box_handle = self.gym.create_actor(self.env_ptr, asset_box, pose, "actor1", i, 0)
            shape_props = self.gym.get_actor_rigid_shape_properties(self.env_ptr, box_handle)
            shape_props[0].restitution = 1
            shape_props[0].compliance = 0.5

            self.gym.set_actor_rigid_shape_properties(self.env_ptr, box_handle, shape_props)
            self.gym.set_rigid_body_color(self.env_ptr, box_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1., 1., 1.))
            
            self.envs.append(self.env_ptr)
            # self.urs.append(self.ur_actor)
            # self.closets.append(self.closet_actor)
            self.camera_handle.append(self.gym.create_camera_sensor(self.envs[i], self.camera_props1))
            # self.camera_handle.append(self.gym.create_camera_sensor(self.envs[i], self.camera_props2))
            
            # self.gym.set_camera_location(self.camera_handle[0], self.env_ptr, self.camera_pos_right, self.camera_rot_right)
            self.gym.set_camera_location(self.camera_handle[i], self.env_ptr, self.camera_pos_left, self.camera_rot_left)
        

        # self.hand_handle = self.gym.find_actor_rigid_body_handle(self.env_ptr, self.ur_actor, "hook")
        self.closet_door = self.gym.find_actor_rigid_body_handle(self.env_ptr, self.closet_actor, "closet_door")
        self.closet_handle = self.gym.find_actor_rigid_body_handle(self.env_ptr, self.closet_actor, "handle1")

        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        # self.rigid_body_states = 

        # print(self.ur_dof_targets.shape)

        self.ur_dof_targets[:self.test_num_envs, :self.num_ur_dofs] = self.ur_default_dof_pos.to("cpu")
        # self.ur_dof_targets[:self.test_num_envs, self.num_ur_dofs:] = self.closet_default_dof_pos.to("cpu")

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
        self.root_tensor = gymtorch.wrap_tensor(_root_tensor).view(self.num_envs, -1, 13)

        # self.box_root_state = self.root_tensor.view(self.num_envs, -1, 13)[:, 2]

        self._jacobian = self.gym.acquire_jacobian_tensor(self.sim, "ur")
        self.jacobian = gymtorch.wrap_tensor(self._jacobian)
        self.gym.refresh_dof_state_tensor(self.sim)

    def set_closet_position(self, position):
        print("OK")
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        # for j in range(self.num_envs):

        #     id = j * 3 + 1
        #     asset_id = j%40

        #     # self.root_tensor[id, 0] = -1.25
        #     # self.root_tensor[id, 1] = 0
        #     self.root_tensor[j,1, :2] = closet_position[asset_id]
        #     self.root_tensor[j, 7:] = 0

        self.root_tensor[:, 1, :2] = self.closet_position[self.number:(self.number + self.num_envs), position]
        # self.root_tensor[:, 2, 0] = self.wall_position[:, i] - 0.4
 
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_tensor))

        self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        # self.get_image(position)
    
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
        
        # self.hand_pos = self.rigid_body_states[:self.test_num_envs, self.closet_handle][:, 0:3].to("cuda")
        # self.hand_rot = self.rigid_body_states[:self.test_num_envs, self.closet_handle][:, 3:7]

        self.get_image(step, position)

    def pre_physics_action(self, step, position):
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
        # target_angle = self.closet_rot[:, position, step] 
        # target_angle += 10

        right_rad = step * 1.57/180
        left_rad = right_rad * -2
        angle = torch.tensor([right_rad, left_rad])


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

        # print(self.action[:, position, step, :3].shape)
        # print(self.rigid_body_states.shape)
        
        self.action[:, position, step, :3] = self.rigid_body_states[:self.num_envs, self.closet_handle][:, :3].clone()
        # print()

        # for i in range(self.rigid_body_states.shape[0]):
        #     r, p, y = gymapi.Quat(self.rigid_body_states[i, self.closet_handle][3], self.rigid_body_states[i, self.closet_handle][4], self.rigid_body_states[i, self.closet_handle][5], self.rigid_body_states[i, self.closet_handle][6]).to_euler_zyx()
        #     self.handle_trajectory[:, k, 3:] = torch.tensor([r, p, y])
        #     # self.handle_trajectory[:, k, 5] += 3.14

        self.action[:, position, step, 5] = -1.57/90 * (step/2)
        # print(f"{step} ===================== {self.rigid_body_states[:, self.closet_handle][10, :3]}")
        # print(self.rigid_body_states[:self.num_envs, self.closet_handle][:, :3].clone())

        # if step == 0 or step == 180:
        #     if position == 0:
        #         self.get_image_trick(step, position)
        
        if step  == 180 and position == 4:
            print(torch.any(torch.isnan(self.action)))
            torch.save(self.action, f"/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/real__近い/trajectory_{self.repeat}.pth")

            # action = torch.zeros(self.num_envs, 5, 40, 6)

            # for env in range(self.closet_rot.shape[0]):
            #     for position in range(self.closet_rot.shape[0]):
            #         for step in range(self.closet_rot.shape[1]):
            #             start_point = self.closet_rot[env, position, step]

            #             action[env, position, step, :6] = self.action[env, position, int(start_point) + 4] - self.action[env, position, int(start_point)]
            #             # output[step, 6:] = handle_trajectory[int(start_point)]
            #             action[env, position, step, 2] = 0

            #             action[env, position, step, :3] /= torch.sqrt(torch.sum(action[env, position, step, :3]**2))
            #             action[env, position, step, 3:] /= torch.sqrt(torch.sum(action[env, position, step, 3:]**2))

            # torch.save(action, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/action/handle_trajectory.pth")

            



    
    def pre_physics_step_brank(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)


    def check_file_exists(self, file_path):
        if os.path.exists(file_path):
            # print(f"ファイルが存在します: {file_path}")
            return True
        else:
            # print(f"ファイルが存在しません: {file_path}")
            return False
        
    def destroy(self):
        self.gym.destroy_sim(self.sim)

    def get_image_trick(self, step, position):
        self.gym.render_all_camera_sensors(self.sim)
        # test = self.num_envs / 2
        # directly_list = ["door_with_hook", "door_only"]
        for l in range(self.num_envs):
            color_image = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[l], 0, gymapi.IMAGE_COLOR)).to(torch.float32)

            color_image = color_image.view(color_image .shape[0],-1,4)[...,:3]
            color_image = color_image.permute(2, 0, 1)
            color_image = self.transform(color_image)
            color_image = self.transform(color_image).permute(1, 2, 0)
            image = color_image.to('cpu').detach().numpy().copy()
            
            image = Image.fromarray((image).astype(np.uint8))

            color_path = f"/home/engawa/py_ws/visual_servo/src/network/sim2real/image/check_trick/{l}_{step}.png"

            # color_path = f"/home/engawa/py_ws/visual_servo/src/network/sim2real/image_test/imagesize_check.png"

            image.save(color_path)


    def get_image(self, step, position):
        self.gym.render_all_camera_sensors(self.sim)
        # test = self.num_envs / 2
        # directly_list = ["door_with_hook", "door_only"]
        for l in range(self.num_envs):
            # attention = torch.zeros([480, 640])
            color_image_left = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[l], 0, gymapi.IMAGE_DEPTH)).to(torch.float32)
            color_image = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[l], 0, gymapi.IMAGE_COLOR)).to(torch.float32)

            color_image = color_image.view(color_image .shape[0],-1,4)[...,:3]
            color_image = color_image.permute(2, 0, 1)
            # color_image = self.transform(color_image)
            color_image = self.transform(color_image).permute(1, 2, 0)
            # print(color_image.shape)
            # color_image = color_image.to('cpu').detach().numpy().copy()
            # attention = self.yolo.make_attension(color_image)
            # color = Image.fromarray((color_image).astype(np.uint8))
            # for i in range(attention.shape[0]):
            #     if i >= estimate_handle[0] and i <= estimate_handle[1]:
            #         for j in range(attention.shape[1]):
            #             if j >= estimate_handle[2] and j <= estimate_handle[3]:
            #                 attention[i, j] = 1
            image = color_image.to('cpu').detach().numpy().copy()
            

            # depth_unit = (depth).astype(np.uint8)
            image = Image.fromarray((image).astype(np.uint8))

            color_image_left *= -1

            # depth_image = torch.where(color_image_left > 3.0, 0, color_image_left)
            # depth_image = torch.where(depth_image < 0.07, 0, depth_image)

            # max_distance = torch.max(color_image_left)
            depth_image = color_image_left
            depth_image = depth_image.unsqueeze(0)

            

            # if step == 40:
            #     # self.depth_dataset[l, :, :3] = color_image.clone()
            #     self.depth_closet_target[l, :, 0] = self.transform(depth_image).clone()
            # else:
            # print(self.transform(depth_image))
            check = torch.isnan(depth_image).any()
            # if check:
            #     print(position)
            if step == 40:
                # self.depth_dataset[l, :, :3] = color_image.clone()
                self.target_image[l, position, :] = self.transform(depth_image).clone()
            else:
                self.current_image[l, position, step] = self.transform(depth_image).clone().squeeze()

            color_path = f"/home/engawa/py_ws/visual_servo/src/network/sim2real/image/val_parameter/{l}_{position}.png"

            # color_path = f"/home/engawa/py_ws/visual_servo/src/network/sim2real/image_test/imagesize_check.png"

            # image.save(color_path)
            # depth.save(depth_path)

        if position == 4 and step == 40:
            # print("yes")
            SAMPLE_DIR = "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/real__近い"
            # /home/engawa/py_ws/visual_servo/src/network/sim2real/dataset_base


            test_input_name = f"/target_ob_0_max.pt"
            test_input_path = SAMPLE_DIR + test_input_name

            
            # torch.save(self.current_image, test_input_path)


            for i in range(self.current_image.shape[0]):
                for j in range(self.current_image.shape[1]):
                    for k in range(self.current_image.shape[2]):    
                        # print(self.current_image[i, j, k])
                        self.current_image[i, j, k] /= torch.max(self.current_image[i, j, k])
                        self.target_image[i, j, k] /= torch.max(self.target_image[i, j, k])
                        
                        nan_expected = torch.isnan(self.current_image[i, j, k]).any()
                        # if nan_expected:
                        #     print(f"{i}, {j}, {k}")
                    # self.depth_closet_target[i, j, 0] /= torch.max(self.depth_closet_target[i, j, 0])

            test_input_name = f"/current_ob_{self.number}.pt"
            test_input_path = SAMPLE_DIR + test_input_name
            target_input_name = f"/target_{self.repeat}.pt"
            target_input_path = SAMPLE_DIR + target_input_name

            
            # torch.save(self.current_image, test_input_path)
            torch.save(self.target_image, target_input_path)
            print(self.repeat)

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

    rend.create_sim()
    rend.create_ground_plane()
    rend.create_envs()

    # for _ in range(10000000000000000000000):
    #     # rend.pre_physics_step(60)
    #     rend.pre_physics_step_brank()

    for position in range(5):
        rend.set_closet_position(position)
        for step in range(41):
            rend.pre_physics_step(step, position)
        # for step in range(181):
        #     rend.pre_physics_action(step, position)




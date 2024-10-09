from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import math
import numpy as np
import torch


from TeleVision import OpenTeleVision
from Preprocessor import VuerPreprocessor
from constants_vuer import tip_indices
from dex_retargeting.retargeting_config import RetargetingConfig
from pytransform3d import rotations

from pathlib import Path
import argparse
import time
import yaml
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore

from ik import ik_arm
import pytorch3d.transforms as pt3d
from transforms3d import euler
from motion_utils import mat_update, fast_mat_inv




def matrix3d_to_euler_angles_zyx(m3dr):
    '''
    from shilong, not use
    '''
    beta_y = np.arctan2(m3dr[0, 2], np.sqrt(m3dr[0, 0] * m3dr[0, 0] + m3dr[0, 1] * m3dr[0, 1]))
    alpha_z = np.arctan2(-m3dr[0, 1] / np.cos(beta_y), m3dr[0, 0] / np.cos(beta_y))
    gamma_x = np.arctan2(-m3dr[1, 2] / np.cos(beta_y), m3dr[2, 2] / np.cos(beta_y))

    if np.abs(beta_y - np.pi / 2) < 10e-4:
        gamma_x = 0
        alpha_z = np.arctan2(m3dr[1, 0], m3dr[1, 1])

    if np.abs(beta_y + np.pi / 2) < 10e-4:
        gamma_x = 0
        alpha_z = np.arctan2(m3dr[1, 0], m3dr[1, 1])

    gamma_x = (gamma_x + np.pi) % (2 * np.pi) - np.pi
    beta_y = (beta_y + np.pi) % (2 * np.pi) - np.pi
    alpha_z = (alpha_z + np.pi) % (2 * np.pi) - np.pi

    return np.array([gamma_x, beta_y, alpha_z])

class VuerTeleop:
    def __init__(self, config_file_path):
        self.resolution = (720, 1280)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
        image_queue = Queue()
        toggle_streaming = Event()
        self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming)
        self.processor = VuerPreprocessor()

        RetargetingConfig.set_default_urdf_dir('../assets')
        with Path(config_file_path).open('r') as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
        right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()

        self.cur_joint_left=[0,0,0,0,0,0,0]
        self.cur_joint_right=[0,0,0,0,0,0,0]

    def step(self):
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process(self.tv)

        head_rmat = head_mat[:3, :3]

        left_pose = np.concatenate([left_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                    rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        right_pose = np.concatenate([right_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                     rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        # left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[0,1,2,9,10,3,4,5,6,7,8]]
        right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[0,1,2,9,10,3,4,5,6,7,8]]

        return head_rmat, left_pose, right_pose, left_qpos, right_qpos

    # def step_hand_arm(self):
    #     head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process(self.tv)
    #
    #     head_rmat = head_mat[:3, :3]
    #
    #     # left_pose = np.concatenate([left_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
    #     #                             rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
    #     # right_pose = np.concatenate([right_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
    #     #                              rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
    #
    #     left_pose = np.concatenate([left_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
    #                                 rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
    #     right_pose = np.concatenate([right_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
    #                                  rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
    #
    #     left_hand_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[0, 1, 2, 9, 10, 3, 4, 5, 6, 7, 8]]
    #     right_hand_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[0, 1, 2, 9, 10, 3, 4, 5, 6, 7, 8]]
    #
    #     left_arm_qpos= ik_arm(0.0, 1.5708, 0.0, 0.0 ,100, 500.0, 1.0, self.cur_joint_left, 0.0)[:7]
    #     right_arm_qpos=ik_arm(0.0, 1.5708, 0.0, 0.0 ,-100, 500.0, -1.0, self.cur_joint_right, 1.0)[:7]
    #
    #     # print("left arm q:", np.array([left_arm_qpos])*180/math.pi)
    #     # print("right arm q:", np.array([right_arm_qpos])*180/math.pi)
    #
    #     #hardcode
    #     left_arm_qpos[1]=-left_arm_qpos[1]
    #     left_arm_qpos[5]=-left_arm_qpos[5]
    #     left_arm_qpos[6] = -left_arm_qpos[6]
    #     right_arm_qpos[5]=-right_arm_qpos[5]
    #     right_arm_qpos[6] = -right_arm_qpos[6]
    #
    #     #head
    #     head_yaw=-30.6*math.pi/180
    #     head_pitch=-10.2*math.pi/180
    #
    #     return head_rmat, left_pose, right_pose, left_hand_qpos, right_hand_qpos,left_arm_qpos,right_arm_qpos,head_yaw,head_pitch

    def step_hand_arm(self):
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process(self.tv)

        head_rmat = head_mat[:3, :3]

        left_pose = np.concatenate([left_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                    rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        right_pose = np.concatenate([right_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                     rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]])

        # left_pose[:3,3] = left_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6])
        # right_pose[:3,3] = right_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6])

        left_hand_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[0, 1, 2, 9, 10, 3, 4, 5, 6, 7, 8]]
        right_hand_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[0, 1, 2, 9, 10, 3, 4, 5, 6, 7, 8]]

        left_arm_qpos= ik_arm(0.0, 1.5708, 0.0, 0.0 ,100, 500.0, 1.0, self.cur_joint_left, 0.0)[:7]
        right_arm_qpos=ik_arm(0.0, 1.5708, 0.0, 0.0 ,-100, 500.0, -1.0, self.cur_joint_right, 1.0)[:7]

        # print("left arm q:", np.array([left_arm_qpos])*180/math.pi)
        # print("right arm q:", np.array([right_arm_qpos])*180/math.pi)

        #hardcode
        left_arm_qpos[1]=-left_arm_qpos[1]
        left_arm_qpos[5]=-left_arm_qpos[5]
        left_arm_qpos[6] = -left_arm_qpos[6]
        right_arm_qpos[5]=-right_arm_qpos[5]
        right_arm_qpos[6] = -right_arm_qpos[6]

        #head
        head_yaw=-30.6*math.pi/180
        head_pitch=-10.2*math.pi/180

        return head_rmat, left_pose, right_pose, left_hand_qpos, right_hand_qpos,left_arm_qpos,right_arm_qpos,head_yaw,head_pitch


class Sim:
    def __init__(self,
                 print_freq=False):
        self.print_freq = print_freq

        # initialize gym
        self.gym = gymapi.acquire_gym()

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1 / 60
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.max_gpu_contact_pairs = 8388608
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = False

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        plane_params = gymapi.PlaneParams()
        plane_params.distance = 0.0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        # load table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, 0.8, 0.8, 0.1, table_asset_options)

        # load cube asset
        cube_asset_options = gymapi.AssetOptions()
        cube_asset_options.density = 10
        cube_asset = self.gym.create_box(self.sim, 0.05, 0.05, 0.05, cube_asset_options)

        robot_asset_root = "../assets"
        robot_asset_file = 'openloong_hand/urdf/AzureDragon.urdf'

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        self.robot_asset = self.gym.load_asset(self.sim, robot_asset_root, robot_asset_file, asset_options)
        self.dof = self.gym.get_asset_dof_count(self.robot_asset)
        print("the robot total dof is: ", self.dof)

        ##get the joint info
        dof_props = self.gym.get_asset_dof_properties(self.robot_asset)
        num_dofs = len(dof_props["lower"])
        joint_names=[]
        for i in range(num_dofs):
            dof_name = self.gym.get_asset_dof_name(self.robot_asset, i)
            joint_names.append([dof_name,i])


        # set up the env grid
        num_envs = 1
        num_per_row = int(math.sqrt(num_envs))
        env_spacing = 1.25
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        np.random.seed(0)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

        # robot
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.8, 0, 1.1)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.robot_handle = self.gym.create_actor(self.env, self.robot_asset, pose, 'robot', 1, 1)
        self.gym.set_actor_dof_states(self.env, self.robot_handle, np.zeros(self.dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)

        # table
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 1.2)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        table_handle = self.gym.create_actor(self.env, table_asset, pose, 'table', 0)
        color = gymapi.Vec3(0.5, 0.5, 0.5)
        self.gym.set_rigid_body_color(self.env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # cube
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 1.25)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        cube_handle = self.gym.create_actor(self.env, cube_asset, pose, 'cube', 0)
        color = gymapi.Vec3(1, 0.5, 0.5)
        self.gym.set_rigid_body_color(self.env, cube_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        #load robot hand for test
        asset_root = "../assets"
        left_asset_path = "zb_hand/zbll/urdf/zbll.urdf"
        right_asset_path = "zb_hand/zbr/urdf/zbr.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        self.left_asset = self.gym.load_asset(self.sim, asset_root, left_asset_path, asset_options)
        self.right_asset = self.gym.load_asset(self.sim, asset_root, right_asset_path, asset_options)
        # self.dof = self.gym.get_asset_dof_count(self.left_asset)

        # left_hand
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.6, 0, 1.6)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.left_handle = self.gym.create_actor(self.env, self.left_asset, pose, 'left', 1, 1)
        self.gym.set_actor_dof_states(self.env, self.left_handle, np.zeros(self.dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        left_idx = self.gym.get_actor_index(self.env, self.left_handle, gymapi.DOMAIN_SIM)

        # right_hand
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.6, 0, 1.6)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.right_handle = self.gym.create_actor(self.env, self.right_asset, pose, 'right', 1, 1)
        self.gym.set_actor_dof_states(self.env, self.right_handle, np.zeros(self.dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        right_idx = self.gym.get_actor_index(self.env, self.right_handle, gymapi.DOMAIN_SIM)

        self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self.root_state_tensor)
        self.left_root_states = self.root_states[left_idx]
        self.right_root_states = self.root_states[right_idx]

        # create default viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()
        cam_pos = gymapi.Vec3(1, 1, 2)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.cam_lookat_offset = np.array([1, 0, 0])
        self.left_cam_offset = np.array([0, 0.033, 0])
        self.right_cam_offset = np.array([0, -0.033, 0])
        self.cam_pos = np.array([-0.6, 0, 1.6])

        # create left 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 720
        self.left_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.left_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset + self.cam_lookat_offset)))

        # create right 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 720
        self.right_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.right_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset + self.cam_lookat_offset)))

        ##current joint for ik
        self.cur_joint_left=[0,0,0,0,0,0,0]
        self.cur_joint_right=[0,0,0,0,0,0,0]

    def step_hand_arm(self, head_rmat, left_pose, right_pose, left_hand_qpos, right_hand_qpos,left_arm_qpos,right_arm_qpos, head_yaw, head_pitch):

        if self.print_freq:
            start = time.time()
        
        #virtual hands
        self.left_root_states[0:7] = torch.tensor(left_pose, dtype=float)
        self.right_root_states[0:7] = torch.tensor(right_pose, dtype=float)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

        #get shoulder pos
        left_arm_link_name="Link_arm_l_02"
        right_arm_link_name="Link_arm_r_02"
        left_arm_link_index = self.gym.find_actor_rigid_body_index(self.env, self.robot_handle, left_arm_link_name, gymapi.DOMAIN_SIM)
        right_arm_link_index = self.gym.find_actor_rigid_body_index(self.env, self.robot_handle, right_arm_link_name,gymapi.DOMAIN_SIM)

        self.rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_state_torch=gymtorch.wrap_tensor(self.rigid_body_state_tensor)
        left_link_state = self.rigid_body_state_torch[left_arm_link_index]
        right_link_state=self.rigid_body_state_torch[right_arm_link_index]
        # [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w,lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_x, ang_vel_y, ang_vel_z]

        world2left_shoulder_R=torch.tensor([[0,1,0],
                                            [0,0,1],
                                            [1,0,0]])
        world2right_shoulder_R=torch.tensor([[0,-1,0],
                                            [0,0,-1],
                                            [1,0,0]])

        world2left_shoulder = torch.eye(4)
        world2left_shoulder[:3,3]=left_link_state[0:3]
        world2left_shoulder[:3,:3]=world2left_shoulder_R
        world2right_shoulder = torch.eye(4)
        world2right_shoulder[:3,3]=right_link_state[0:3]
        world2right_shoulder[:3,:3]=world2right_shoulder_R

        #transform the target left hand pose and right hand pose to the shoulder frame for ik
        left_pose=torch.tensor(left_pose, dtype=float)
        right_pose=torch.tensor(right_pose, dtype=float)
        left_pose_matrix=torch.eye(4)
        left_pose_matrix[:3,3]=left_pose[0:3]
        left_pose_matrix[:3,:3]=pt3d.quaternion_to_matrix(left_pose[3:7][[3,0,1,2]]) #left_pose(xyzw)
        right_pose_matrix=torch.eye(4)
        right_pose_matrix[:3,3]=right_pose[0:3]
        right_pose_matrix[:3,:3]=pt3d.quaternion_to_matrix(right_pose[3:7][[3,0,1,2]])

        left_hand_in_shoulder = torch.inverse(world2left_shoulder)@left_pose_matrix
        right_hand_in_shoulder = torch.inverse(world2right_shoulder)@right_pose_matrix


        ##align target hand pose to flange frame
        l_flange2hand=torch.eye(4)
        l_flange2hand[:3,:3]=torch.tensor([[1,0,0],
                                           [0,0,1],
                                           [0,-1,0]])
        l_flange2hand[:3,3]=torch.tensor([0,0,-0.05])##translation from hand to flange

        #align flange frame to arm ee frame
        l_flange2ee=torch.eye(4)
        l_flange2ee[:3,:3]=torch.tensor([[0,0,-1],
                                         [0,1,0],
                                         [1,0,0]])

        left_pose_in_shoulder=left_hand_in_shoulder@torch.inverse(l_flange2hand)@l_flange2ee

        r_flange2hand=torch.eye(4)
        r_flange2hand[:3,:3]=torch.tensor([[-1,0,0],
                                           [0,0,-1],
                                           [0,-1,0]])
        r_flange2hand[:3,3]=torch.tensor([0,0,-0.05])##
        r_flange2ee=torch.eye(4)
        r_flange2ee[:3,:3]=torch.tensor([[0,0,-1],
                                         [0,1,0],
                                         [1,0,0]])

        right_pose_in_shoulder=right_hand_in_shoulder@torch.inverse(r_flange2hand)@r_flange2ee

        rz,ry,rx=euler.mat2euler(left_pose_in_shoulder,'szyx')
        left_pose_in_shoulder[:3,3]=left_pose_in_shoulder[:3,3]*1000
        left_arm_qpos= ik_arm(rz, ry, rx, left_pose_in_shoulder[0,3] ,left_pose_in_shoulder[1,3], left_pose_in_shoulder[2,3], 1.0, self.cur_joint_left, 0.0)[:7]

        rz, ry, rx = euler.mat2euler(right_pose_in_shoulder,'szyx')
        right_pose_in_shoulder[:3,3]=right_pose_in_shoulder[:3,3]*1000
        right_arm_qpos=ik_arm(rz, ry, rx, right_pose_in_shoulder[0,3],right_pose_in_shoulder[1,3], right_pose_in_shoulder[2,3], -1.0, self.cur_joint_right, 1.0)[:7]

        #hardcode important!
        left_arm_qpos[1]=-left_arm_qpos[1]
        left_arm_qpos[5]=-left_arm_qpos[5]
        left_arm_qpos[6]= -left_arm_qpos[6] 
        right_arm_qpos[5]=-right_arm_qpos[5]
        right_arm_qpos[6] = -right_arm_qpos[6]

        ##set robot joint
        self.gym.refresh_dof_state_tensor(self.sim)
        joint_states=np.zeros(self.dof, dtype=gymapi.DofState.dtype)

        joint_states['pos'][:7]=left_arm_qpos
        joint_states['pos'][18:25]=right_arm_qpos
        joint_states['pos'][7:18]=left_hand_qpos
        joint_states['pos'][25:36]=right_hand_qpos

        #head move
        joint_states['pos'][36]=head_yaw
        joint_states['pos'][37]=head_pitch

        self.gym.set_actor_dof_states(self.env, self.robot_handle, joint_states, gymapi.STATE_POS)

        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        curr_lookat_offset = self.cam_lookat_offset @ head_rmat.T
        curr_left_offset = self.left_cam_offset @ head_rmat.T
        curr_right_offset = self.right_cam_offset @ head_rmat.T

        self.gym.set_camera_location(self.left_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + curr_left_offset)),
                                     gymapi.Vec3(*(self.cam_pos + curr_left_offset + curr_lookat_offset)))
        self.gym.set_camera_location(self.right_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + curr_right_offset)),
                                     gymapi.Vec3(*(self.cam_pos + curr_right_offset + curr_lookat_offset)))
        left_image = self.gym.get_camera_image(self.sim, self.env, self.left_camera_handle, gymapi.IMAGE_COLOR)
        right_image = self.gym.get_camera_image(self.sim, self.env, self.right_camera_handle, gymapi.IMAGE_COLOR)
        left_image = left_image.reshape(left_image.shape[0], -1, 4)[..., :3]
        right_image = right_image.reshape(right_image.shape[0], -1, 4)[..., :3]

        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

        if self.print_freq:
            end = time.time()
            print('Frequency:', 1 / (end - start))

        return left_image, right_image
    
    def visulize_frame(self,left_pose,right_pose):
        '''
        only for test
        '''

        left_pose=left_pose.cpu().numpy()
        right_pose=right_pose.cpu().numpy()
        #visulize the hand axis
        left_pose_frame = gymapi.Transform(
            p=gymapi.Vec3(*left_pose[:3].copy().tolist()),
            r=gymapi.Quat(*left_pose[3:7].copy().tolist()),
        )
        right_pose_frame = gymapi.Transform(
            p=gymapi.Vec3(*right_pose[:3].copy().tolist()),
            r=gymapi.Quat(*right_pose[3:7].copy().tolist()),
        )
        gymutil.draw_lines(
            gymutil.AxesGeometry(scale=0.2),
            self.gym,
            self.viewer,
            self.env,
            left_pose_frame,
        )
        gymutil.draw_lines(
            gymutil.AxesGeometry(scale=0.2),
            self.gym,
            self.viewer,
            self.env,
            right_pose_frame,
        )
        ##visualize the left and right hand in the arm
        left_arm_link_name="left_hand_base_link"
        right_arm_link_name="right_hand_base_link"
        left_arm_link_index = self.gym.find_actor_rigid_body_index(self.env, self.robot_handle, left_arm_link_name, gymapi.DOMAIN_SIM)
        right_arm_link_index = self.gym.find_actor_rigid_body_index(self.env, self.robot_handle, right_arm_link_name,gymapi.DOMAIN_SIM)

        self.rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_state_torch=gymtorch.wrap_tensor(self.rigid_body_state_tensor)
        left_hand_state = self.rigid_body_state_torch[left_arm_link_index]
        right_hand_state=self.rigid_body_state_torch[right_arm_link_index]
        left_hand_arm_pose=left_hand_state.cpu().numpy()
        right_hand_arm_pose=right_hand_state.cpu().numpy()
        left_hand_arm_frame = gymapi.Transform(
            p=gymapi.Vec3(*left_hand_arm_pose[:3].copy().tolist()),
            r=gymapi.Quat(*left_hand_arm_pose[3:7].copy().tolist()),
        )
        right_hand_arm_frame = gymapi.Transform(
            p=gymapi.Vec3(*right_hand_arm_pose[:3].copy().tolist()),
            r=gymapi.Quat(*right_hand_arm_pose[3:7].copy().tolist()),
        )
        gymutil.draw_lines(
            gymutil.AxesGeometry(scale=0.2),
            self.gym,
            self.viewer,
            self.env,
            left_hand_arm_frame,
        )
        gymutil.draw_lines(
            gymutil.AxesGeometry(scale=0.2),
            self.gym,
            self.viewer,
            self.env,
            right_hand_arm_frame,
        )

        left_arm_link_name="Link_arm_l_02"
        right_arm_link_name="Link_arm_r_02"
        left_arm_link_index = self.gym.find_actor_rigid_body_index(self.env, self.robot_handle, left_arm_link_name, gymapi.DOMAIN_SIM)
        right_arm_link_index = self.gym.find_actor_rigid_body_index(self.env, self.robot_handle, right_arm_link_name,gymapi.DOMAIN_SIM)

        self.rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_state_torch=gymtorch.wrap_tensor(self.rigid_body_state_tensor)
        left_shoulder_state = self.rigid_body_state_torch[left_arm_link_index]
        right_shoulder_state=self.rigid_body_state_torch[right_arm_link_index]
        left_shoulder_pose = left_shoulder_state.cpu().numpy()
        right_shoulder_pose = right_shoulder_state.cpu().numpy()
        left_shoulder_frame = gymapi.Transform(
            p=gymapi.Vec3(*left_shoulder_pose[:3].copy().tolist()),
            r=gymapi.Quat(*left_shoulder_pose[3:7].copy().tolist()),
        )
        right_shoulder_frame = gymapi.Transform(
            p=gymapi.Vec3(*right_shoulder_pose[:3].copy().tolist()),
            r=gymapi.Quat(*right_shoulder_pose[3:7].copy().tolist()),
        )
        gymutil.draw_lines(
            gymutil.AxesGeometry(scale=0.2),
            self.gym,
            self.viewer,
            self.env,
            left_shoulder_frame,
        )
        gymutil.draw_lines(
            gymutil.AxesGeometry(scale=0.2),
            self.gym,
            self.viewer,
            self.env,
            right_shoulder_frame,
        )


    def end(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


if __name__ == '__main__':
    teleoperator = VuerTeleop('zb_hand.yml')
    simulator = Sim()

    try:
        while True:

            head_rmat, left_pose, right_pose, left_hand_qpos, right_hand_qpos,left_arm_qpos,right_arm_qpos,head_yaw,head_pitch = teleoperator.step_hand_arm()

            left_img, right_img = simulator.step_hand_arm(head_rmat, left_pose, right_pose, left_hand_qpos, right_hand_qpos,left_arm_qpos,right_arm_qpos,head_yaw, head_pitch )

            np.copyto(teleoperator.img_array, np.hstack((left_img, right_img)))

            ##update the frame visulazition
            simulator.gym.clear_lines(simulator.viewer)
    except KeyboardInterrupt:
        simulator.end()
        exit(0)

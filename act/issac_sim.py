import numpy as np
from isaacgym import gymapi, gymutil
import torch
# from teleop import teleop_hand_zb
import os
import h5py

# Create Gym Environment
def create_environment():
    # Acquire Isaac Gym interface
    gym = gymapi.acquire_gym()

    # Initialize simulation parameters
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    # Create simulation
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise Exception("Failed to create simulation")

    # Add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # Set asset loading options
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = False
    asset_options.use_mesh_materials = True

    # Load dexterous hand URDF asset
    hand_asset = gym.load_asset(
        sim, "/home/guozhuoran/Openloong_TeleVision-main/assets/zb_hand/zbll/urdf/", "zbll.urdf", asset_options
    )
    if hand_asset is None:
        raise Exception("Failed to load hand asset")

    # Load cube URDF asset (object to manipulate)
    cube_asset_options = gymapi.AssetOptions()
    cube_asset_options.density = 10
    cube_asset = gym.create_box(sim, 0.05, 0.05, 0.05, cube_asset_options)

    # load table asset
    table_asset_options = gymapi.AssetOptions()
    table_asset_options.disable_gravity = True
    table_asset_options.fix_base_link = True
    table_asset = gym.create_box(sim, 0.8, 0.8, 0.1, table_asset_options)

    # Environment setup: Create actors (robots and objects)
    env = gym.create_env(sim, gymapi.Vec3(-1.0, 0.0, 0.0), gymapi.Vec3(1.0, 0.0, 0.0), 1)

    # Set initial pose for hand and block
    hand_pose = gymapi.Transform()
    hand_pose.p = gymapi.Vec3(0, 0, 1)
    hand_pose.r = gymapi.Quat(0, 0, 0, 1)

    block_pose = gymapi.Transform()
    block_pose.p = gymapi.Vec3(0.5, 0.0, 0.5)
    block_pose.r = gymapi.Quat(0, 0, 0, 1)

    # Add actors to the environment
    hand_handle = gym.create_actor(env, hand_asset, hand_pose, "hand", 0, 0)
    block_handle = gym.create_actor(env, cube_asset, block_pose, "block", 0, 0)

    return gym, sim, env, hand_handle, block_handle

def save_episode_data(hdf5_file, episode, observations, actions, rewards):

    path = os.path.dirname(hdf5_file)
    all_data_path = os.path.join(path, "tele")
    os.makedirs(all_data_path, exist_ok=True)

    with h5py.File(all_data_path + f"/tele_{episode}.hdf5", 'w') as hf:
       
        hf.create_dataset('observations', data=observations)
        hf.create_dataset('actions', data=actions)
        hf.create_dataset('rewards', data=rewards)

def get_observation(env, hand_handle, block_handle):
    gym = gymapi.acquire_gym()
    hand_states = gym.get_actor_rigid_body_states(env, hand_handle, gymapi.STATE_ALL)
    block_states = gym.get_actor_rigid_body_states(env, block_handle, gymapi.STATE_ALL)
    hand_pos = hand_states['pose']['p']
    hand_orient = hand_states['pose']['r']
    block_pos = block_states['pose']['p']
    block_orient = block_states['pose']['r']
    observation = np.concatenate([hand_pos, hand_orient, block_pos, block_orient])
    return observation

def calculate_reward(hand_states, block_states):
    hand_pos = hand_states['pose']['p']
    block_pos = block_states['pose']['p']
    distance = np.linalg.norm(np.array(hand_pos) - np.array(block_pos))
    reward = -distance
    return reward

def get_action_from_policy(observation):
    action = np.random.uniform(low=-1, high=1, size=(20,))  # Replace this with real action generation logic
    return action

def main():   

    gym, sim, env, hand_handle, block_handle = create_environment()

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise Exception("Failed to create viewer")

    hdf5_file = h5py.File('recorded_episodes.hdf5', 'w')
    num_episodes = 10
    for episode in range(num_episodes):
        observations = []
        actions = []
        rewards = []

        for t in range(1000):
            gym.simulate(sim)
            gym.fetch_results(sim, True)

            obs = get_observation(env, hand_handle, block_handle)
            action = get_action_from_policy(obs)
            reward = calculate_reward(gym.get_actor_rigid_body_states(env, hand_handle, gymapi.STATE_ALL),
                                      gym.get_actor_rigid_body_states(env, block_handle, gymapi.STATE_ALL))

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)

            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)

            if gym.query_viewer_has_closed(viewer):
                break

        save_episode_data(hdf5_file, episode, np.array(observations), np.array(actions), np.array(rewards))
        print(f"Episode {episode} recorded.")

    hdf5_file.close()
    gym.destroy_viewer(viewer)

if __name__ == "__main__":
    main()

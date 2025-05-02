# from myosuite.utils import gym
from tcdm.envs import suite
import numpy as np
# from myosuite.utils.quat_math import quat2euler
from matplotlib import pyplot as plt

from pyquaternion import Quaternion

env = suite.load('train','play1')

from tcdm.envs.wrappers import GymWrapper

env = GymWrapper(env)

breakpoint()

# table.xml altered to change/remove visuals

# simpleGraniteTable_id = env.unwrapped.sim.model.body("simpleGraniteTable").id

# body_geoms = []
# world_geoms = []

# for geom_id in range(env._base_env.physics.model.ngeom):  # ngeom is the total number of geoms
#     # if env.unwrapped.sim.model.geom_bodyid[geom_id] == simpleGraniteTable_id:
#         # body_geoms.append(geom_id)  # Store the geom index for the body
#     if env._base_env.physics.model.geom_bodyid[geom_id] == env._base_env.physics.model.body("world").id:
#         world_geoms.append(geom_id) 

# breakpoint()

table_geom_ids = []
for geom_id in range(env._base_env.physics.model.ngeom):  # ngeom is the total number of geoms
    # if env.unwrapped.sim.model.geom_bodyid[geom_id] == simpleGraniteTable_id:
        # body_geoms.append(geom_id)  # Store the geom index for the body
    if env._base_env.physics.model.geom_bodyid[geom_id] ==  env._base_env.physics.model.body("table").id:
        table_geom_ids.append(geom_id) 

# for i in range(12):
#     env.unwrapped.sim.model.geom_rgba[body_geoms[i]] = [1, 1, 1, 0]
#     env.unwrapped.sim.model.geom_contype[body_geoms[i]] = 0
#     env.unwrapped.sim.model.geom_conaffinity[body_geoms[i]] = 0

env.reset()

# def _myo_touching_object(env):
        
#         object_body_id = env.unwrapped.sim.model.body('cup').id
#         myo_bodies_ids = [env.unwrapped.sim.model.body(i).id for i in range(env.unwrapped.sim.model.nbody)
#                         if not env.unwrapped.sim.model.body(i).name in ["simpleGraniteTable", "world", "full_body", "thorax"]]

#         for con in env.unwrapped.sim.data.contact:
#             if env.unwrapped.sim.model.geom(con.geom1).bodyid == object_body_id:
#                 if env.unwrapped.sim.model.geom(con.geom2).bodyid in myo_bodies_ids:
#                     return 1.
#             elif env.unwrapped.sim.model.geom(con.geom2).bodyid == object_body_id:
#                 if env.unwrapped.sim.model.geom(con.geom1).bodyid in myo_bodies_ids:
#                     return 1.
#         return 0.

def _object_touching_table(env):

    object_body_id = env._base_env.physics.model.body("train/object").id
    # table_geom_id = env._base_env.physics.model.body("table").id

    # breakpoint()

    for con in env._base_env.physics.data.contact:
        if env._base_env.physics.model.geom(con.geom1).bodyid[0] == object_body_id:
            # if env.unwrapped.sim.model.geom(con.geom2).bodyid == table_body_id:
            if con.geom2 == table_geom_ids[0]:
                return 1.
        elif env._base_env.physics.model.geom(con.geom2).bodyid[0] == object_body_id:
            # if env.unwrapped.sim.model.geom(con.geom1).bodyid == table_body_id:
            if con.geom1 == table_geom_ids[0]:
                return 1.
    return 0.
# def _to_quat(arr): 
#     if isinstance(arr, Quaternion):
#         return arr.unit
#     if len(arr.shape) == 2:
#         return Quaternion(matrix=arr).unit
#     elif len(arr.shape) == 1 and arr.shape[0] == 9:
#         return Quaternion(matrix=arr.reshape((3,3))).unit
#     return Quaternion(array=arr).unit

# def _rotation_distance(q1, q2):
#     delta_quat = _to_quat(q2) * _to_quat(q1).inverse
#     return np.abs(delta_quat.angle)

# fin0 = env.unwrapped.sim.model.site_name2id("THtip")
# fin1 = env.unwrapped.sim.model.site_name2id("IFtip")
# fin2 = env.unwrapped.sim.model.site_name2id("MFtip")
# fin3 = env.unwrapped.sim.model.site_name2id("RFtip")
# fin4 = env.unwrapped.sim.model.site_name2id("LFtip")

grasp_errors_all = []
obj_com_err_all = []
obj_rot_err_all = []

import imageio

# Example: sequence of 100 RGB frames (height=64, width=64)
frames = []

# for t in range(80):

#     _ = env.step(np.random.randn(30))

#     frames.append(env.render())

# env._base_env.task.reference_motion._reference_motion['object_translation']
# target_sid = env._base_env.physics.model.site_name2id("target")

delta_qpos = env._base_env.physics.data.qpos[30:30+3]-env._base_env.task.reference_motion._reference_motion['object_translation'][0,:]

for t in range(80):
    # if t == 0:
    #     # env._base_env.physics.data.qpos[:29]=env.ref.reference['robot_init'].copy()
    #     # breakpoint()
    #     env._base_env.physics.data.qpos[30:30+3]=env.ref.reference['object_init'][:3].copy()
    #     env._base_env.physics.data.qpos[-3:]=quat2euler(env.ref.reference['object_init'][3:]).copy()
    #     # env.sim.data.qpos[3]-=0.06
    #     # env.sim.data.qpos[31]-=0.06
    #     env.sim.forward()
    # # elif t < 13:
    # else:
    # env._base_env.physics.data.qpos[:29]=env.ref.reference['robot'][t,:].copy()
    env._base_env.physics.data.qpos[30:30+3]=env._base_env.task.reference_motion._reference_motion['object_translation'][t,:].copy()+delta_qpos
    env._base_env.physics.data.qpos[-3:]=quat2euler(env._base_env.task.reference_motion._reference_motion['object_orientation'][t,:]).copy()
    # env._base_env.physics.model.site_pos[target_sid] = env.ref.reference['object'][t,:3].copy()
    # env.sim.data.qpos[3]-=0.06
    # env.sim.data.qpos[31]-=0.06
    env._base_env.physics.forward()
    # else:
    #     _ = env.step(env.action_space.sample())

    # obj_pos = env.unwrapped.sim.data.body('wineglass').xpos
    # obj_ori = np.reshape(env.unwrapped.sim.data.body('wineglass').xmat, (3, 3))

    # if t == 0:
    #     init_tip0 = np.linalg.inv(obj_ori) @ (env.unwrapped.sim.data.site_xpos[fin0]-obj_pos)
    #     init_tip1 = np.linalg.inv(obj_ori) @ (env.unwrapped.sim.data.site_xpos[fin1]-obj_pos)
    #     init_tip2 = np.linalg.inv(obj_ori) @ (env.unwrapped.sim.data.site_xpos[fin2]-obj_pos)
    #     init_tip3 = np.linalg.inv(obj_ori) @ (env.unwrapped.sim.data.site_xpos[fin3]-obj_pos)
    #     init_tip4 = np.linalg.inv(obj_ori) @ (env.unwrapped.sim.data.site_xpos[fin4]-obj_pos)

    # tip0 = np.linalg.inv(obj_ori) @ (env.unwrapped.sim.data.site_xpos[fin0]-obj_pos)
    # tip1 = np.linalg.inv(obj_ori) @ (env.unwrapped.sim.data.site_xpos[fin1]-obj_pos)
    # tip2 = np.linalg.inv(obj_ori) @ (env.unwrapped.sim.data.site_xpos[fin2]-obj_pos)
    # tip3 = np.linalg.inv(obj_ori) @ (env.unwrapped.sim.data.site_xpos[fin3]-obj_pos)
    # tip4 = np.linalg.inv(obj_ori) @ (env.unwrapped.sim.data.site_xpos[fin4]-obj_pos)

    # err_tip0 = init_tip0-tip0
    # err_tip1 = init_tip1-tip1
    # err_tip2 = init_tip2-tip2
    # err_tip3 = init_tip3-tip3
    # err_tip4 = init_tip4-tip4

    # grasp_errors = np.array((np.linalg.norm(err_tip0),
    #                             np.linalg.norm(err_tip1),
    #                             np.linalg.norm(err_tip2),
    #                             np.linalg.norm(err_tip3),
    #                             np.linalg.norm(err_tip4)))
    
    # grasp_errors_all.append(grasp_errors.mean())

    # obs_dict = env.unwrapped.get_obs_dict(env.unwrapped.sim)
    # obj_rot_err_all.append(rotation_distance(env.ref.reference['object_init'][3:], obs_dict["targ_obj_rot"]) / np.pi)
    
    # obj_rot_err_all.append(env.rotation_distance(obs_dict["curr_obj_rot"], obs_dict["targ_obj_rot"], False) / np.pi)
    # obj_rot_err_all.append(env.rotation_distance(env.unwrapped.sim.data.xquat[env.object_bid], obs_dict["targ_obj_rot"], False) / np.pi)

    # obs_dict = env.unwrapped.get_obs_dict(env.unwrapped.sim)
    # obs_dict = env.obs_dict
    # obj_com_err_all.append(np.linalg.norm(obs_dict["obj_com_err"]))
    # # env.sim.data.qpos[env.ref.robot_dim : env.ref.robot_dim + 3]
    # print(obs_dict["curr_obj_com"])
    # print(obs_dict["targ_obj_com"])
    # print(np.linalg.norm(obs_dict["obj_com_err"]))
    # print(np.linalg.norm(env.sim.data.qpos[env.ref.robot_dim : env.ref.robot_dim + 3]-obs_dict["targ_obj_com"]))
    # obj_com_err_all.append(np.linalg.norm(env.sim.data.qpos[env.ref.robot_dim : env.ref.robot_dim + 3]-obs_dict["targ_obj_com"]))

    # _ = env.step(env.action_space.sample()*0.)

    # env.mj_render()



    # print(t, _myo_touching_object(env))

    print(t, _object_touching_table(env))

    # breakpoint()

    frames.append(env.render())

    # print([env.sim.data.qvel[-1], env.sim.data.cvel[env.object_bid][-1]])

    # print(t, grasp_errors.mean())

    # print(t, obj_com_err_all[-1])

    # print(t, obs_dict["obj_com_err"], obj_com_err_all[-1], obs_dict["curr_obj_com"][-1], obs_dict["targ_obj_com"][-1])

imageio.mimsave('output_video.mp4', frames, fps=30)
# plt.plot(obj_rot_err_all)
# plt.show()
# breakpoint()
breakpoint()

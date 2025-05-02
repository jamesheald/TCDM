# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from tcdm.rl.models.OBJEX.ppo import PPO as OBJEX_PPO
from tcdm.rl.trainers.util import make_env, make_policy_kwargs, \
                                   InfoCallback, FallbackCheckpoint, \
                                   get_warm_start
from wandb.integration.sb3 import WandbCallback
from tcdm.rl.models.OBJEX.policies import ActorCriticPolicy
# from tcdm.rl.models.policies import ActorCriticPolicy
from tcdm.rl.trainers.eval import make_eval_env, EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback


def objex_ppo_trainer(config, resume_model=None):
    total_timesteps = config.total_timesteps
    eval_freq = int(config.eval_freq // config.n_envs)
    save_freq = int(config.save_freq // config.n_envs)
    restore_freq = int(config.restore_checkpoint_freq // config.n_envs)
    n_steps = int(config.agent.params.n_steps // config.n_envs)
    multi_proc = bool(config.agent.multi_proc)
    env = make_env(multi_proc=multi_proc, **config.env)

    from tcdm import suite
    from omegaconf import OmegaConf
    domain, task = config.env['name'].split('-')
    temp_env = suite.load(domain, task, OmegaConf.to_container(config.env['task_kwargs']), dict(config.env['env_kwargs']), gym_wrap=True)
    object_bodyid = temp_env._base_env.physics.model.name2id(domain + '/object', 'body')
    # breakpoint()
    # import numpy as np
    # np.isclose(envv.reset()['state'][:36],temp_env._base_env.physics.data.qpos).all()
    # np.isclose(envv.reset()['state'][155:(155+36)],temp_env._base_env.physics.data.qvel).all()

    from tcdm.envs import mj_models
    import mujoco
    from mujoco import mjx
    import jax
    from jax import numpy as jp
    
    object_class = mj_models.get_object(domain) # object_model = object_class()
    object_model = object_class() # robot_model = robot_class(limp=False)
    robot_model = mj_models.Adroit(limp=False)

    temp_env = mj_models.TableEnv()
    temp_env.attach(robot_model)
    temp_env.attach(object_model)

    mjcf_model = temp_env.mjcf_model
    xml_string = mjcf_model.to_xml_string()

    assets = dict(mjcf_model.get_assets())
    model = mujoco.MjModel.from_xml_string(xml_string, assets=assets)
    mjx_model = mjx.put_model(model) # NB i had to remove margins in common.xml
    mjx_model = mjx_model.replace(opt=mjx_model.opt.replace(iterations=1))
    mjx_model = mjx_model.replace(opt=mjx_model.opt.replace(ls_iterations=1))
    mjx_data =  mjx.make_data(mjx_model)
    # joint_dict = {}
    # for dof_i in range(mjx_model.nv):
    #     jnt_id = mjx_model.dof_jntid[dof_i]
    #     jnt_name = mjx_model.names[mjx_model.name_jntadr[jnt_id]:].split(b'\x00')[0].decode('utf-8') 
    #     joint_dict[jnt_name] = jnt_id
    temp_env = suite.load(domain, task, OmegaConf.to_container(config.env['task_kwargs']), dict(config.env['env_kwargs']), gym_wrap=True)
    temp_env.reset()
    mjx_data = mjx_data.replace(qpos=jp.array(temp_env._base_env.physics.data.qpos))
    mjx_data = mjx_data.replace(qvel=jp.array(temp_env._base_env.physics.data.qvel))
    mjx_data = jax.jit(mjx.forward)(mjx_model, mjx_data)
    def get_body_name_to_id_dict(mjx_model):
        """Returns a dictionary mapping body names to their IDs."""
        body_name_to_id = {}
        for body_id in range(mjx_model.nbody):
            # Get the start index of this body's name in the names buffer
            start = mjx_model.name_bodyadr[body_id]
            # Find the end of the name (next null byte)
            end = start
            while end < len(mjx_model.names) and mjx_model.names[end] != 0:
                end += 1
            # Extract the name as a string
            name_bytes = mjx_model.names[start:end]
            name = name_bytes.decode('utf-8')
            body_name_to_id[name] = body_id
        return body_name_to_id
    body_dict = get_body_name_to_id_dict(mjx_model)
    adroit_bodies = [body_id for name, body_id in body_dict.items() if 'adroit' in name]
    # def _object_touching_table(mjx_data, mjx_model, table_geom_ids, object_body_id):
    #     for i in range(len(mjx_data.contact.geom1)):
    #         if mjx_model.geom_bodyid[mjx_data.contact.geom1[i]] == object_body_id:
    #             if mjx_data.contact.geom2[i] == table_geom_ids[0]:
    #                 return 1.
    #         elif mjx_model.geom_bodyid[mjx_data.contact.geom2[i]] == object_body_id:
    #             if mjx_data.contact.geom1[i] == table_geom_ids[0]:
    #                 return 1.
    #     return 0.
    # table_geom_ids = [geom_id for geom_id in range(mjx_model.ngeom) if mjx_model.geom_bodyid[geom_id] == body_dict['table']]
    # object_body_id = body_dict['hammer/object']
    # print(_object_touching_table(mjx_data, mjx_model, table_geom_ids, object_body_id))
    # def object_dynamics(ctrl, mjx_data, mjx_model, object_bodyid):
    #     mjx_data = mjx_data.replace(ctrl=ctrl)
    #     mjx_data = mjx.step(mjx_model, mjx_data)
    #     mjx_data = mjx.forward(mjx_model, mjx_data)
    #     return mjx_data.cvel[object_bodyid]
    #     # return mjx_data.qvel[-6:]
    # get_jacobian = jax.jit(jax.jacrev(object_dynamics))
    # # Jac = get_jacobian(jp.ones(mjx_model.nu), mjx_data, mjx_model, object_bodyid)
    # def _object_touching_adroit(mjx_data, mjx_model, adroit_bodies, object_body_id):
    #     for i in range(len(mjx_data.contact.geom1)):
    #         if mjx_model.geom_bodyid[mjx_data.contact.geom1[i]] == object_body_id:
    #             if mjx_model.geom_bodyid[mjx_data.contact.geom2[i]] in adroit_bodies:
    #                 return 1.
    #         elif mjx_model.geom_bodyid[mjx_data.contact.geom2[i]] == object_body_id:
    #             if mjx_model.geom_bodyid[mjx_data.contact.geom1[i]] in adroit_bodies:
    #                 return 1.
    #     return 0.
    # from mujoco.mjx._src import support
    # jit_single_contact_normal_force = jax.jit(single_contact_normal_force, static_argnums=(2,))
    # def normal_force(mjx_data, mjx_model, adroit_bodies, object_body_id):
    #     total_normal_force = 0.
    #     for i_con in range(len(mjx_data.contact.geom1)):
    #         if mjx_model.geom_bodyid[mjx_data.contact.geom1[i_con]] == object_body_id:
    #             if mjx_model.geom_bodyid[mjx_data.contact.geom2[i_con]] in adroit_bodies:
    #                 total_normal_force += jit_single_contact_normal_force(mjx_model, mjx_data, i_con)
    #         elif mjx_model.geom_bodyid[mjx_data.contact.geom2[i_con]] == object_body_id:
    #             if mjx_model.geom_bodyid[mjx_data.contact.geom1[i_con]] in adroit_bodies:
    #                 total_normal_force += jit_single_contact_normal_force(mjx_model, mjx_data, i_con)
    #     return total_normal_force
    # import jax
    # import jax.numpy as jnp
    # import flax.linen as nn
    # from functools import partial
    # from mujoco.mjx._src import support
    # from typing import List
    # class BranchedComputationModule(nn.Module):
    #     num_branches: int
    #     object_body_id: List
    #     adroit_bodies: List

    #     def setup(self):
    #         def branch_function(mjx_model, mjx_data, pred, i):
    #             true_fn = partial(lambda mjx_model, mjx_data, i: support.contact_force(mjx_model, mjx_data, i), i=i)
    #             false_fn = lambda mjx_model, mjx_data: jp.zeros(6,)
    #             return jax.lax.cond(pred, true_fn, false_fn, mjx_model, mjx_data)
    #             # return support.contact_force(mjx_model, mjx_data, i)

    #         self.branches = [partial(branch_function, i=j) for j in range(self.num_branches)]
    #         self.branch_switcher = lambda idx, mjx_model, mjx_data, pred: jax.lax.switch(idx, self.branches, mjx_model, mjx_data, pred)

    #     def __call__(self, mjx_model, mjx_data, geom_bodyid):

    #         def body_fn(i, pred, mjx_model, mjx_data):
    #             contact_force = self.branch_switcher(i, mjx_model, mjx_data, pred)
    #             force_vector = contact_force[:3]
    #             contact_normal = mjx_data.contact.frame[i][0,:]
    #             normal_force = jp.dot(force_vector, contact_normal)
    #             return normal_force
    #         batch_body_fn = jax.vmap(body_fn, in_axes=(0,0,None,None))

    #         obj1 = jp.isin(geom_bodyid[mjx_data.contact.geom1],jp.array(self.object_body_id))
    #         adr2 = jp.isin(geom_bodyid[mjx_data.contact.geom2],jp.array(self.adroit_bodies))
    #         obj2 = jp.isin(geom_bodyid[mjx_data.contact.geom2],jp.array(self.object_body_id))
    #         adr1 = jp.isin(geom_bodyid[mjx_data.contact.geom1],jp.array(self.adroit_bodies))
    #         obj_adr_contact = jp.logical_or(jp.logical_and(obj1,adr2),jp.logical_and(obj2,adr1))

    #         normal_force = batch_body_fn(jnp.arange(self.num_branches).astype(jp.int32), obj_adr_contact, mjx_model, mjx_data)
    #         total_force = jnp.sum(normal_force)
    #         return total_force
    # module = BranchedComputationModule(num_branches=mjx_data.contact.geom1.shape[0],
    #                                    object_body_id=object_body_id,
    #                                    adroit_bodies=adroit_bodies)
    # jit_forces = jax.jit(module.apply)
    # total_force = jit_forces({}, mjx_model, mjx_data, jp.array(mjx_model.geom_bodyid))
    # breakpoint()
    
    # import numpy as np
    # from functools import partial
    # def _decode_pyramid(pyramid, mu, condim):
    #     """Converts pyramid representation to contact force."""
    #     force = jp.zeros(6, dtype=float)
    #     force = jax.lax.cond(condim == 1, lambda force, pyramid: force.at[0].set(pyramid[0]), lambda force, pyramid: force, force, pyramid)
    #     force = force.at[0].set(pyramid[0 : 2 * (condim - 1)].sum())
    #     i = jp.arange(0, condim - 1)
    #     force = force.at[i + 1].set((pyramid[2 * i] - pyramid[2 * i + 1]) * mu[i])
    #     return force
    # def _contact_force(mjx_data, contact_id, branch_switcher, contact_efc_address, contact_dim):
    #     efc_address = contact_efc_address[contact_id]
    #     condim = contact_dim[contact_id]
    #     pyramid = jax.lax.dynamic_slice(mjx_data.efc_force,(efc_address,),(6,))
    #     # efc_address = mjx_data.contact.efc_address[contact_id]
    #     # condim = mjx_data.contact.dim[contact_id]
    #     # force = _decode_pyramid(mjx_data.efc_force[efc_address:], mjx_data.contact.friction[contact_id], condim)
    #     force = branch_switcher(condim, pyramid, mjx_data.contact.friction[contact_id])
    #     return force
    # # from mujoco.mjx._src import support
    # def single_contact_normal_force(mjx_model, mjx_data, i_con, branch_switcher, contact_efc_address, contact_dim):
    #     # https://roboti.us/book/programming.html
    #     # https://github.com/google-deepmind/mujoco/blob/d3bc0544d39c0933067e493f4c1c7ebfd15ae9f3/mjx/mujoco/mjx/_src/support.py
    #     # contact_force = support.contact_force(mjx_model, mjx_data, i_con)
    #     contact_force = _contact_force(mjx_data, i_con, branch_switcher, contact_efc_address, contact_dim)
    #     force_vector = contact_force[:3]
    #     contact_normal = mjx_data.contact.frame[i_con][0,:]
    #     normal_force = jp.dot(force_vector, contact_normal)
    #     return normal_force
    # # breakpoint()
    # # mjx_data = mjx_data.replace(contact=mjx_data.contact.replace(efc_address=jp.array(mjx_data.contact.efc_address)))
    # # mjx_data = mjx_data.replace(contact=mjx_data.contact.replace(dim=jp.array(mjx_data.contact.dim)))

    # num_branches = 4
    # branches = [partial(_decode_pyramid, condim=j+1) for j in range(num_branches)]
    # branch_switcher = lambda condim, pyramid, mu: jax.lax.switch(condim, branches, pyramid, mu)
    # batch_single_contact_normal_force = jax.jit(jax.vmap(partial(single_contact_normal_force, branch_switcher=branch_switcher,contact_efc_address=jp.array(mjx_data.contact.efc_address),contact_dim=jp.array(mjx_data.contact.dim)), in_axes=(None,None,0)))
    # xxx = batch_single_contact_normal_force(mjx_model, mjx_data, jp.arange(mjx_data.contact.geom1.shape[0]))
    # breakpoint()
    # print(_object_touching_adroit(mjx_data, mjx_model, adroit_bodies, object_body_id))
    # print(Jac)
    # breakpoint()
    # print(normal_force(mjx_data, mjx_model, adroit_bodies, object_body_id))
    # from functools import partial
    # jax.jit(partial(normal_force, adroit_bodies=adroit_bodies, object_body_id=object_body_id))(mjx_data, mjx_model)
    # obj1 = jp.isin(mjx_model.geom_bodyid[mjx_data.contact.geom1],jp.array(object_body_id))
    # adr2 = jp.isin(mjx_model.geom_bodyid[mjx_data.contact.geom2],jp.array(adroit_bodies))
    # obj2 = jp.isin(mjx_model.geom_bodyid[mjx_data.contact.geom2],jp.array(object_body_id))
    # adr1 = jp.isin(mjx_model.geom_bodyid[mjx_data.contact.geom1],jp.array(adroit_bodies))
    # obj_adr_contact = jp.logical_or(jp.logical_and(obj1,adr2),jp.logical_and(obj2,adr1))
    # total_normal_force = 0.
    # for i_con in range(len(obj_adr_contact)):
        # if obj_adr_contact[i_con]:
            # total_normal_force += jax.jit(single_contact_normal_force, static_argnuns=(2,))(mjx_model, mjx_data, i_con)
    # breakpoint()
    # body_rotation = mjx_data.xmat[object_body_id].reshape(3, 3)
    # qvel_angular = body_rotation @ mjx_data.cvel[object_bodyid][0:3]  
    # qvel_linear = mjx_data.cvel[object_bodyid][3:6]  
    # qvel = np.concatenate([qvel_linear, qvel_angular])
    # mjx_data.qvel[-6:]


    # qvel_linear  = body_rotation @ mjx_data.cvel[object_bodyid][3:6]
    # qvel_angular = body_rotation @ mjx_data.cvel[object_bodyid][0:3]
    # import numpy as np
    # qvel = np.concatenate([qvel_linear, qvel_angular])
    # mjx_data.cvel[object_bodyid]

    # breakpoint()

    if resume_model:
        model = OBJEX_PPO.load(resume_model, env)
        model._last_obs = None
        reset_num_timesteps = False
        total_timesteps -= model.num_timesteps
        if total_timesteps <= 0:
            return model
    else:
        model = OBJEX_PPO(
                        ActorCriticPolicy, 
                        env, verbose=1, 
                        tensorboard_log=f"logs/", 
                        n_steps=n_steps, 
                        gamma=config.agent.params.gamma,
                        gae_lambda=config.agent.params.gae_lambda,
                        learning_rate=config.agent.params.learning_rate,
                        ent_coef=config.agent.params.ent_coef,
                        vf_coef=config.agent.params.vf_coef,
                        guide_coef=config.agent.params.guide_coef,
                        clip_range=config.agent.params.clip_range,
                        batch_size=config.agent.params.batch_size,
                        n_epochs=config.agent.params.n_epochs,
                        policy_kwargs=make_policy_kwargs(config.agent.policy_kwargs)
                    )
        # initialize the agent with behavior cloning if desired
        if config.agent.params.warm_start_mean:
            warm_start = get_warm_start(config.env)
            bias = torch.from_numpy(warm_start)
            model.policy.set_action_bias(bias)
        reset_num_timesteps = True
    
    # initialize callbacks and train
    eval_env = make_eval_env(multi_proc, config.n_eval_envs, **config.env)
    eval_callback = EvalCallback(eval_freq, eval_env)
    restore_callback = FallbackCheckpoint(restore_freq)
    log_info = InfoCallback()
    checkpoint = CheckpointCallback(save_freq=save_freq, save_path=f'logs/', 
                                    name_prefix='rl_models')
    wandb = WandbCallback(model_save_path="models/", verbose=2)
    return model.learn(
                        total_timesteps=total_timesteps,
                        mjx_model=mjx_model,
                        object_bodyid=object_bodyid,
                        adroit_bodies=adroit_bodies,
                        callback=[log_info, eval_callback, checkpoint, restore_callback, wandb],
                        reset_num_timesteps=reset_num_timesteps
                      )

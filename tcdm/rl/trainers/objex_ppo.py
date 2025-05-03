# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from tcdm.rl.models.OBJEX.ppo import PPO as OBJEX_PPO
# from tcdm.rl.trainers.util import make_env, make_policy_kwargs, \
from tcdm.rl.models.OBJEX.util import make_env
from tcdm.rl.trainers.util import make_policy_kwargs, \
                                   InfoCallback, FallbackCheckpoint, \
                                   get_warm_start
from wandb.integration.sb3 import WandbCallback
from tcdm.rl.models.OBJEX.policies import ActorCriticPolicy
# from tcdm.rl.models.policies import ActorCriticPolicy
# from tcdm.rl.trainers.eval import make_eval_env, EvalCallback
from tcdm.rl.trainers.eval import EvalCallback
from tcdm.rl.models.OBJEX.eval import make_eval_env
from stable_baselines3.common.callbacks import CheckpointCallback

def objex_ppo_trainer(config, resume_model=None):
    total_timesteps = config.total_timesteps
    eval_freq = int(config.eval_freq // config.n_envs)
    save_freq = int(config.save_freq // config.n_envs)
    restore_freq = int(config.restore_checkpoint_freq // config.n_envs)
    n_steps = int(config.agent.params.n_steps // config.n_envs)
    multi_proc = bool(config.agent.multi_proc)
    env = make_env(multi_proc=multi_proc, **config.env)

    # import mujoco
    # object_geom_name_to_id = {}
    # adroit_geom_name_to_id = {}
    # for i in range(env.get_attr("unwrapped")[0]._base_env.physics.model.ngeom):
    #     name = mujoco.mj_id2name(env.get_attr("unwrapped")[0]._base_env.physics.model._model, mujoco.mjtObj.mjOBJ_GEOM, i)
    #     if "adroit/C_th" in name or "adroit/C_ff" in name or "adroit/C_mf" in name or "adroit/C_rf" in name or "adroit/C_lf" in name:
    #         adroit_geom_name_to_id[name] = i
    #     elif "hammer/" in name:
    #         object_geom_name_to_id[name] = i

    # breakpoint()

    # env.reset()
    # for i in range(50):
    #     env.step(env.action_space.sample())
    # import mujoco
    # for con in env.get_attr("unwrapped")[0]._base_env.physics.data.contact:
    #     body_id = env.get_attr("unwrapped")[0]._base_env.physics.model.geom(con.geom1).bodyid[0]
    #     print(mujoco.mj_id2name(env.get_attr("unwrapped")[0]._base_env.physics.model._model, mujoco.mjtObj.mjOBJ_BODY, body_id))
    #     body_id = env.get_attr("unwrapped")[0]._base_env.physics.model.geom(con.geom2).bodyid[0]
    #     print(mujoco.mj_id2name(env.get_attr("unwrapped")[0]._base_env.physics.model._model, mujoco.mjtObj.mjOBJ_BODY, body_id))
    #     print(mujoco.mj_id2name(env.get_attr("unwrapped")[0]._base_env.physics.model._model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1))
    #     print(mujoco.mj_id2name(env.get_attr("unwrapped")[0]._base_env.physics.model._model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2))

    # breakpoint()

    # def get_body_name_to_id_dict(mjx_model):
    #     """Returns a dictionary mapping body names to their IDs."""
    #     body_name_to_id = {}
    #     for body_id in range(mjx_model.nbody):
    #         # Get the start index of this body's name in the names buffer
    #         start = mjx_model.name_bodyadr[body_id]
    #         # Find the end of the name (next null byte)
    #         end = start
    #         while end < len(mjx_model.names) and mjx_model.names[end] != 0:
    #             end += 1
    #         # Extract the name as a string
    #         name_bytes = mjx_model.names[start:end]
    #         name = name_bytes.decode('utf-8')
    #         body_name_to_id[name] = body_id
    #     return body_name_to_id
    # body_dict = get_body_name_to_id_dict(env.get_attr("unwrapped")[0]._base_env.physics.model)

    # import mujoco
    # for i in range(env.get_attr("unwrapped")[0]._base_env.physics.model.ngeom):
    #     name = mujoco.mj_id2name(env.get_attr("unwrapped")[0]._base_env.physics.model._model, mujoco.mjtObj.mjOBJ_GEOM, i)
    #     print(f"{i}: {name}")

    # breakpoint()

    if resume_model:
        model = OBJEX_PPO.load(resume_model, env)
        model._last_obs = None
        reset_num_timesteps = False
        total_timesteps -= model.num_timesteps
        if total_timesteps <= 0:
            return model
    else:
        policy_kwargs = make_policy_kwargs(config.agent.policy_kwargs)
        policy_kwargs['pi_and_Q_observations']=env.get_attr('pi_and_Q_observations')[0]
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
                        dynamics_learning_rate=config.agent.params.dynamics_learning_rate,
                        dynamics_n_epochs=config.agent.params.dynamics_n_epochs,
                        net_arch_dyn=config.agent.params.net_arch_dyn,
                        dynamics_dropout=config.agent.params.dynamics_dropout,
                        pi_and_Q_observations=env.get_attr('pi_and_Q_observations')[0],
                        controlled_variables=env.get_attr('controlled_variables')[0],
                        policy_kwargs=policy_kwargs
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
                        callback=[log_info, eval_callback, checkpoint, restore_callback, wandb],
                        reset_num_timesteps=reset_num_timesteps
                      )

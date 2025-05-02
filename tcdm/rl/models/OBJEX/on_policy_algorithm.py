import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy#, ActorCriticPolicy
from tcdm.rl.models.OBJEX.common_policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

from mujoco import mjx
import jax
from jax import numpy as jp
from functools import partial
import flax.linen as nn
from mujoco.mjx._src import support
from typing import List

class BranchedComputationModule(nn.Module):
    num_branches: int
    num_geoms: int
    object_body_id: List
    adroit_bodies: List

    def setup(self):

        # https://github.com/google-deepmind/mujoco/issues/1555
        def decode_pyramid(pyramid, condim):
            """Converts pyramid representation to contact force."""
            normal_force = abs(jax.lax.cond(condim == 1, lambda pyramid: pyramid[0], lambda pyramid: pyramid[0 : 2 * (condim - 1)].sum(), pyramid))
            return normal_force

        # self.branches = [partial(decode_pyramid, condim=j+1) for j in range(self.num_branches)]
        # self.branch_switcher = partial(jax.lax.switch, branches=self.branches)
        self.branches = [partial(decode_pyramid, condim=j) for j in [1,3,4]]
        self.branch_switcher = lambda condim, pyramid: jax.lax.switch(condim, self.branches, pyramid)

    def __call__(self, mjx_data, geom_bodyid):

        def _contact_force(mjx_data, contact_id, efc_address, contact_dim):
            condim = contact_dim[contact_id]
            pyramid = jax.lax.dynamic_slice(mjx_data.efc_force,(efc_address,),(6,))
            normal_force = self.branch_switcher(condim, pyramid)
            return normal_force
        def single_contact_normal_force(contact_id, mjx_data, contact_efc_address, contact_dim):
            efc_address = contact_efc_address[contact_id]
            normal_force = _contact_force(mjx_data, contact_id, efc_address, contact_dim)
            return normal_force * (efc_address >= 0)

        batch_single_contact_normal_force = jax.vmap(single_contact_normal_force, in_axes=(0,None,None,None))

        obj1 = jp.isin(geom_bodyid[mjx_data.contact.geom1],jp.array(self.object_body_id))
        adr2 = jp.isin(geom_bodyid[mjx_data.contact.geom2],jp.array(self.adroit_bodies))
        obj2 = jp.isin(geom_bodyid[mjx_data.contact.geom2],jp.array(self.object_body_id))
        adr1 = jp.isin(geom_bodyid[mjx_data.contact.geom1],jp.array(self.adroit_bodies))
        obj_adr_contact = jp.logical_or(jp.logical_and(obj1,adr2),jp.logical_and(obj2,adr1))

        normal_force = batch_single_contact_normal_force(jp.arange(self.num_geoms), mjx_data, jp.array(mjx_data.contact.efc_address), jp.array(mjx_data.contact.dim))
        total_force = jp.sum(normal_force * obj_adr_contact)
        return total_force

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

#         normal_force = batch_body_fn(jp.arange(self.num_branches).astype(jp.int32), obj_adr_contact, mjx_model, mjx_data)
#         total_force = jp.sum(normal_force)
#         return total_force

class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param policy_base: The base policy used by this method
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        policy_base: Type[BasePolicy] = ActorCriticPolicy,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
    ):

        super(OnPolicyAlgorithm, self).__init__(
            policy=policy,
            env=env,
            policy_base=policy_base,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else RolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    @partial(jax.jit, static_argnames=["self"])
    def batch_get_channelled_noise(self, jax_dist_mean, jax_obs, jax_actions, mjx_data, mjx_model, object_bodyid):

        def modified_gram_schmidt(vectors):
            """
            adapted from https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/math/gram_schmidt.py
            fundamental change: while_loop replaced with scan
            Args:
            vectors: A Tensor of shape `[d, n]` of `d`-dim column vectors to
              orthonormalize.

            Returns:
            A Tensor of shape `[d, n]` corresponding to the orthonormalization.
            """
            def body_fn(vecs, i):
                u = jp.nan_to_num(vecs[:,i]/jp.linalg.norm(vecs[:,i]))
                weights = u@vecs
                masked_weights = jp.where(jp.arange(num_vectors) > i, weights, 0.)
                vecs = vecs - jp.outer(u,masked_weights)
                return vecs, None
            num_vectors = vectors.shape[-1]
            vectors, _ = jax.lax.scan(body_fn, vectors, jp.arange(num_vectors - 1))
            vec_norm = jp.linalg.norm(vectors, axis=0, keepdims=True)
            return jp.nan_to_num(vectors/vec_norm)
        
        def object_dynamics(ctrl, obs_tensor, mjx_data, mjx_model, object_bodyid):
                mjx_data = mjx_data.replace(qpos=obs_tensor[:36])
                mjx_data = mjx_data.replace(qvel=obs_tensor[155:(155+36)])
                mjx_data = mjx_data.replace(ctrl=ctrl)
                mjx_data = mjx.step(mjx_model, mjx_data)
                # object forces not considered - will want to add later
                mjx_data = mjx.forward(mjx_model, mjx_data)
                return mjx_data.cvel[object_bodyid]
                # total_force = self.total_force({}, mjx_model, mjx_data, jp.array(mjx_model.geom_bodyid))
                # return jp.concatenate((mjx_data.cvel[object_bodyid],total_force[...,None]))
                # return mjx_data.qvel[-6:]
        
        # get_jacobian = jax.jit(jax.vmap(jax.jacrev(object_dynamics), in_axes=(0,0,None,None,None)))
        get_jacobian = jax.jacrev(object_dynamics)

        def get_channel(F):

            action_dim = self.action_space.shape[0]
            F_augmented = jp.concatenate((F.T, jp.eye(action_dim)[:,:(action_dim-F.shape[0])]), axis=1)
            channel = modified_gram_schmidt(F_augmented)

            # scale = jp.ones(channel.shape)
            # scale = scale.at[:,:(action_dim-F.shape[0])].set(0.1)
            # channel = scale * channel

            return channel

        def get_channelled_noise(dist_mean, obs_tensor, actions, mjx_data, mjx_model, object_bodyid):

            # F = np.array(get_jacobian(jp.array(dist_mean), jp.array(obs_tensor.cpu()), mjx_data, mjx_model, object_bodyid))
            # FT = np.transpose(F,(0,2,1))
            # F_augmented = np.concatenate((FT, np.eye(self.action_space.shape[0])[:,:(self.action_space.shape[0]-6)]), axis=1)
            # synergies = modified_gram_schmidt(F_augmented)
            # noise = actions - dist_mean
            # channelled_noise = np.einsum("...ij,...j->...i", synergies, noise)
            # channelled_actions =  dist_mean + channelled_noise
            F = get_jacobian(dist_mean, obs_tensor, mjx_data, mjx_model, object_bodyid)
            channel = jax.lax.cond(jp.all(F == 0.), lambda F: jp.eye(self.action_space.shape[0]), get_channel, F)
            # channel = get_channel(F)
            noise = actions - dist_mean
            channelled_noise = channel @ noise
            return channelled_noise
        
        batch_get_channelled_noise = jax.vmap(get_channelled_noise, in_axes=(0,0,0,None,None,None))

        channelled_noise = batch_get_channelled_noise(jax_dist_mean, jax_obs, jax_actions, mjx_data, mjx_model, object_bodyid)

        return channelled_noise

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        mjx_model,
        mjx_data,
        # object_bodyid: int,
        # adroit_bodies,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        # def modified_gram_schmidt(vectors):
        #     """
        #     adapted from https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/math/gram_schmidt.py
        #     fundamental change: while_loop replaced with scan
        #     Args:
        #     vectors: A Tensor of shape `[d, n]` of `d`-dim column vectors to
        #       orthonormalize.

        #     Returns:
        #     A Tensor of shape `[d, n]` corresponding to the orthonormalization.
        #     """
        #     def body_fn(vecs, i):
        #         u = np.nan_to_num(vecs[:,i]/jp.linalg.norm(vecs[:,i]))
        #         weights = u@vecs
        #         masked_weights = np.where(np.arange(num_vectors) > i, weights, 0.)
        #         vecs = vecs - np.outer(u,masked_weights)
        #         return vecs
        #     num_vectors = vectors.shape[-1]
        #     for i in np.arange(num_vectors - 1):
        #         vectors = body_fn(vectors, i)
        #     vec_norm = np.linalg.norm(vectors, axis=0, keepdims=True)
        #     return np.nan_to_num(vectors/vec_norm)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs, distribution = self.policy.forward(obs_tensor)

            ######## synergies ########
            # jax_actions = jp.array(actions).astype(jp.float32)
            # actions = actions.cpu().numpy()
            # dist_mean = distribution.mode() #.cpu().numpy()
            # jax_dist_mean = jp.array(dist_mean.contiguous()).astype(jp.float32)
            # jax_obs = jp.array(obs_tensor).astype(jp.float32)
            # # batch_jax_data = jax.jit(jax.vmap(lambda mjx_data, jax_obs: mjx_data.replace(qpos=jax_obs[...,:36]), in_axes=(None,0)))(mjx_data, jax_obs)
            # # batch_jax_data = jax.jit(jax.vmap(lambda mjx_data, jax_obs: mjx_data.replace(qvel=jax_obs[...,155:(155+36)])))(batch_jax_data, jax_obs)
            # # channelled_noise = batch_get_channelled_noise(jax_dist_mean, jax_obs, jax_actions, mjx_data, mjx_model, object_bodyid)
            # channelled_noise = self.batch_get_channelled_noise(jax_dist_mean, jax_obs, jax_actions, mjx_data, mjx_model, object_bodyid)
            # channelled_actions = dist_mean.cpu().numpy() + np.array(channelled_noise)
            # clipped_actions = channelled_actions
            # if isinstance(self.action_space, gym.spaces.Box):
            #     clipped_actions = np.clip(channelled_actions, self.action_space.low, self.action_space.high)
            
            
            # F = np.array(get_jacobian(jp.array(dist_mean), jp.array(obs_tensor.cpu()), mjx_data, mjx_model, object_bodyid))
            # FT = np.transpose(F,(0,2,1))
            # F_augmented = np.concatenate((FT, np.eye(self.action_space.shape[0])[:,:(self.action_space.shape[0]-6)]), axis=1)
            # synergies = modified_gram_schmidt(F_augmented)
            # noise = actions - dist_mean
            # channelled_noise = np.einsum("...ij,...j->...i", synergies, noise)
            # channelled_actions = dist_mean + channelled_noise
            # synergies = np.eye(env.action_space.low.size)[None,:,:].repeat(obs_tensor.shape[0], axis=0)
            # rotated_actions = np.einsum("...ij,...j->...i", synergies, actions)
            ######## synergies ########
            actions = actions.cpu().numpy()
            clipped_actions = actions
            # Rescale and perform action
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # if self.num_timesteps > 20_000:
            #     # check observations are not normalized
            #     np.isclose(env.get_attr("unwrapped")[0]._base_env.physics.data.qpos,new_obs[0,:36]).all()
            #     np.isclose(env.get_attr("unwrapped")[0]._base_env.physics.data.qvel,new_obs[0,155:(155+36)]).all()

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = obs_as_tensor(new_obs, self.device)
            _, values, _, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self,
        mjx_model,
        object_bodyid: int,
        adroit_bodies,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        mjx_data =  mjx.make_data(mjx_model)

        # module = BranchedComputationModule(num_branches=mjx_data.contact.geom1.shape[0],
        #                                     object_body_id=object_bodyid,
        #                                     adroit_bodies=adroit_bodies)
        module = BranchedComputationModule(num_branches=4,
                                           num_geoms=mjx_data.contact.geom1.shape[0],
                                           object_body_id=object_bodyid,
                                           adroit_bodies=adroit_bodies)
        # jit_forces = jax.jit(module.apply)
        self.total_force = module.apply

        def batch_get_channel(jax_dist_mean, jax_obs, mjx_data, mjx_model, object_bodyid, get_force):

            def modified_gram_schmidt(vectors):
                """
                adapted from https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/math/gram_schmidt.py
                fundamental change: while_loop replaced with scan
                Args:
                vectors: A Tensor of shape `[d, n]` of `d`-dim column vectors to
                orthonormalize.

                Returns:
                A Tensor of shape `[d, n]` corresponding to the orthonormalization.
                """
                def body_fn(vecs, i):
                    u = jp.nan_to_num(vecs[:,i]/jp.linalg.norm(vecs[:,i]))
                    weights = u@vecs
                    masked_weights = jp.where(jp.arange(num_vectors) > i, weights, 0.)
                    vecs = vecs - jp.outer(u,masked_weights)
                    return vecs, None
                num_vectors = vectors.shape[-1]
                vectors, _ = jax.lax.scan(body_fn, vectors, jp.arange(num_vectors - 1))
                vec_norm = jp.linalg.norm(vectors, axis=0, keepdims=True)
                return jp.nan_to_num(vectors/vec_norm)
            
            def denormalize_action(action, mjx_model):
                ac_min, ac_max = mjx_model.actuator_ctrlrange.T
                ac_mid = 0.5 * (ac_max + ac_min)
                ac_range = 0.5 * (ac_max - ac_min)
                return jp.clip(action, -1., 1.) * ac_range + ac_mid
            
            def object_dynamics(ctrl, obs_tensor, mjx_data, mjx_model, object_bodyid, get_force):
                    mjx_data = mjx_data.replace(qpos=obs_tensor[:36])
                    mjx_data = mjx_data.replace(qvel=obs_tensor[155:(155+36)])
                    mjx_data = mjx_data.replace(ctrl=denormalize_action(ctrl, mjx_model))
                    mjx_data = mjx.step(mjx_model, mjx_data)
                    mjx_data = mjx.forward(mjx_model, mjx_data)
                    return mjx_data.cvel[object_bodyid]
                    # total_force = get_force({}, mjx_data, jp.array(mjx_model.geom_bodyid))
                    # return jp.concatenate((mjx_data.cvel[object_bodyid], total_force[...,None]))
                    # return mjx_data.qvel[-6:]

            get_jacobian = jax.jacrev(object_dynamics)

            def get_channelled_noise(dist_mean, obs_tensor, mjx_data, mjx_model, object_bodyid, get_force):
                F = get_jacobian(dist_mean, obs_tensor, mjx_data, mjx_model, object_bodyid, get_force)
                # only do GS if F is full rank (all elements of the controlled variable can be influenced)
                # 2 options in other case: simply normalize rows of F or return a zero matrix
                is_full_rank = (jp.linalg.matrix_rank(F) == min(F.shape))
                channel = jax.lax.cond(is_full_rank, lambda FT: modified_gram_schmidt(FT), lambda FT: FT * 0., F.T)
                # channel = jax.lax.cond(is_full_rank, lambda FT: FT / jp.linalg.norm(FT, axis=0, keepdims=True), lambda FT: FT * 0., F.T)
                # channel = jax.lax.cond(jp.all(jp.any(F != 0., axis=1)), lambda FT: FT / jp.linalg.norm(FT, axis=0, keepdims=True), lambda FT: modified_gram_schmidt(FT), F.T)
                return channel
            batch_get_channelled_noise = jax.vmap(get_channelled_noise, in_axes=(0,0,None,None,None,None))

            channel = batch_get_channelled_noise(jax_dist_mean, jax_obs, mjx_data, mjx_model, object_bodyid, get_force)

            return channel

        self.policy.jit_batch_get_channel = jax.jit(partial(batch_get_channel, mjx_data=mjx_data, mjx_model=mjx_model, object_bodyid=object_bodyid, get_force=module.apply))

        while self.num_timesteps < total_timesteps:

            # continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, mjx_model, mjx_data, object_bodyid, adroit_bodies, n_rollout_steps=self.n_steps)
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, mjx_model, mjx_data, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
import warnings
from typing import Any, Dict, Optional, Type, Union, List

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from types import SimpleNamespace

# from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
# from stable_baselines3.common.policies import ActorCriticPolicy
from tcdm.rl.models.OBJEX.on_policy_algorithm import OnPolicyAlgorithm
from tcdm.rl.models.OBJEX.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from tcdm.rl.models.OBJEX.dynamics import Dynamics

from torch import nn
from torch.distributions import Normal, Independent

class EntropyCoefModule(nn.Module):
    def __init__(self, names, ent_coefs, device='cuda'):
        super().__init__()
        assert len(names) == len(ent_coefs), "Names and ent_coefs must be the same length"
        self.params = nn.ParameterDict({
            name: nn.Parameter(th.ones(1, device=device) * np.log(ent_coef))
            for name, ent_coef in zip(names, ent_coefs)
        })

class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        guide_coef: float = 0.,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        dynamics_learning_rate: float = 0.001,
        net_arch_dyn: List = [],
        pi_and_Q_observations: List = [],
        controlled_variables: List = [],
        dynamics_dropout: List = [],
        dynamics_n_epochs: int = 40,
        target_entropy: Dict[str, Any] = None,
        ent_coef_lr: float = 1e-5,
        clip_grad_ent_coef: bool = True,
        diagonal_entropy_when_touching: bool = True,
        low_rank_ent_scale: float = 1.,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(PPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        assert (
            batch_size > 1
        ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl

        self.guide_coef = guide_coef
        self.dynamics_learning_rate = dynamics_learning_rate
        self.net_arch_dyn = net_arch_dyn
        self.dynamics_dropout = dynamics_dropout
        self.dynamics_n_epochs = dynamics_n_epochs
        self.pi_and_Q_observations = pi_and_Q_observations
        self.controlled_variables = controlled_variables
        self.target_entropy = target_entropy
        self.ent_coef_lr = ent_coef_lr
        self.clip_grad_ent_coef = clip_grad_ent_coef
        self.diagonal_entropy_when_touching = diagonal_entropy_when_touching
        self.low_rank_ent_scale = low_rank_ent_scale
        self.standard_PPO = policy_kwargs['standard_PPO']

        self.action_dim = self.env.action_space.shape[0]

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(PPO, self)._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

        if not self.standard_PPO:

            self.policy.dynamics = SimpleNamespace()

            self.policy.dynamics.model = Dynamics(action_dim=self.action_dim,
                                                net_arch_dyn=self.net_arch_dyn,
                                                dynamics_observations=self.pi_and_Q_observations,
                                                controlled_variables=self.controlled_variables,
                                                drop_out_rates=self.dynamics_dropout).to("cuda")

            self.policy.dynamics.optimizer = th.optim.Adam(self.policy.dynamics.model.parameters(),
                                                        lr=self.dynamics_learning_rate,
            )

            self.policy.log_ent_coef = SimpleNamespace()

            names = ['diagonal', 'explore']
            ent_coefs = [self.ent_coef] * len(names)
            self.policy.log_ent_coef = EntropyCoefModule(names, ent_coefs)

            self.policy.log_ent_coef.optimizer = th.optim.Adam(self.policy.log_ent_coef.parameters(), lr=self.ent_coef_lr)
        
        # self.policy.log_ent_coef.param = nn.Parameter(th.ones(1, device='cuda') * np.log(self.ent_coef), requires_grad=True).to("cuda")
        # self.policy.log_ent_coef.optimizer = th.optim.Adam([self.policy.log_ent_coef.param],
                                                            # lr=self.ent_coef_lr,
        # )

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """

        dynamics_losses = []
        if not self.standard_PPO:

            for epoch in range(self.dynamics_n_epochs):
                for rollout_data in self.rollout_buffer.get(self.batch_size):

                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        actions = rollout_data.actions.long().flatten()              

                    s_prime_mean, s_prime_log_var = self.policy.dynamics.model(rollout_data.observations, th.clamp(actions, -1., 1.))
                    
                    delta_obs = rollout_data.next_observations-rollout_data.observations

                    # log likelihood loss
                    dist = Independent(Normal(loc=s_prime_mean, scale=(0.5*s_prime_log_var).exp()), 1)
                    log_probs = dist.log_prob(delta_obs[:,self.controlled_variables]) # shape: [batch_size]
                    dynamics_loss = -log_probs.mean()

                    # mse loss
                    # dynamics_loss = F.mse_loss(s_prime_mean[touching_object], delta_obs[touching_object][:,self.controlled_variables])

                    # Optimization step
                    self.policy.dynamics.optimizer.zero_grad()
                    dynamics_loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.dynamics.model.parameters(), self.max_grad_norm)
                    self.policy.dynamics.optimizer.step()

                    dynamics_losses.append(dynamics_loss.item())

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        diagonal_entropies = []
        explore_entropies = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            total_sum = 0.0
            total_count = 0
            ostds = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                # TODO: investigate why there is no issue with the gradient
                # if that line is commented (as in SAC)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy, diagonal_entropy, explore_entropy, ostd, zstd = self.policy.evaluate_actions(rollout_data.observations, actions)

                with th.no_grad():
                    ostds.append(ostd.mean().cpu().numpy())
                
                non_nan_mask = ~th.isnan(zstd)
                sum_non_nan = th.sum(zstd[non_nan_mask])
                count_non_nan = th.sum(non_nan_mask)

                # Accumulate
                total_sum += sum_non_nan
                total_count += count_non_nan
                
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if self.standard_PPO:

                    # if entropy is None:
                    #     # Approximate entropy when no analytical form
                    #     entropy_loss = -th.mean(self.ent_coef * -log_prob)
                    # else:
                    #     entropy_loss = -th.mean(self.ent_coef * entropy)

                    entropy_loss = -th.mean(self.ent_coef * entropy)

                else:

                    if self.diagonal_entropy_when_touching:
                        diagonal_entropy_loss = -self.ent_coef * th.mean(diagonal_entropy)
                    else:
                        touching_object = rollout_data.observations[:,-1] > 0
                        diagonal_entropy_loss = -self.ent_coef * th.mean(diagonal_entropy[~touching_object])
                    explore_entropy_loss = -self.ent_coef * th.mean(explore_entropy) * self.low_rank_ent_scale
                    entropy_loss = diagonal_entropy_loss + explore_entropy_loss

                    entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + entropy_loss
                 )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        zstd_mean = total_sum / total_count

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/diagonal_entropy", np.mean(diagonal_entropies))
        self.logger.record("train/explore_entropy", np.mean(explore_entropies))
        # self.logger.record("train/ent_coeff_explore", self.policy.log_ent_coef.params['explore'].exp().item())
        # self.logger.record("train/ent_coeff_diagonal", self.policy.log_ent_coef.params['diagonal'].exp().item())
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/dynamics_loss", np.mean(dynamics_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            # self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
            self.logger.record("train/std", np.mean(ostds))
            self.logger.record("train/stdz", zstd_mean.item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "PPO":

        return super(PPO, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )
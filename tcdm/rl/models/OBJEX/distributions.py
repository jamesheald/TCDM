"""Probability distributions."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import torch as th
from gym import spaces
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal

from stable_baselines3.common.preprocessing import get_action_dim

from torch.distributions import Normal, Independent, TransformedDistribution
from torch.distributions.transforms import TanhTransform

from jax import numpy as jp

class Distribution(ABC):
    """Abstract base class for distributions."""

    def __init__(self):
        super(Distribution, self).__init__()
        self.distribution = None

    @abstractmethod
    def proba_distribution_net(self, *args, **kwargs) -> Union[nn.Module, Tuple[nn.Module, nn.Parameter]]:
        """Create the layers and parameters that represent the distribution.

        Subclasses must define this, but the arguments and return type vary between
        concrete classes."""

    @abstractmethod
    def proba_distribution(self, *args, **kwargs) -> "Distribution":
        """Set parameters of the distribution.

        :return: self
        """

    @abstractmethod
    def log_prob(self, x: th.Tensor) -> th.Tensor:
        """
        Returns the log likelihood

        :param x: the taken action
        :return: The log likelihood of the distribution
        """

    @abstractmethod
    def entropy(self) -> Optional[th.Tensor]:
        """
        Returns Shannon's entropy of the probability

        :return: the entropy, or None if no analytical form is known
        """

    @abstractmethod
    def sample(self) -> th.Tensor:
        """
        Returns a sample from the probability distribution

        :return: the stochastic action
        """

    @abstractmethod
    def mode(self) -> th.Tensor:
        """
        Returns the most likely action (deterministic output)
        from the probability distribution

        :return: the stochastic action
        """

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()

    @abstractmethod
    def actions_from_params(self, *args, **kwargs) -> th.Tensor:
        """
        Returns samples from the probability distribution
        given its parameters.

        :return: actions
        """

    @abstractmethod
    def log_prob_from_params(self, *args, **kwargs) -> Tuple[th.Tensor, th.Tensor]:
        """
        Returns samples and the associated log probabilities
        from the probability distribution given its parameters.

        :return: actions and log prob
        """


def sum_independent_dims(tensor: th.Tensor) -> th.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class DiagGaussianDistribution(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int):
        super(DiagGaussianDistribution, self).__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        # mean_actions = nn.Linear(latent_dim, self.action_dim)
        mean_actions_and_logstd = nn.Linear(latent_dim, self.action_dim * 2)
        log_std = nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=True)
        # return mean_actions_and_logstd, log_std
        return mean_actions_and_logstd, log_std

    def proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor) -> "DiagGaussianDistribution":
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        action_std = th.ones_like(mean_actions) * log_std.exp() * 0.1
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> th.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        return self.distribution.mean

    # def actions_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, deterministic: bool = False) -> th.Tensor:
    #     # Update the proba distribution
    #     self.proba_distribution(mean_actions, log_std)
    #     return self.get_actions(deterministic=deterministic)

    # def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    #     """
    #     Compute the log probability of taking an action
    #     given the distribution parameters.

    #     :param mean_actions:
    #     :param log_std:
    #     :return:
    #     """
    #     actions = self.actions_from_params(mean_actions, log_std)
    #     log_prob = self.log_prob(actions)
    #     return actions, log_prob


class SquashedDiagGaussianDistribution(DiagGaussianDistribution):
    """
    Gaussian distribution with diagonal covariance matrix, followed by a squashing function (tanh) to ensure bounds.

    :param action_dim: Dimension of the action space.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, action_dim: int, epsilon: float = 1e-6):
        super(SquashedDiagGaussianDistribution, self).__init__(action_dim)
        # Avoid NaN (prevents division by zero or log of zero)
        self.epsilon = epsilon
        self.gaussian_actions = None

    def proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor) -> "SquashedDiagGaussianDistribution":
        super(SquashedDiagGaussianDistribution, self).proba_distribution(mean_actions, log_std)
        return self

    def log_prob(self, actions: th.Tensor, gaussian_actions: Optional[th.Tensor] = None) -> th.Tensor:
        # Inverse tanh
        # Naive implementation (not stable): 0.5 * torch.log((1 + x) / (1 - x))
        # We use numpy to avoid numerical instability
        if gaussian_actions is None:
            # It will be clipped to avoid NaN when inversing tanh
            gaussian_actions = TanhBijector.inverse(actions)

        # Log likelihood for a Gaussian distribution
        log_prob = super(SquashedDiagGaussianDistribution, self).log_prob(gaussian_actions)
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        log_prob -= th.sum(th.log(1 - actions ** 2 + self.epsilon), dim=1)
        return log_prob

    def entropy(self) -> Optional[th.Tensor]:
        # No analytical form,
        # entropy needs to be estimated using -log_prob.mean()
        return None

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        self.gaussian_actions = super().sample()
        return th.tanh(self.gaussian_actions)

    def mode(self) -> th.Tensor:
        self.gaussian_actions = super().mode()
        # Squash the output
        return th.tanh(self.gaussian_actions)

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        action = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(action, self.gaussian_actions)
        return action, log_prob


class CategoricalDistribution(Distribution):
    """
    Categorical distribution for discrete actions.

    :param action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int):
        super(CategoricalDistribution, self).__init__()
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(self, action_logits: th.Tensor) -> "CategoricalDistribution":
        self.distribution = Categorical(logits=action_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        return th.argmax(self.distribution.probs, dim=1)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class MultiCategoricalDistribution(Distribution):
    """
    MultiCategorical distribution for multi discrete actions.

    :param action_dims: List of sizes of discrete action spaces
    """

    def __init__(self, action_dims: List[int]):
        super(MultiCategoricalDistribution, self).__init__()
        self.action_dims = action_dims

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits (flattened) of the MultiCategorical distribution.
        You can then get probabilities using a softmax on each sub-space.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """

        action_logits = nn.Linear(latent_dim, sum(self.action_dims))
        return action_logits

    def proba_distribution(self, action_logits: th.Tensor) -> "MultiCategoricalDistribution":
        self.distribution = [Categorical(logits=split) for split in th.split(action_logits, tuple(self.action_dims), dim=1)]
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        # Extract each discrete action and compute log prob for their respective distributions
        return th.stack(
            [dist.log_prob(action) for dist, action in zip(self.distribution, th.unbind(actions, dim=1))], dim=1
        ).sum(dim=1)

    def entropy(self) -> th.Tensor:
        return th.stack([dist.entropy() for dist in self.distribution], dim=1).sum(dim=1)

    def sample(self) -> th.Tensor:
        return th.stack([dist.sample() for dist in self.distribution], dim=1)

    def mode(self) -> th.Tensor:
        return th.stack([th.argmax(dist.probs, dim=1) for dist in self.distribution], dim=1)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class BernoulliDistribution(Distribution):
    """
    Bernoulli distribution for MultiBinary action spaces.

    :param action_dim: Number of binary actions
    """

    def __init__(self, action_dims: int):
        super(BernoulliDistribution, self).__init__()
        self.action_dims = action_dims

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Bernoulli distribution.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Linear(latent_dim, self.action_dims)
        return action_logits

    def proba_distribution(self, action_logits: th.Tensor) -> "BernoulliDistribution":
        self.distribution = Bernoulli(logits=action_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self.distribution.log_prob(actions).sum(dim=1)

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy().sum(dim=1)

    def sample(self) -> th.Tensor:
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        return th.round(self.distribution.probs)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class StateDependentNoiseDistribution(Distribution):
    """
    Distribution class for using generalized State Dependent Exploration (gSDE).
    Paper: https://arxiv.org/abs/2005.05719

    It is used to create the noise exploration matrix and
    compute the log probability of an action with that noise.

    :param action_dim: Dimension of the action space.
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,)
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this ensures bounds are satisfied.
    :param learn_features: Whether to learn features for gSDE or not.
        This will enable gradients to be backpropagated through the features
        ``latent_sde`` in the code.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(
        self,
        action_dim: int,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        learn_features: bool = False,
        epsilon: float = 1e-6,
    ):
        super(StateDependentNoiseDistribution, self).__init__()
        self.action_dim = action_dim
        self.latent_sde_dim = None
        self.mean_actions = None
        self.log_std = None
        self.weights_dist = None
        self.exploration_mat = None
        self.exploration_matrices = None
        self._latent_sde = None
        self.use_expln = use_expln
        self.full_std = full_std
        self.epsilon = epsilon
        self.learn_features = learn_features
        if squash_output:
            self.bijector = TanhBijector(epsilon)
        else:
            self.bijector = None

    def get_std(self, log_std: th.Tensor) -> th.Tensor:
        """
        Get the standard deviation from the learned parameter
        (log of it by default). This ensures that the std is positive.

        :param log_std:
        :return:
        """
        if self.use_expln:
            # From gSDE paper, it allows to keep variance
            # above zero and prevent it from growing too fast
            below_threshold = th.exp(log_std) * (log_std <= 0)
            # Avoid NaN: zeros values that are below zero
            safe_log_std = log_std * (log_std > 0) + self.epsilon
            above_threshold = (th.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            # Use normal exponential
            std = th.exp(log_std)

        if self.full_std:
            return std
        # Reduce the number of parameters:
        return th.ones(self.latent_sde_dim, self.action_dim).to(log_std.device) * std

    def sample_weights(self, log_std: th.Tensor, batch_size: int = 1) -> None:
        """
        Sample weights for the noise exploration matrix,
        using a centered Gaussian distribution.

        :param log_std:
        :param batch_size:
        """
        std = self.get_std(log_std)
        self.weights_dist = Normal(th.zeros_like(std), std)
        # Reparametrization trick to pass gradients
        self.exploration_mat = self.weights_dist.rsample()
        # Pre-compute matrices in case of parallel exploration
        self.exploration_matrices = self.weights_dist.rsample((batch_size,))

    def proba_distribution_net(
        self, latent_dim: int, log_std_init: float = -2.0, latent_sde_dim: Optional[int] = None
    ) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the deterministic action, the other parameter will be the
        standard deviation of the distribution that control the weights of the noise matrix.

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :param latent_sde_dim: Dimension of the last layer of the features extractor
            for gSDE. By default, it is shared with the policy network.
        :return:
        """
        # Network for the deterministic action, it represents the mean of the distribution
        mean_actions_net = nn.Linear(latent_dim, self.action_dim)
        # When we learn features for the noise, the feature dimension
        # can be different between the policy and the noise network
        self.latent_sde_dim = latent_dim if latent_sde_dim is None else latent_sde_dim
        # Reduce the number of parameters if needed
        log_std = th.ones(self.latent_sde_dim, self.action_dim) if self.full_std else th.ones(self.latent_sde_dim, 1)
        # Transform it to a parameter so it can be optimized
        log_std = nn.Parameter(log_std * log_std_init, requires_grad=True)
        # Sample an exploration matrix
        self.sample_weights(log_std)
        return mean_actions_net, log_std

    def proba_distribution(
        self, mean_actions: th.Tensor, log_std: th.Tensor, latent_sde: th.Tensor
    ) -> "StateDependentNoiseDistribution":
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :param latent_sde:
        :return:
        """
        # Stop gradient if we don't want to influence the features
        self._latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        variance = th.mm(self._latent_sde ** 2, self.get_std(log_std) ** 2)
        self.distribution = Normal(mean_actions, th.sqrt(variance + self.epsilon))
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        if self.bijector is not None:
            gaussian_actions = self.bijector.inverse(actions)
        else:
            gaussian_actions = actions
        # log likelihood for a gaussian
        log_prob = self.distribution.log_prob(gaussian_actions)
        # Sum along action dim
        log_prob = sum_independent_dims(log_prob)

        if self.bijector is not None:
            # Squash correction (from original SAC implementation)
            log_prob -= th.sum(self.bijector.log_prob_correction(gaussian_actions), dim=1)
        return log_prob

    def entropy(self) -> Optional[th.Tensor]:
        if self.bijector is not None:
            # No analytical form,
            # entropy needs to be estimated using -log_prob.mean()
            return None
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        noise = self.get_noise(self._latent_sde)
        actions = self.distribution.mean + noise
        if self.bijector is not None:
            return self.bijector.forward(actions)
        return actions

    def mode(self) -> th.Tensor:
        actions = self.distribution.mean
        if self.bijector is not None:
            return self.bijector.forward(actions)
        return actions

    def get_noise(self, latent_sde: th.Tensor) -> th.Tensor:
        latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        # Default case: only one exploration matrix
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return th.mm(latent_sde, self.exploration_mat)
        # Use batch matrix multiplication for efficient computation
        # (batch_size, n_features) -> (batch_size, 1, n_features)
        latent_sde = latent_sde.unsqueeze(1)
        # (batch_size, 1, n_actions)
        noise = th.bmm(latent_sde, self.exploration_matrices)
        return noise.squeeze(1)

    def actions_from_params(
        self, mean_actions: th.Tensor, log_std: th.Tensor, latent_sde: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std, latent_sde)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self, mean_actions: th.Tensor, log_std: th.Tensor, latent_sde: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(mean_actions, log_std, latent_sde)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class TanhBijector(object):
    """
    Bijective transformation of a probability distribution
    using a squashing function (tanh)
    TODO: use Pyro instead (https://pyro.ai/)

    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, epsilon: float = 1e-6):
        super(TanhBijector, self).__init__()
        self.epsilon = epsilon

    @staticmethod
    def forward(x: th.Tensor) -> th.Tensor:
        return th.tanh(x)

    @staticmethod
    def atanh(x: th.Tensor) -> th.Tensor:
        """
        Inverse of Tanh

        Taken from Pyro: https://github.com/pyro-ppl/pyro
        0.5 * torch.log((1 + x ) / (1 - x))
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y: th.Tensor) -> th.Tensor:
        """
        Inverse tanh.

        :param y:
        :return:
        """
        eps = th.finfo(y.dtype).eps
        # Clip the action to avoid NaN
        return TanhBijector.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))

    def log_prob_correction(self, x: th.Tensor) -> th.Tensor:
        # Squash correction (from original SAC implementation)
        return th.log(1.0 - th.tanh(x) ** 2 + self.epsilon)


def make_proba_distribution(
    action_space: gym.spaces.Space, use_sde: bool = False, switching_mean: bool = False, dist_kwargs: Optional[Dict[str, Any]] = None
) -> Distribution:
    """
    Return an instance of Distribution for the correct type of action space

    :param action_space: the input action space
    :param use_sde: Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    :param dist_kwargs: Keyword arguments to pass to the probability distribution
    :return: the appropriate Distribution object
    """
    if dist_kwargs is None:
        dist_kwargs = {}

    if isinstance(action_space, spaces.Box):
        assert len(action_space.shape) == 1, "Error: the action space must be a vector"
        cls = StateDependentNoiseDistribution if use_sde else FullGaussianDistribution
        return cls(get_action_dim(action_space), **dist_kwargs)
    elif isinstance(action_space, spaces.Discrete):
        return CategoricalDistribution(action_space.n, **dist_kwargs)
    elif isinstance(action_space, spaces.MultiDiscrete):
        return MultiCategoricalDistribution(action_space.nvec, **dist_kwargs)
    elif isinstance(action_space, spaces.MultiBinary):
        return BernoulliDistribution(action_space.n, **dist_kwargs)
    else:
        raise NotImplementedError(
            "Error: probability distribution, not implemented for action space"
            f"of type {type(action_space)}."
            " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary."
        )


def kl_divergence(dist_true: Distribution, dist_pred: Distribution) -> th.Tensor:
    """
    Wrapper for the PyTorch implementation of the full form KL Divergence

    :param dist_true: the p distribution
    :param dist_pred: the q distribution
    :return: KL(dist_true||dist_pred)
    """
    # KL Divergence for different distribution types is out of scope
    assert dist_true.__class__ == dist_pred.__class__, "Error: input distributions should be the same type"

    # MultiCategoricalDistribution is not a PyTorch Distribution subclass
    # so we need to implement it ourselves!
    if isinstance(dist_pred, MultiCategoricalDistribution):
        assert dist_pred.action_dims == dist_true.action_dims, "Error: distributions must have the same input space"
        return th.stack(
            [th.distributions.kl_divergence(p, q) for p, q in zip(dist_true.distribution, dist_pred.distribution)],
            dim=1,
        ).sum(dim=1)

    # Use the PyTorch kl_divergence implementation
    else:
        return th.distributions.kl_divergence(dist_true.distribution, dist_pred.distribution)
    
from torch.distributions import MultivariateNormal
class FullGaussianDistribution(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int, epsilon: float = 1e-6, **kwargs):
        super(FullGaussianDistribution, self).__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None
        self.state_dependent_std = kwargs['state_dependent_std']
        self.use_tanh_bijector = kwargs['use_tanh_bijector']
        self.epsilon = epsilon # Avoid NaN (prevents division by zero or log of zero)
        self.gaussian_actions = None
        self.controlled_variables_dim = kwargs['controlled_variables_dim']

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions = nn.Linear(latent_dim, self.controlled_variables_dim)

        log_std = nn.Parameter(th.ones(self.controlled_variables_dim) * log_std_init, requires_grad=True)

        if self.state_dependent_std['low_rank']:
            log_std = []
        else:
            log_std = nn.Parameter(th.ones(self.controlled_variables_dim) * log_std_init, requires_grad=True)

        return mean_actions, log_std

    def proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor, channel: th.Tensor, log_std_init: float = 0.0) -> "FullGaussianDistribution":
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :param channel:
        :return:
        """
        if self.state_dependent_std['low_rank']:
            action_std = (log_std+log_std_init).exp()
        else:
            action_std = th.ones_like(mean_actions) * (log_std).exp()

        intermediate = th.bmm(channel, th.diag_embed(action_std**2))
        low_rank = th.bmm(intermediate, channel.transpose(1, 2))
        covariance_matrix = low_rank + th.diag_embed(th.ones_like(low_rank[:,:,0]) * 1e-5)

        loc = th.bmm(channel, mean_actions.unsqueeze(-1)).squeeze(-1)

        self.distribution = MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix)

        return self, action_std
            
    # def log_prob(self, actions: th.Tensor) -> th.Tensor:
        # """
        # Get the log probabilities of actions according to the distribution.
        # Note that you must first call the ``proba_distribution()`` method.

        # :param actions:
        # :return:
        # """
        # log_prob = self.distribution.log_prob(actions)
        # return sum_independent_dims(log_prob)
        # return log_prob

    def log_prob(self, actions: th.Tensor, gaussian_actions: Optional[th.Tensor] = None) -> th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return log_prob

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy()

    # def sample(self) -> th.Tensor:
    #     # Reparametrization trick to pass gradients
    #     return self.distribution.rsample()

    # def mode(self) -> th.Tensor:
    #     return self.distribution.mean

    def sample(self) -> th.Tensor:
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        return self.distribution.mean

    def actions_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, synergies: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        # self.proba_distribution(mean_actions, log_std, synergies)
        # return self.get_actions(deterministic=deterministic)
        return None

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, synergies: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param mean_actions:
        :param log_std:
        :return:
        """
        # actions = self.actions_from_params(mean_actions, log_std, synergies)
        # log_prob = self.log_prob(actions)
        # return actions, log_prob
        return None
    
from torch.distributions import LowRankMultivariateNormal
class SwitchingGaussianDistribution(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int, epsilon: float = 1e-6, **kwargs):
        super(SwitchingGaussianDistribution, self).__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None
        self.state_dependent_std = kwargs['state_dependent_std']
        self.use_tanh_bijector = kwargs['use_tanh_bijector']
        self.epsilon = epsilon # Avoid NaN (prevents division by zero or log of zero)
        self.gaussian_actions = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions = nn.Linear(latent_dim, self.action_dim + 7)

        if self.state_dependent_std['diagonal']==False and self.state_dependent_std['low_rank']==False:
            log_std = nn.Parameter(th.ones(self.action_dim + 7) * log_std_init, requires_grad=True)
        elif self.state_dependent_std['diagonal']==False and self.state_dependent_std['low_rank']:
            log_std = nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=True)
        elif self.state_dependent_std['diagonal'] and self.state_dependent_std['low_rank']==False:
            log_std = nn.Parameter(th.ones(7) * log_std_init, requires_grad=True)
        else:
            log_std = []

        return mean_actions, log_std

    def proba_distribution(self, mean_actions: th.Tensor, zlogstd: th.Tensor, log_std: th.Tensor, channel: th.Tensor, touching_table, log_std_init: float = 0.0) -> "FullGaussianDistribution":
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :param channel:
        :return:
        """
        if self.state_dependent_std['diagonal']:
            action_std = (log_std+log_std_init).exp()
        else:
            action_std = th.ones_like(mean_actions[..., :self.action_dim]) * (log_std).exp()

        if self.state_dependent_std['low_rank']:
            explore_std = (zlogstd+log_std_init).exp()
        else:
            # explore_std = (zlogstd[None,:].repeat(channel.shape[0],1)).exp()
            explore_std = th.ones_like(mean_actions[..., self.action_dim:]) * (zlogstd).exp()

        mean_actions_full = mean_actions[..., :self.action_dim]
        cov_diag_full = action_std**2
        cov_factor_full = th.zeros(mean_actions.shape[0], self.action_dim, mean_actions.shape[1]-self.action_dim, device='cuda')

        if (~touching_table).sum() == 0:

            mean_actions_manifold = th.zeros_like(mean_actions_full)
            cov_diag_manifold = th.zeros_like(cov_diag_full)
            cov_factor_manifold = th.zeros_like(cov_factor_full)

        else:

            mean_actions_latent = mean_actions[..., self.action_dim:]
            mean_actions_manifold = th.bmm(channel, mean_actions_latent.unsqueeze(-1)).squeeze(-1)
            cov_factor_manifold = th.bmm(channel, th.diag_embed(explore_std))
            cov_diag_manifold = (th.ones_like(action_std) * 1e-3)**2 # add diagonal std of 1e-3 to handle manifold changes during dynamics updates
            # intermediate = th.bmm(channel, th.diag_embed(explore_std**2))
            # low_rank = th.bmm(intermediate, channel.transpose(1, 2))
            # covariance_matrix_manifold = low_rank + th.diag_embed((th.ones_like(action_std) * 1e-3)**2) # add diagonal std of 1e-3 to handle manifold changes during dynamics updates

        # mean_actions = th.where(touching_table[:,None].expand_as(mean_actions_full), mean_actions_full, mean_actions_manifold)
        # covariance_matrix = th.where(touching_table[:,None,None].expand_as(covariance_matrix_full), covariance_matrix_full, covariance_matrix_manifold)
        # self.distribution = MultivariateNormal(loc=mean_actions, covariance_matrix=covariance_matrix)

        mean_actions = th.where(touching_table[:,None].expand_as(mean_actions_full), mean_actions_full, mean_actions_manifold)
        cov_diag = th.where(touching_table[:,None].expand_as(cov_diag_full), cov_diag_full, cov_diag_manifold)
        cov_factor = th.where(touching_table[:,None,None].expand_as(cov_factor_full), cov_factor_full, cov_factor_manifold)
        self.distribution = LowRankMultivariateNormal(loc=mean_actions, cov_factor=cov_factor, cov_diag=cov_diag)

        if self.state_dependent_std['low_rank']:
            return self, action_std, th.where((~touching_table)[:,None].expand_as(explore_std), explore_std, th.full(explore_std.shape, float('nan'), device=zlogstd.device, dtype=explore_std.dtype)), None, None, th.tensor(0.0)
        else:
            return self, action_std, explore_std, None, None, th.tensor(0.0)
            
    # def log_prob(self, actions: th.Tensor) -> th.Tensor:
        # """
        # Get the log probabilities of actions according to the distribution.
        # Note that you must first call the ``proba_distribution()`` method.

        # :param actions:
        # :return:
        # """
        # log_prob = self.distribution.log_prob(actions)
        # return sum_independent_dims(log_prob)
        # return log_prob

    def log_prob(self, actions: th.Tensor, gaussian_actions: Optional[th.Tensor] = None) -> th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        if self.use_tanh_bijector:
            if gaussian_actions is None:
                # It will be clipped to avoid NaN when inversing tanh
                gaussian_actions = TanhBijector.inverse(actions)

            # Log likelihood for a Gaussian distribution
            log_prob = self.distribution.log_prob(gaussian_actions)
            # Squash correction (from original SAC implementation)
            # this comes from the fact that tanh is bijective and differentiable
            log_prob -= th.sum(th.log(1 - actions ** 2 + self.epsilon), dim=1)
            return log_prob
        else:
            log_prob = self.distribution.log_prob(actions)
            return log_prob

    def entropy(self) -> th.Tensor:
        # return sum_independent_dims(self.distribution.entropy())
        if self.use_tanh_bijector:
            return None
        else:
            return self.distribution.entropy()

    # def sample(self) -> th.Tensor:
    #     # Reparametrization trick to pass gradients
    #     return self.distribution.rsample()

    # def mode(self) -> th.Tensor:
    #     return self.distribution.mean

    def sample(self) -> th.Tensor:
        if self.use_tanh_bijector:
            # Reparametrization trick to pass gradients
            self.gaussian_actions = self.distribution.rsample()
            return th.tanh(self.gaussian_actions)
        else:
            return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        if self.use_tanh_bijector:
            self.gaussian_actions = self.distribution.mean
            # Squash the output
            return th.tanh(self.gaussian_actions)
        else:
            return self.distribution.mean

    def actions_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, synergies: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        # self.proba_distribution(mean_actions, log_std, synergies)
        # return self.get_actions(deterministic=deterministic)
        return None

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, synergies: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param mean_actions:
        :param log_std:
        :return:
        """
        # actions = self.actions_from_params(mean_actions, log_std, synergies)
        # log_prob = self.log_prob(actions)
        # return actions, log_prob
        return None

from torch.distributions import Categorical, MixtureSameFamily
import torch.nn.functional as F
class MixtureGaussianDistribution(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int, epsilon: float = 1e-6, **kwargs):
        super(MixtureGaussianDistribution, self).__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None
        self.state_dependent_std = kwargs['state_dependent_std']
        self.use_tanh_bijector = kwargs['use_tanh_bijector']
        self.epsilon = epsilon # Avoid NaN (prevents division by zero or log of zero)
        self.gaussian_actions = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions = nn.Linear(latent_dim, self.action_dim + 7 + 2)

        if self.state_dependent_std['diagonal']==False and self.state_dependent_std['low_rank']==False:
            log_std = nn.Parameter(th.ones(self.action_dim + 7) * log_std_init, requires_grad=True)
        elif self.state_dependent_std['diagonal']==False and self.state_dependent_std['low_rank']:
            log_std = nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=True)
        elif self.state_dependent_std['diagonal'] and self.state_dependent_std['low_rank']==False:
            log_std = nn.Parameter(th.ones(7) * log_std_init, requires_grad=True)
        else:
            log_std = []

        return mean_actions, log_std

    def proba_distribution(self, mean_actions: th.Tensor, zlogstd: th.Tensor, log_std: th.Tensor, channel: th.Tensor, touching_table, log_std_init: float = 0.0) -> "FullGaussianDistribution":
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :param channel:
        :return:
        """
        if self.state_dependent_std['diagonal']:
            action_std = (log_std+log_std_init).exp()
        else:
            action_std = th.ones_like(mean_actions[..., :self.action_dim]) * (log_std).exp()

        if self.state_dependent_std['low_rank']:
            explore_std = (zlogstd+log_std_init).exp()
        else:
            # explore_std = (zlogstd[None,:].repeat(channel.shape[0],1)).exp()
            explore_std = th.ones_like(mean_actions[..., :7]) * (zlogstd).exp()

        mean_actions_full = mean_actions[..., :self.action_dim]
        cov_diag_full = action_std**2
        cov_factor_full = th.zeros(mean_actions.shape[0], self.action_dim, 7, device='cuda')

        mean_actions_latent = mean_actions[..., self.action_dim:(self.action_dim+7)]
        mean_actions_manifold = th.bmm(channel, mean_actions_latent.unsqueeze(-1)).squeeze(-1)
        cov_factor_manifold = th.bmm(channel, th.diag_embed(explore_std))
        cov_diag_manifold = (th.ones_like(action_std) * (mean_actions[..., -2].unsqueeze(-1)-7).exp())**2 # add diagonal std of 1e-3 to handle manifold changes during dynamics updates

        # Stack across new axis for mixture components
        means = th.stack([mean_actions_full, mean_actions_manifold], dim=1)      # [B, 2, d]
        factors = th.stack([cov_factor_full, cov_factor_manifold], dim=1)  # [B, 2, d, r]
        diags = th.stack([cov_diag_full, cov_diag_manifold], dim=1)        # [B, 2, d]

        # Mixture weights
        # Create two-component logits by appending a zero for the second component
        logits = mean_actions[..., -1]
        mix_logits = th.stack([logits+2., th.zeros_like(logits)], dim=1)  # shape [B, 2]
        mix = Categorical(logits=mix_logits)        # batch of mixtures

        # LowRankMultivariateNormal as components
        components = LowRankMultivariateNormal(means, factors, diags)  # shape [B, 2]

        # Mixture distribution
        self.distribution = MixtureSameFamily(mix, components)

        diagonal_dist = Independent(Normal(loc=th.zeros_like(action_std), scale=action_std), 1)
        diagonal_entropy = diagonal_dist.entropy()
        explore_dist = Independent(Normal(loc=th.zeros_like(explore_std), scale=explore_std), 1)
        explore_entropy = explore_dist.entropy()

        # Convert to probabilities using softmax over dim=1
        mix_probs = F.softmax(mix_logits, dim=1)[:,0]

        return self, action_std, explore_std, diagonal_entropy, explore_entropy, mix_probs
            
    # def log_prob(self, actions: th.Tensor) -> th.Tensor:
        # """
        # Get the log probabilities of actions according to the distribution.
        # Note that you must first call the ``proba_distribution()`` method.

        # :param actions:
        # :return:
        # """
        # log_prob = self.distribution.log_prob(actions)
        # return sum_independent_dims(log_prob)
        # return log_prob

    def log_prob(self, actions: th.Tensor, gaussian_actions: Optional[th.Tensor] = None) -> th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        if self.use_tanh_bijector:
            if gaussian_actions is None:
                # It will be clipped to avoid NaN when inversing tanh
                gaussian_actions = TanhBijector.inverse(actions)

            # Log likelihood for a Gaussian distribution
            log_prob = self.distribution.log_prob(gaussian_actions)
            # Squash correction (from original SAC implementation)
            # this comes from the fact that tanh is bijective and differentiable
            log_prob -= th.sum(th.log(1 - actions ** 2 + self.epsilon), dim=1)
            return log_prob
        else:
            log_prob = self.distribution.log_prob(actions)
            return log_prob

    def entropy(self) -> th.Tensor:
        actions = self.get_actions(deterministic=False)
        log_prob = self.log_prob(actions)
        return -log_prob

    # def sample(self) -> th.Tensor:
    #     # Reparametrization trick to pass gradients
    #     return self.distribution.rsample()

    # def mode(self) -> th.Tensor:
    #     return self.distribution.mean

    def sample(self) -> th.Tensor:
        if self.use_tanh_bijector:
            self.gaussian_actions = self.distribution.sample()
            return th.tanh(self.gaussian_actions)
        else:
            return self.distribution.sample()

    def mode(self) -> th.Tensor:
        if self.use_tanh_bijector:
            self.gaussian_actions = self.distribution.mean
            # Squash the output
            return th.tanh(self.gaussian_actions)
        else:
            return self.distribution.mean

    def actions_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, synergies: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        # self.proba_distribution(mean_actions, log_std, synergies)
        # return self.get_actions(deterministic=deterministic)
        return None

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, synergies: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param mean_actions:
        :param log_std:
        :return:
        """
        # actions = self.actions_from_params(mean_actions, log_std, synergies)
        # log_prob = self.log_prob(actions)
        # return actions, log_prob
        return None
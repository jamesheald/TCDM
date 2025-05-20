import torch as th
import torch.nn as nn
import torch.nn.functional as F

class Dynamics(nn.Module):
    def __init__(self, action_dim, net_arch_dyn, dynamics_observations, controlled_variables, drop_out_rates):
        super().__init__()
        self.dynamics_observations = dynamics_observations
        self.controlled_variables = controlled_variables
        self.drop_out_rates = drop_out_rates

        layers = []
        input_dim = len(dynamics_observations) + action_dim
        self.net_arch_dyn = net_arch_dyn

        # Build dynamics network
        for i, h_dim in enumerate(net_arch_dyn):
            seq = nn.Sequential(
                nn.Linear(input_dim if i == 0 else net_arch_dyn[i - 1], h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(drop_out_rates[i]) if drop_out_rates[i] > 0 else nn.Identity()
            )
            layers.append(seq)
        
        self.dynamics = nn.ModuleList(layers)
        self.dynamics_out = nn.Linear(net_arch_dyn[-1], len(controlled_variables) * 2)

    def get_log_var(self, x):
        sigma = F.softplus(x) + 1e-6
        log_var = 2 * th.log(sigma)
        return log_var

    def forward(self, obs, u, deterministic=True):
        # obs, u: [batch, dim]
        x = th.cat((obs[..., self.dynamics_observations], u), dim=-1)
        for layer in self.dynamics:
            x = layer(x)
        x = self.dynamics_out(x)
        s_prime_mean, s_prime_scale = th.chunk(x, 2, dim=-1)
        s_prime_log_var = self.get_log_var(s_prime_scale)
        return s_prime_mean, s_prime_log_var
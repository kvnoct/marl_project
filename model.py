from torch import nn
import numpy as np
import utils

class MLPBase(nn.Module):
    def __init__(self, num_inputs, hidden_size=64):
        super(MLPBase, self).__init__()
        self.hidden_size = hidden_size

        init_ = lambda m: utils.net_init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

    def forward(self, inputs):
        x = inputs

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor
    
class Policy(nn.Module):
    def __init__(self, obs_space, action_space):
        super(Policy, self).__init__()

        obs_shape = obs_space[0].shape[0]

        self.base = MLPBase(obs_shape)

        num_outputs = action_space[0].n
        self.dist = utils.Categorical(self.base.hidden_size, num_outputs)

    def act(self, inputs, deterministic=False):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs

    def get_value(self, inputs):
        value, _ = self.base(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy
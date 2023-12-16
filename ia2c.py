from gym.spaces.utils import flatdim
from model import Policy
from storage import RolloutStorage
import torch.optim as optim
from torch import nn
import os
import torch

class IA2C:
    def __init__(
        self,
        agent_id,
        obs_space,
        action_space,
        device = 'cpu',
        use_gae = True,
        lr = 3e-4,
        gamma = 0.99,
        gae_lambda = 0.95,
        adam_eps = 0.001,
        num_steps = 5,
        num_processes = 1,
    ):
        self.agent_id = agent_id
        self.obs_size = flatdim(obs_space)
        self.action_size = flatdim(action_space)
        
        self.obs_space = obs_space
        self.action_space = action_space

        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.model = Policy(
            obs_space, action_space
        )

        self.storage = RolloutStorage(
            obs_space,
            action_space,
            num_steps,
            num_processes,
        )

        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr, eps=adam_eps)

        # self.intr_stats = RunningStats()
        self.saveables = {
            "model": self.model,
            "optimizer": self.optimizer,
        }

    def save(self, path):
        torch.save(self.saveables, os.path.join(path, "models.pt"))

    def restore(self, path):
        checkpoint = torch.load(os.path.join(path, "models.pt"))
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())


    def compute_returns(self):
        with torch.no_grad():
            next_value = self.model.get_value(
                self.storage.obs[-1]
            ).detach()

        self.storage.compute_returns(
            next_value, self.use_gae, self.gamma, self.gae_lambda,
        )

    def update(
        self,
        storages,
        value_loss_coef = 0.5,
        entropy_coef = 0.01,
        max_grad_norm = 0.5,
        device = 'cpu',
    ):

        obs_shape = self.storage.obs.size()[2:]
        action_shape = self.storage.actions.size()[-1]
        
        num_steps, num_processes, _ = self.storage.rewards.size()

        
        values, action_log_probs, dist_entropy = self.model.evaluate_actions(
            self.storage.obs[:-1].view(-1, *obs_shape),
            self.storage.actions.view(-1, action_shape),
        )

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = self.storage.returns[:-1] - values

        policy_loss = -(advantages.detach() * action_log_probs).mean()
        value_loss = advantages.pow(2).mean()



        self.optimizer.zero_grad()
        (
            policy_loss
            + value_loss_coef * value_loss
            - entropy_coef * dist_entropy
        ).backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss_coef * value_loss.item(),
            "dist_entropy": entropy_coef * dist_entropy.item(),
        }
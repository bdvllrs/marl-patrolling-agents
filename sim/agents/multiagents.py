import math
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam

from model.dqn import DQNCritic, DQNActor
from sim.agents.agents import Agent, soft_update
from utils import Config

config = Config('./config')


class AgentMADDPG(Agent):
    def __init__(self, type, agent_id, device, agent_config):
        super(AgentMADDPG, self).__init__(type, agent_id, device, agent_config)

        self.policy_critic = DQNCritic().to(self.device)  # Q'
        self.target_critic = DQNCritic().to(self.device)  # Q

        self.policy_actor = DQNActor().to(self.device)  # mu'
        self.target_actor = DQNActor().to(self.device)  # mu

        self.critic_optimizer = Adam(self.policy_critic.parameters(), lr=config.agents.lr)
        self.actor_optimizer = Adam(self.policy_actor.parameters(), lr=config.agents.lr_actor)

        self.update(self.target_critic, self.policy_critic)
        self.update(self.target_actor, self.policy_actor)

        self.target_critic.eval()
        self.target_actor.eval()

        self.steps_done = 0
        self.agents = None
        self.current_agent_idx = None

    def add_agents(self, agents, idx):
        self.agents = agents
        self.current_agent_idx = idx

    def draw_action(self, state, no_exploration=False):
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1. * self.steps_done / self.EPS_DECAY)
        with torch.no_grad():
            state = torch.tensor(state).to(self.device).unsqueeze(dim=0).reshape(1, -1)
            p = np.random.random()
            if no_exploration or p > eps_threshold:
                action_probs = self.policy_actor(state).detach().cpu().numpy()
                action = np.argmax(action_probs[0])
            else:
                action = random.randrange(self.number_actions)
        self.steps_done += 1
        return action

    def learn_critic(self, batch):
        """
        :param batch:
        :return:
        """
        state_batch, next_state_batch, action_batch, reward_batch = batch

        state_batch = torch.FloatTensor(state_batch[:, self.current_agent_idx]).to(self.device)  # batch x dim
        next_state_batch = torch.FloatTensor(next_state_batch[:, self.current_agent_idx]).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)  # batch x agents x action_dim
        reward_batch = torch.FloatTensor(reward_batch[:, self.current_agent_idx]).to(self.device).reshape(action_batch.size(0), 1)  # batch x 1

        self.critic_optimizer.zero_grad()

        target_actions = []
        policy_actions = []
        for a in range(len(self.agents)):
            target_action = self.agents[a].target_actor(next_state_batch)
            target_actions.append(target_action)
            if config.learning.gumbel_softmax:
                action = F.gumbel_softmax(action_batch[:, a], tau=config.learning.gumbel_softmax_tau)
            else:
                action = action_batch[:, a]
            policy_actions.append(action)

        predicted_q = self.policy_critic(state_batch, policy_actions)  # dim (batch_size x 1)
        target = self.target_critic(next_state_batch, target_actions)
        target_q = reward_batch + self.gamma * target

        loss = F.mse_loss(predicted_q, target_q)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_critic.parameters(), 1)

        self.critic_optimizer.step()

        if not self.steps_done % config.agents.soft_update_frequency:
            soft_update(self.target_critic, self.policy_critic)

        return loss.detach().cpu().item()

    def learn_actor(self, batch):
        state_batch, next_state_batch, action_batch, reward_batch = batch

        state_batch = torch.FloatTensor(state_batch[:, self.current_agent_idx]).to(self.device)  # batch  x dim
        action_batch = torch.FloatTensor(action_batch).to(self.device)  # batch x agents x action_dim

        self.actor_optimizer.zero_grad()
        predicted_action = self.policy_actor(state_batch)

        policy_actions = []
        for a in range(len(self.agents)):
            if a == self.current_agent_idx:
                policy_actions.append(predicted_action)
            else:
                if config.learning.gumbel_softmax:
                    action = F.gumbel_softmax(action_batch[:, a], tau=config.learning.gumbel_softmax_tau)
                else:
                    action = action_batch[:, a]
                policy_actions.append(action)

        actor_loss = -self.policy_critic(state_batch, policy_actions)
        actor_loss = actor_loss.mean()
        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_actor.parameters(), 1)

        self.actor_optimizer.step()

        if not self.steps_done % config.agents.soft_update_frequency:
            soft_update(self.target_actor, self.policy_actor)

        return actor_loss.detach().cpu().item()

    def save(self, name):
        """
        load models
        :param name: adress of saved models
        :return: models saved
        :return:
        """
        save_dict = {
            'policy_critic': self.policy_critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'policy_actor': self.policy_actor.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict()
        }

        torch.save(save_dict, name)

    def load(self, name):
        """
        load models
        :param name: adress of saved models
        :return: models init
        """
        params = torch.load(name)
        self.policy_critic.load_state_dict(params['policy_critic'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_actor.load_state_dict(params['policy_actor'])
        self.target_actor.load_state_dict(params['target_actor'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
        self.actor_optimizer.load_state_dict(params['actor_optimizer'])

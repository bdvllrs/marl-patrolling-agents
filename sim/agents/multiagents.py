import math
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam

from model.dqn import DQNCritic, DQNActor
from sim.agents.agents import Agent
from utils import Config, to_onehot
from utils.misc import gumbel_softmax

config = Config('./config')


class AgentMADQN(Agent):

    def __init__(self, type, agent_id, device, agent_config):
        super(AgentMADQN, self).__init__(type, agent_id, device, agent_config)

        self.policy_critic = DQNCritic().to(self.device)  # Q'
        self.target_critic = DQNCritic().to(self.device)  # Q

        self.policy_actor = DQNActor().to(self.device)  # mu'
        self.target_actor = DQNActor().to(self.device)  # mu

        self.policy_optimizer = Adam([{"params": self.policy_critic.parameters()},
                                      {"params": self.policy_actor.parameters()}], lr=config.agents.lr)
        self.update(self.target_critic, self.policy_critic)

        self.target_critic.eval()
        self.target_actor.eval()

        self.n_iter = 0
        self.steps_done = 0

    def draw_action(self, state, no_exploration=False):
        """
        Args:
            state:
            no_exploration: If True, use only exploitation policy
        """
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        with torch.no_grad():
            p = np.random.random()
            state = torch.tensor(state).to(self.device).unsqueeze(dim=0).reshape(1, -1)
            if no_exploration or p > eps_threshold:
                action_probs = self.policy_actor(state).detach().cpu().numpy()
                action = np.argmax(action_probs[0])
            else:
                action = random.randrange(self.number_actions)
            return action

    def hard_update(self, target, policy):
        """
        Copy network parameters from source to target
        """
        target.load_state_dict(policy.state_dict())

    def soft_update(self, target, policy, tau=config.learning.tau):
        for target_param, param in zip(target.parameters(), policy.parameters()):
            target_param.data.copy_(target_param.data * tau + param.data * (1. - tau))

    def learn(self, batch, target_actors, idx):
        state_batch, next_state_batch, action_batch, reward_batch = batch
        state_batch = torch.FloatTensor(state_batch).to(self.device)  # batch x agents x dim
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device).unsqueeze(2)  # batch x agents x 1
        reward_batch = torch.FloatTensor(reward_batch[:, idx], device=self.device)  # batch x dim

        state_batch = state_batch.reshape(state_batch.size(0), -1)  # batch x state_dim
        next_state_batch = next_state_batch.reshape(next_state_batch.size(0), -1)  # batch x state_dim

        self.policy_optimizer.zero_grad()

        action_dim = 7 if config.env.world_3D else 5

        target_actions = []
        policy_actions = []
        for a in range(len(target_actors)):
            target_action = target_actors[a](next_state_batch).max(1)[1].unsqueeze(1)
            onehot_target_action = to_onehot(target_action, action_dim)
            onehot_policy_action = to_onehot(action_batch[:, a], action_dim)
            target_actions.append(onehot_target_action)
            policy_actions.append(onehot_policy_action)


        predicted_q = self.policy_critic(state_batch, *policy_actions)  # dim (batch_size x 1)
        target_q = reward_batch + self.gamma * self.target_critic(next_state_batch, *target_actions)

        loss = F.mse_loss(predicted_q, target_q)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.target_critic.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.target_actor.parameters(), 1)

        self.policy_optimizer.step()

        self.soft_update(self.target_actor, self.policy_actor)
        self.soft_update(self.target_critic, self.policy_critic)

        return loss.detach().cpu().item()

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
            'policy_optimizer': self.policy_optimizer.state_dict(),
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
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])


class AgentMADDPG(Agent):
    def __init__(self, type, agent_id, device, agent_config):
        super(AgentMADDPG, self).__init__(type, agent_id, device, agent_config)

        self.policy_critic = DQNCritic().to(self.device)  # Q'
        self.target_critic = DQNCritic().to(self.device)  # Q

        self.policy_actor = DQNActor().to(self.device)  # mu'
        self.target_actor = DQNActor().to(self.device)  # mu

        self.critic_optimizer = Adam(self.policy_critic.parameters(), lr=config.agents.lr)
        self.actor_optimizer = Adam(self.policy_actor.parameters(), lr=config.agents.lr)


        self.update(self.target_critic, self.policy_critic)
        self.update(self.target_actor, self.policy_actor)

        self.target_critic.eval()
        self.target_actor.eval()

        self.n_iter = 0
        self.steps_done = 0


    def draw_action(self, state, no_exploration):
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        with torch.no_grad():
            state = torch.tensor(state).to(self.device).unsqueeze(dim=0).reshape(1, -1)
            if no_exploration:
                action_probs = self.policy_actor(state).detach().cpu().numpy()
                action = np.argmax(action_probs[0])
            elif config.learning.gumbel_softmax:
                action = gumbel_softmax(self.policy_actor(state), hard=True).max(1)[1].detach().cpu().numpy()[0]
            else:
                p = np.random.random()
                if no_exploration or p > eps_threshold:
                    action_probs = self.policy_actor(state).detach().cpu().numpy()
                    action = np.argmax(action_probs[0])
                else:
                    action = random.randrange(self.number_actions)
        return action


    def learn_critic(self, batch, target_actors, idx):
        """

        :param batch:
        :return:
        """
        state_batch, next_state_batch, action_batch, reward_batch = batch
        state_batch = torch.FloatTensor(state_batch[:, idx]).to(self.device)  # batch x agents x dim
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        next_state_batch_idx = torch.FloatTensor(next_state_batch[:, idx]).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device).unsqueeze(2)  # batch x agents x 1
        reward_batch = torch.FloatTensor(reward_batch[:, idx], device=self.device)  # batch x dim


        self.critic_optimizer.zero_grad()

        action_dim = 7 if config.env.world_3D else 5

        target_actions = []
        policy_actions = []
        for a in range(len(target_actors)):
            target_action = target_actors[a](next_state_batch_idx).max(1)[1].unsqueeze(1)
            onehot_target_action = to_onehot(target_action, action_dim)
            onehot_policy_action = to_onehot(action_batch[:, a], action_dim)
            target_actions.append(onehot_target_action)
            policy_actions.append(onehot_policy_action)

        predicted_q = self.policy_critic(state_batch, *policy_actions)  # dim (batch_size x 1)
        target_q = reward_batch + self.gamma * self.target_critic(next_state_batch_idx, *target_actions)

        loss = F.mse_loss(predicted_q, target_q)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_critic.parameters(), 1)

        self.critic_optimizer.step()

        self.soft_update(self.target_critic, self.policy_critic)

        return loss.detach().cpu().item()

    def learn_actor(self, batch, actors, idx):
        """

        :param batch: for 1 agent, learn
        :return: loss
        """
        state_batch, next_state_batch, action_batch, reward_batch = batch
        state_batch = torch.FloatTensor(state_batch).to(self.device)  # batch x agents x dim
        state_batch_agent = torch.FloatTensor(state_batch[:, idx]).to(self.device)  # batch  x dim

        self.actor_optimizer.zero_grad()
        action_dim = 7 if config.env.world_3D else 5
        n_agents = len(actors)

        curr_pol_out = self.policy_actor(state_batch_agent)
        #add noise !
        if config.learning.gumbel_softmax:
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_vf_in = to_onehot(curr_pol_out.max(1)[1].unsqueeze(1), action_dim)
        all_pol_acs = []
        for num, ac, ob in zip(range(n_agents), actors, state_batch.transpose(0, 1)):
            if num == idx:
                all_pol_acs.append(curr_pol_vf_in)
            else:
                all_pol_acs.append(to_onehot(ac(ob).max(1)[1].unsqueeze(1), action_dim))



        pol_loss = - self.policy_critic(state_batch_agent, *all_pol_acs).mean()
        pol_loss += (curr_pol_out ** 2).mean() * 1e-3
        pol_loss.backward()


        torch.nn.utils.clip_grad_norm_(self.policy_actor.parameters(), 1)

        self.actor_optimizer.step()
        self.soft_update(self.target_actor, self.policy_actor)

        return pol_loss.detach().cpu().item()


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

    def hard_update(self, target, policy):
        """
        Copy network parameters from source to target
        """
        target.load_state_dict(policy.state_dict())

    def soft_update(self, target, policy, tau=config.learning.tau):
        for target_param, param in zip(target.parameters(), policy.parameters()):
            target_param.data.copy_(target_param.data * tau + param.data * (1. - tau))
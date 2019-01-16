import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from torch.optim import Adam

from model.ActorCritic import ActorNetwork, CriticNetwork
from model.DQN import DQNUnit
from utils.config import Config
import torch.nn.functional as F
import math

from utils.misc import onehot_from_logits, gumbel_softmax

config = Config('./config')


class Agent:
    type = "prey"  # or predator
    id = 0
    # For RL
    gamma = 0.9
    EPS_START = 0.01
    lr = 0.1
    update_frequency = 0.1
    update_type = "hard"

    def __init__(self, type, agent_id, device, agent_config):
        assert type in ["prey", "predator"], "Agent type is not correct."
        self.type = type
        self.id = agent_id
        self.memory = None
        self.number_actions = agent_config.number_actions

        # For RL
        self.gamma = agent_config.gamma
        self.EPS_START = agent_config.EPS_START
        self.EPS_END = agent_config.EPS_END
        self.EPS_DECAY = agent_config.EPS_DECAY
        self.lr = agent_config.lr
        self.update_frequency = agent_config.update_frequency
        assert agent_config.update_type in ["hard", "soft"], "Update type is not correct."
        self.update_type = agent_config.update_type

        self.colors = {"prey": "#a1beed", "predator": "#ffd2a0"}

        self.device = device

    def draw_action(self, observation, no_exploration=False):
        raise NotImplementedError

    def update(self, *params):
        if self.update_type == "hard":
            self.hard_update(*params)
        elif self.update_type == "soft":
            self.soft_update(*params)

    def plot(self, position, reward, radius, ax: plt.Axes):
        x, y = position
        circle = plt.Circle((x, y), radius=radius, color=self.colors[self.type])
        ax.add_artist(circle)
        ax.text(x - radius/2, y, self.id)
        ax.text(x - radius/2, y-0.05, "Reward: {}".format(round(reward, 3)))

    def soft_update(self, *params):
        raise NotImplementedError

    def hard_update(self, *params):
        raise NotImplementedError

    def learn(self, batch):
        raise NotImplementedError

    def save(self, name):
        raise NotImplementedError

    def load(self, name):
        raise NotImplementedError


class AgentDQN(Agent):
    def __init__(self, type, agent_id, device, agent_config):
        super(AgentDQN, self).__init__(type, agent_id, device, agent_config)

        self.policy_net = DQNUnit().to(self.device)
        self.target_net = DQNUnit().to(self.device)
        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=config.agents.lr)
        self.update(self.target_net, self.policy_net)
        self.target_net.eval()

        self.n_iter = 0
        self.steps_done = 0

    def hard_update(self, target, policy):
        """
        Copy network parameters from source to target
        """
        target.load_state_dict(policy.state_dict())

    def soft_update(self, target, policy):
        raise NotImplementedError

    def draw_action(self, state, no_exploration=False):
        """
        Args:
            state:
            no_exploration: If True, use only exploitation policy
        """
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                                  math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        with torch.no_grad():
            p = np.random.random()
            state = torch.tensor(state).to(self.device).unsqueeze(dim=0)
            if no_exploration or p > eps_threshold:
                action_probs = self.policy_net(state).detach().cpu().numpy()
                action = np.argmax(action_probs[0])
            else:
                action = random.randrange(self.number_actions)
            return action

    def load(self, name):
        """
        load models
        :param name: adress of saved models
        :return: models init
        """
        params = torch.load(name)
        self.policy_net.load_state_dict(params['policy'])
        self.target_net.load_state_dict(params['target_policy'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])

    def save(self, name):
        """
        load models
        :param name: adress of saved models
        :return: models saved
        :return:
        """
        save_dict = {'policy': self.policy_net.state_dict(),
                     'target_policy': self.target_net.state_dict(),
                     'policy_optimizer': self.policy_optimizer.state_dict()}
        torch.save(save_dict, name)

    def learn(self, batch):
        """

        :param batch: for 1 agent, learn
        :return: loss
        """
        state_batch, next_state_batch, action_batch, reward_batch = batch
        state_batch = torch.FloatTensor(state_batch, device=self.device)
        next_state_batch = torch.FloatTensor(next_state_batch, device=self.device)
        action_batch = torch.LongTensor(action_batch, device=self.device)
        reward_batch = torch.FloatTensor(reward_batch, device=self.device)

        action_batch = action_batch.reshape(action_batch.size(0), 1)
        reward_batch = reward_batch.reshape(reward_batch.size(0), 1)

        policy_output = self.policy_net(state_batch)
        action_by_policy = policy_output.gather(1, action_batch)

        if config.learning.DDQN:
            actions_next = self.policy_net(next_state_batch).detach().max(1)[1].unsqueeze(1)
            Qsa_prime_targets = self.target_net(next_state_batch).gather(1, actions_next)

        else:
            Qsa_prime_targets = self.target_net(next_state_batch).detach().max(1)[0]

        actions_by_cal = reward_batch + (self.gamma * Qsa_prime_targets)

        loss = F.mse_loss(action_by_policy, actions_by_cal)
        self.policy_optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy_optimizer.step()

        if not self.n_iter % self.update_frequency:
            self.update(self.target_net, self.policy_net)

        self.n_iter += 1

        return loss.detach().cpu().item()

class AgentDDPG(Agent):
    def __init__(self, type, agent_id, device, agent_config):
        super(AgentDDPG, self).__init__(type, agent_id, device, agent_config)

        self.policy_net = ActorNetwork().to(self.device)
        self.critic_net = CriticNetwork().to(self.device)
        self.target_policy = ActorNetwork().to(self.device)
        self.target_critic = CriticNetwork().to(self.device)

        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=config.agents.lr)
        self.critic_optimizer = Adam(self.critic_net.parameters(), lr=config.agents.lr)

        self.update(self.target_policy, self.policy_net)
        self.update(self.target_critic, self.critic_net)
        self.n_iter =0



    def hard_update(self, target, policy):
        """
        Copy network parameters from source to target
        """
        target.load_state_dict(policy.state_dict())

    def soft_update(self, target, policy, tau = config.learning.tau):
        for target_param, param in zip(target.parameters(), policy.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


    def draw_action(self, state):

        with torch.no_grad():
            state = torch.tensor(state).to(self.device).unsqueeze(dim=0)
            action = gumbel_softmax(self.policy_net(state), hard=True).max(1)[1].detach().cpu().numpy()
            return action

    def load(self, name):
        """
        load models
        :param name: adress of saved models
        :return: models init
        """
        params = torch.load(name)
        self.policy_net.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_net.load_state_dict(params['critic'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

    def save(self, name):
        """
        load models
        :param name: adress of saved models
        :return: models saved
        :return:
        """
        save_dict = {'policy': self.policy_net.state_dict(),
                     'target_policy': self.target_policy.state_dict(),
                     'policy_optimizer': self.policy_optimizer.state_dict(),
                     'critic': self.critic_net.state_dict(),
                     'target_critic': self.target_critic.state_dict(),
                     'critic_optimizer': self.critic_optimizer.state_dict()}

        torch.save(save_dict, name)


    def learn_critic(self, batch):
        """

        :param batch:
        :return:
        """
        state_batch, next_state_batch, action_batch, reward_batch = batch
        state_batch = torch.FloatTensor(state_batch, device=self.device)
        next_state_batch = torch.FloatTensor(next_state_batch, device=self.device)
        action_batch = torch.LongTensor(action_batch, device=self.device)
        reward_batch = torch.FloatTensor(reward_batch, device=self.device)

        self.critic_optimizer.zero_grad()

        all_trgt_acs = self.target_policy(next_state_batch).max(1)[1].unsqueeze(1)
        target_value = reward_batch + (self.gamma * self.target_critic(next_state_batch, all_trgt_acs))
        actual_value = self.critic_net(state_batch, action_batch.unsqueeze(1))
        loss = F.mse_loss(actual_value, target_value.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 0.5)
        self.critic_optimizer.step()

        return loss.detach().cpu().item()




    def learn_policy(self, batch, idx):
        """

        :param batch: for 1 agent, learn
        :return: loss
        """
        state_batch, next_state_batch, action_batch, reward_batch = batch
        state_batch = torch.FloatTensor(state_batch, device=self.device)

        self.policy_optimizer.zero_grad()

        curr_pol_out = self.policy_net(state_batch)
        action = gumbel_softmax(curr_pol_out, hard=True).max(1)[1].unsqueeze(1)

        #action = self.policy_net(state_batch).max(1)[1].unsqueeze(1)
        # actor_loss is used to maximize the Q value for the predicted action
        actor_loss = - self.critic_net(state_batch, action)
        actor_loss = actor_loss.mean()
        actor_loss += (curr_pol_out ** 2).mean() * 1e-3
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()

        # update actor target network and critic target network
        if not self.n_iter % self.update_frequency:
            self.soft_update(self.target_critic, self.critic_net)
            self.soft_update(self.target_policy, self.policy_net)
        self.n_iter += 1

        return actor_loss.detach().cpu().item()


class AgentMADDPG(Agent):
    def __init__(self, type, agent_id, device, agent_config):
        super(AgentMADDPG, self).__init__(type, agent_id, device, agent_config)

        self.policy_net = ActorNetwork().to(self.device)
        self.critic_net = CriticNetwork().to(self.device)
        self.target_policy = ActorNetwork().to(self.device)
        self.target_critic = CriticNetwork().to(self.device)

        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=config.agents.lr)
        self.critic_optimizer = Adam(self.critic_net.parameters(), lr=config.agents.lr)

        self.update(self.target_policy, self.policy_net)
        self.update(self.target_critic, self.critic_net)
        self.n_iter =0



    def hard_update(self, target, policy):
        """
        Copy network parameters from source to target
        """
        target.load_state_dict(policy.state_dict())

    def soft_update(self, target, policy, tau = config.learning.tau):
        for target_param, param in zip(target.parameters(), policy.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


    def draw_action(self, state):

        with torch.no_grad():
            state = torch.tensor(state).to(self.device).unsqueeze(dim=0)
            action = gumbel_softmax(self.policy_net(state), hard=True).max(1)[1].detach().cpu().numpy()
            return action

    def load(self, name):
        """
        load models
        :param name: adress of saved models
        :return: models init
        """
        params = torch.load(name)
        self.policy_net.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_net.load_state_dict(params['critic'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

    def save(self, name):
        """
        load models
        :param name: adress of saved models
        :return: models saved
        :return:
        """
        save_dict = {'policy': self.policy_net.state_dict(),
                     'target_policy': self.target_policy.state_dict(),
                     'policy_optimizer': self.policy_optimizer.state_dict(),
                     'critic': self.critic_net.state_dict(),
                     'target_critic': self.target_critic.state_dict(),
                     'critic_optimizer': self.critic_optimizer.state_dict()}

        torch.save(save_dict, name)


    def learn_critic(self, batch, target_policies):
        """

        :param batch:
        :return:
        """
        state_batch, next_state_batch, action_batch, reward_batch = batch
        state_batch = torch.FloatTensor(state_batch, device=self.device)
        next_state_batch = torch.FloatTensor(next_state_batch, device=self.device)
        action_batch = torch.LongTensor(action_batch, device=self.device)
        reward_batch = torch.FloatTensor(reward_batch, device=self.device)
        self.critic_optimizer.zero_grad()

        self.n_agents = config.agents.number_preys+config.agents.number_predators
        batch_size = config.learning.batch_size
        all_trgt_acs = torch.cat( [(pi(nobs)).max(1)[1] for pi, nobs in
                        zip(target_policies, next_state_batch)] , 0).reshape(self.n_agents, batch_size)

        next_state_batch = next_state_batch.transpose(0, 1)
        all_trgt_acs = all_trgt_acs.transpose(0, 1)
        all_trgt_acs = all_trgt_acs.unsqueeze(2)
        state_batch = state_batch.transpose(0, 1)
        action_batch = action_batch.transpose(0, 1)


        target_value = reward_batch + (self.gamma * self.target_critic(next_state_batch, all_trgt_acs))
        actual_value = self.critic_net(state_batch, action_batch.unsqueeze(2))
        loss = F.mse_loss(actual_value, target_value.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 0.5)
        self.critic_optimizer.step()

        return loss.detach().cpu().item()




    def learn_policy(self, batch, idx, policies):
        """

        :param batch: for 1 agent, learn
        :return: loss
        """
        state_batch, next_state_batch, action_batch, reward_batch = batch
        state_batch = torch.FloatTensor(state_batch, device=self.device)

        self.policy_optimizer.zero_grad()

        batch_size = config.learning.batch_size

        action = gumbel_softmax(self.policy_net(state_batch[idx]), hard=True).max(1)[1].unsqueeze(1)


        all_pol_acs = []
        for i, pi, ob in zip(range(self.n_agents), policies, state_batch):
            if i == idx:
                all_pol_acs.append(action.squeeze(1))
            else:
                all_pol_acs.append(pi(ob).max(1)[1])
        all_pol_acs = torch.cat(all_pol_acs, 0).reshape(self.n_agents, batch_size)
        all_pol_acs = all_pol_acs.transpose(0,1)
        state_batch = state_batch.transpose(0,1)
        #action = self.policy_net(state_batch).max(1)[1].unsqueeze(1)
        #actor_loss is used to maximize the Q value for the predicted action
        actor_loss = - self.critic_net(state_batch, all_pol_acs.unsqueeze(2))
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()

        # update actor target network and critic target network
        if not self.n_iter % self.update_frequency:
            self.soft_update(self.target_critic, self.critic_net)
            self.soft_update(self.target_policy, self.policy_net)
        self.n_iter += 1

        return actor_loss.detach().cpu().item()



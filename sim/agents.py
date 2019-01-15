import torch
import numpy as np
import random
from torch.optim import Adam
from model.DQN import DQN_unit
from utils.config import Config

config = Config('./builds')

class Agent:
    type = "prey"  # or predator
    id = 0
    # For RL
    gamma = 0.9
    epsilon_greedy = 0.01
    lr = 0.1
    update_frequency = 0.1
    update_type = "hard"

    def __init__(self, type, agent_id, agent_config):
        assert type in ["prey", "predator"], "Agent type is not correct."
        self.type = type
        self.id = agent_id
        self.memory = None

        # For RL
        self.gamma = agent_config.gamma
        self.epsilon_greedy = agent_config.epsilon_greedy
        self.lr = agent_config.lr
        self.update_frequency = agent_config.update_frequency
        assert agent_config.update_type in ["hard", "soft"], "Update type is not correct."
        self.update_type = agent_config.update_type

        self.device = torch.device('cpu')

    def draw_action(self, observation):
        raise NotImplementedError

    def update(self):
        if self.update_type == "hard":
            self.hard_update()
        elif self.update_type == "soft":
            self.soft_update()

    def plot(self, state):
        pass

    def soft_update(self):
        raise NotImplementedError

    def hard_update(self):
        raise NotImplementedError

    def learn(self, batch):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def to(self, device):
        self.device = device
        return self


class AgentDQN(Agent):
    def __init__(self, type, agent_id, agent_config):
        super(AgentDQN, self).__init__(type, agent_id, agent_config)
        self.policy_net = DQN_unit()
        self.target_net = DQN_unit()
        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=config.agents.lr)
        self.hard_update(self.target_policy, self.policy)

    def hard_update(self, target, policy):
        """
        Copy network parameters from source to target

        """
        target.load_state_dict(policy.state_dict())

    def draw_action(self, state):
        p = np.random.random()
        if p < self.eps_greedy:
            action_probs = self.policy_net(state)
            action = np.argmax(action_probs[0])
        else:
            action = random.randrange(config.agents.nbre_actions)
        return action

    def load(self, name):
        """
        load models
        :param name: adress of saved models
        :return: models init
        """
        params = torch.load(name)
        self.policy.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])

    def save(self, name):
        """
        load models
        :param name: adress of saved models
        :return: models saved
        :return:
        """
        save_dict = {'policy': self.policy_net.state_dict(),
         'target_policy': self.target_policy.state_dict(),
         'policy_optimizer': self.policy_optimizer.state_dict()}
        torch.save(save_dict, name)


    def learn(self, batch):
        """

        :param batch: for 1 agent, learn
        :return: loss
        """
        pass


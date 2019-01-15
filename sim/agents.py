import torch


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

        self.policy_net = None
        self.target_net = None

    def draw_action(self, observation):
        action = 0
        return action


import random
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from utils import choice, possible_directions, position_from_direction, get_distance_between, state_from_observation
from model import DQN
import torch

__all__ = ["Officer", "Target", "Headquarters"]
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    type_id = None
    type = None
    position = None
    last_reward = None
    last_action = None
    color = "red"
    limit_board = None
    view_radius = 10  # manhattan distance
    max_size_history = 20000
    can_learn = False

    def __init__(self, name=None):
        assert (self.type_id is not None and
                self.type is not None), "This agent does not have any type information."
        if name is None:
            name = self.type + " " + str(np.random.randint(0, 1000))
        self.name = name
        self.id_to_action = ['none', 'top', 'left', 'right', 'bottom', 'top-left', 'bottom-left', 'top-right',
                             'bottom-right']
        self.action_to_id = {a: i for i, a in enumerate(self.id_to_action)}
        self.histories = []  # Memory for all the episodes
        self.loss_values = []

    def reset(self):
        if len(self.histories) > 0:
            self.histories[-1]["terminal"] = True  # New history

    def set_position(self, position):
        if len(self.histories) >= self.max_size_history:
            self.histories.pop(0)
        self.histories.append({"position": position, "terminal": False})  # Keep history of all positions
        self.position = position
        if len(self.histories) > 1 and not self.histories[-2]["terminal"]:
            self.histories[-1]["prev_position"] = self.histories[-2]["position"]

    def add_to_history(self, action, obs):
        self.histories[-1]['action'] = action
        self.last_action = action
        self.histories[-1]['state'] = state_from_observation(self, self.histories[-1]['position'], obs)
        if len(self.histories) > 2 and "state" in self.histories[-2].keys():
            self.histories[-1]["prev_state"] = self.histories[-2]["state"]

    def set_reward(self, reward):
        """
        Sets reward for last action.
        Has to be called after a set_position
        Args:
            reward:
        """
        self.histories[-1]["reward"] = reward
        self.last_reward = reward

    def view_area(self, position=None):
        """
        Defines the points in the board the agent can see
        """
        position = self.position if position is None else position
        positions = [position]
        x, y = position
        for i in range(-self.view_radius, self.view_radius + 1):
            for j in range(-self.view_radius, self.view_radius + 1):
                if 0 <= x - i < self.limit_board[0] and 0 <= y - j < self.limit_board[1]:
                    positions.append((x - i, y - j))
        # self.view_area_from(position, self.view_radius, positions)
        return positions

    def view_area_from(self, position, hops, positions):
        """
        Defines the points in the board the agent can see from the given position
        Args:
            positions: list of seen positions
            position: original position
            hops: Number of allowed hops
        """
        if hops == -1:
            return
        directions = possible_directions(self.limit_board, position)
        directions.remove('none')
        for direction in directions:
            new_position = position_from_direction(position, direction)
            if new_position not in positions:
                positions.append(new_position)
                self.view_area_from(new_position, hops - 1, positions)
        return

    def set_size_board(self, width, height):
        self.limit_board = (width, height)

    def plot(self, radius):
        x, y = self.position
        circle = plt.Circle((x, y), radius=radius, color=self.color)
        plt.gcf().gca().add_artist(circle)

    def draw_action(self, obs):
        """
        Select the action to perform
        Args:
            obs: information on where are the other agents. List of agents.
        """
        return choice(possible_directions(self.limit_board, self.position))


class RLAgent(Agent):
    can_learn = True

    def __init__(self, name, width=100, height=100, gamma=0.9):
        super(RLAgent, self).__init__(name)
        self.gamma = gamma
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.steps_done = 0
        # TODO: Define here policy and target net

        self.policy_net = DQN(height).to(device)
        self.target_net = DQN(height).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()


    # Exploration policy (epsilon greedy)
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                                  np.math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        state = torch.from_numpy(state).float().to(device)

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest value for column of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1]#.view(1, 1)
        else:
            return torch.tensor([[random.randrange(9)]], device=device, dtype=torch.long)


class Officer(RLAgent):
    """
    No MARL for this one. Each of them learns individually
    """
    type_id = 0
    type = "officer"
    color = "blue"
    view_radius = 100

    def draw_action(self, obs):
        """
        Select the action to perform
        Args:
            obs: information on where are the other agents. List of agents.
        """
        state = state_from_observation(self, self.position, obs)
        # TODO: predict the right action with the target net
        action = self.select_action(state)
        actions_possible = ['none', 'top', 'left', 'right', 'bottom', 'top-left', 'top-right', 'bottom-right',
         'bottom-left']
        index = action[0][0].item()
        return actions_possible[index]


class Headquarters(Agent):
    """
    MARL - represents several Officers
    """
    type_id = 2
    type = "hq"
    color = "black"


class Target(Agent):
    type_id = 1
    type = "target"

    def distance_to_officers(self, obs, direction=None):
        position = self.position if direction is None else position_from_direction(self.position, direction)
        sum_dist = np.inf
        for agent in obs:
            if agent.type == 'officer':
                if sum_dist == np.inf:
                    sum_dist = 0
                sum_dist += get_distance_between(self.limit_board, position, agent.position)
        return sum_dist

    def draw_action(self, obs):
        """
        Choose the position as far away from patrol
        Args:
            obs: information on where are the other agents. List of agents.
        """
        if self.distance_to_officers(obs) == np.inf:
            return 'none'
        directions = possible_directions(self.limit_board, self.position)
        chosen_direction = max(directions, key=lambda direction: self.distance_to_officers(obs, direction))
        return chosen_direction

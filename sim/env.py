from typing import List

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
from sim.rewards import reward_full
from sim.agents.agents import Agent
import random


class Env:
    agents: List[Agent]

    def __init__(self, env_config, config):
        self.reward_type = env_config.reward_type
        self.noise = env_config.noise
        self.board_size = env_config.board_size

        self.plot_radius = env_config.plot_radius

        self.possible_location_values = [float(k) / float(self.board_size) for k in range(self.board_size)]

        self.current_iteration = 0
        self.max_iterations = env_config.max_iterations

        self.infinite_world = env_config.infinite_world
        self.config = config

        self.obstacles = env_config.obstacles

        self.obstacle_positions = []
        for (x, y) in self.obstacles:
            self.obstacle_positions.append(self.possible_location_values[x])
            self.obstacle_positions.append(self.possible_location_values[y])

        self.agents = []
        self.initial_positions = []

    def add_agent(self, agent: Agent, position=None):
        """
        Args:
            agent:
            position: If None, random position at each new episode.
        """
        assert position is None or (0 <= position[0] < 1 and 0 <= position[1] < 1), "Initial position is incorrect."
        if self.config.env.world_3D:
            assert position is None or len(position) == 3, "Please provide 3D positions if use 3D world."
        if position is not None:
            assert position not in self.obstacles, "Initial position in an obstacle"
        if position is not None:
            x, y = self.possible_location_values[-1], self.possible_location_values[-1]
            z = position[2] if len(position) == 3 else 0
            for k in range(self.board_size):
                if position[0] <= self.possible_location_values[k]:
                    x = self.possible_location_values[k]
                if position[1] <= self.possible_location_values[k]:
                    y = self.possible_location_values[k]
                if len(position) == 2 and position[2] <= self.possible_location_values[k]:
                    z = self.possible_location_values[k]
            position = x, y, z
        self.agents.append(agent)
        self.initial_positions.append(position)

    def _get_random_position(self):
        possible_values = [(x, y) for x in range(self.board_size)
                           for y in range(self.board_size) if (x, y) not in self.obstacles]
        x, y = random.sample(possible_values, 1)[0]
        x, y = self.possible_location_values[x], self.possible_location_values[y]
        z = self.possible_location_values[0]
        if self.config.env.world_3D:
            z = random.sample(self.possible_location_values, 1)[0]
        return x, y, z

    def _get_position_from_action(self, current_position, action):
        """
        From an action number, returns the new position.
        If position is not correct, then the position stays the same.
        Args:
            current_position: (x_cur, y_cur)
            action: in {0, 1, 2, 3, 4}
        Returns: (x_new, y_new)
        """
        index_x = self.possible_location_values.index(current_position[0])
        index_y = self.possible_location_values.index(current_position[1])
        index_z = 0
        if self.config.env.world_3D:
            index_z = self.possible_location_values.index(current_position[2])

        if action == 1:  # Front
            position = index_x, index_y + 1, index_z
        elif action == 2:  # Left
            position = index_x - 1, index_y, index_z
        elif action == 3:  # Back
            position = index_x, index_y - 1, index_z
        elif action == 4:  # Right
            position = index_x + 1, index_y, index_z
        elif self.config.env.world_3D and action == 5:  # Top
            position = index_x, index_y, index_z + 1
        elif self.config.env.world_3D and action == 6:  # Bottom
            position = index_x, index_y, index_z - 1
        else:  # None (=0)
            position = index_x, index_y, index_z
        if not self.infinite_world:
            if position[0] < 0 or position[0] >= len(self.possible_location_values):
                position = index_x, position[1], position[2]
            if position[1] < 0 or position[1] >= len(self.possible_location_values):
                position = position[0], index_y, position[2]
            if position[2] < 0 or position[2] >= len(self.possible_location_values):
                position = position[0], position[1], index_z
        else:
            # If infinite world, goes back to the other side
            position = (position[0] % len(self.possible_location_values),
                        position[1] % len(self.possible_location_values),
                        position[2] % len(self.possible_location_values))
        if [position[0], position[1]] in self.obstacles:
            position = index_x, index_y, index_z

        return (self.possible_location_values[position[0]], self.possible_location_values[position[1]],
                self.possible_location_values[position[2]])

    def _get_state_from_positions(self, positions):
        # return positions
        states = []
        for k in range(len(self.agents)):
            relative_positions = []
            x = positions[3 * k]  # Position of the current agent
            y = positions[3 * k + 1]  # Position of the current agent
            z = positions[3 * k + 2]  # Position of the current agent
            # Compute relative positions
            for i in range(len(self.agents)):
                x_other = positions[3 * i]  # Position of the current agent
                y_other = positions[3 * i + 1]  # Position of the current agent
                z_other = positions[3 * i + 2]  # Position of the current agent
                relative_positions.append(x - x_other)
                relative_positions.append(y - y_other)
                relative_positions.append(z - z_other)
            state = positions[:]
            state.extend(relative_positions)
            state.extend(self.obstacle_positions)
            states.append(state)
        return states

    def _get_possible_positions(self, current_position):
        """
        Return possible positions from the given one
        Args:
            current_position:
        Returns: x_index, y_index of the possible new positions
        """
        index_x = self.possible_location_values.index(current_position[0])
        index_y = self.possible_location_values.index(current_position[1])
        index_z = self.possible_location_values.index(current_position[2])
        max_len = len(self.possible_location_values)
        indexes = [(index_x, index_y, index_z)]
        if (self.infinite_world or index_x > 0) and ((index_x - 1) % max_len, index_y) not in self.obstacles:
            indexes.append(((index_x - 1) % max_len, index_y, index_z))  # Left
        if (self.infinite_world or index_x < len(self.possible_location_values) - 1) and (
                ((index_x + 1) % max_len, index_y) not in self.obstacles):  # Right
            indexes.append(((index_x + 1) % max_len, index_y, index_z))
        if (self.infinite_world or index_y > 0) and (
                (index_x, (index_y - 1) % max_len) not in self.obstacles):  # Back
            indexes.append((index_x, (index_y - 1) % max_len, index_z))
        if (self.infinite_world or index_y < len(self.possible_location_values) - 1) and (
                (index_x, (index_y + 1) % max_len) not in self.obstacles):  # Front
            indexes.append((index_x, (index_y + 1) % max_len, index_z))
        if self.config.env.world_3D:
            if (self.infinite_world or index_z < len(self.possible_location_values) - 1) and (
                    (index_x, index_y) not in self.obstacles):  # Top
                indexes.append((index_x, index_y, (index_z + 1) % max_len))
            if (self.infinite_world or index_z > 0) and (
                    (index_x, index_y) not in self.obstacles):  # Bottom
                indexes.append((index_x, index_y, (index_z - 1) % max_len))
        return indexes

    def reset(self):
        """
        Returns: State for each agent. Size is (number_agents, 4 * number_agents)
            for each agent, the state is the positions [x_1, y_1, x_2, y_2, ..., x_n, y_n]
            concatenated with the relative position with each other agent:
            [x_i-x_1, y_i - y_1, ..., x_i - x_n, y_i - y_n].
        """
        self.current_iteration = 0
        # Get all positions
        absolute_positions = []
        for k in range(len(self.initial_positions)):
            position = self.initial_positions[k]
            if position is None:  # If random position
                position = self._get_random_position()
            # Absolute positions for the state. List of size num_agents * 2.
            absolute_positions.append(position[0])
            absolute_positions.append(position[1])
            absolute_positions.append(position[2])
        # Define the initial states
        return self._get_state_from_positions(absolute_positions)

    def step(self, prev_states, actions):
        """
        Args:
            prev_states: states for each agent. Size (num_agents, 4 * num_agents)
            actions: actions for each agent
        """
        positions = []
        for k in range(len(self.agents)):
            # Retrieve absolute positions
            position = prev_states[0][3 * k], prev_states[0][3 * k + 1], prev_states[0][3 * k + 2]
            new_position = self._get_position_from_action(position, actions[k])
            if random.random() < self.noise:
                index_x, index_y, index_z = random.sample(self._get_possible_positions(position), 1)[0]
                new_position = (self.possible_location_values[index_x], self.possible_location_values[index_y],
                                self.possible_location_values[index_z])
            positions.append(new_position[0])
            positions.append(new_position[1])
            positions.append(new_position[2])
        next_state = self._get_state_from_positions(positions)
        # Determine rewards
        border_positions = [self.possible_location_values[0], self.possible_location_values[-1]]
        rewards = reward_full(positions, self.agents, border_positions, self.current_iteration)
        self.current_iteration += 1
        terminal = False
        if self.current_iteration == self.max_iterations:
            terminal = True
        return next_state, rewards, terminal

    def plot(self, state, rewards, ax):
        # Add obstacles
        tick_labels = np.arange(0, self.board_size)
        ax.set_xticks(self.possible_location_values)
        ax.set_yticks(self.possible_location_values)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
        ax.set_xlim(0, self.possible_location_values[-1])
        ax.set_ylim(0, self.possible_location_values[-1])
        if self.config.env.world_3D:
            ax.set_zticks(self.possible_location_values)
            ax.set_zticklabels(tick_labels)
            ax.set_zlim(0, self.possible_location_values[-1])
        ax.grid(which="major", alpha=0.5)
        for x, y in self.obstacles:
            x, y = self.possible_location_values[x], self.possible_location_values[y]
            side = self.possible_location_values[1]
            block = plt.Rectangle((x - side / 2, y - side / 2), width=side, height=side, linewidth=0, color="black")
            ax.add_patch(block)
            if self.config.env.world_3D:
                art3d.pathpatch_2d_to_3d(block, z=0)

        for k in range(len(self.agents)):
            if self.config.env.world_3D:
                position = state[0][3 * k], state[0][3 * k + 1], state[0][3 * k + 2]
            else:
                position = state[0][3 * k], state[0][3 * k + 1]
            radius = self.config.env.plot_radius_3D if self.config.env.world_3D else self.plot_radius
            self.agents[k].plot(position, rewards[k], radius, ax)

from sim import reward_full
import random


class Env:
    def __init__(self, env_config):
        self.reward_type = env_config.reward_type
        self.noise = env_config.noise
        self.board_size = env_config.board_size

        self.possible_location_values = [k / self.board_size for k in range(self.board_size)]

        self.agents = []
        self.initial_positions = []

    def add_agent(self, agent, position=None):
        """
        Args:
            agent:
            position: If None, random position at each new episode.
        """
        assert position is None or (0 <= position[0] < 1 and 0 <= position[1] < 1), "Initial position is incorrect."
        if position is not None:
            x, y = self.possible_location_values[-1], self.possible_location_values[-1]
            for k in range(self.board_size):
                if position[0] <= self.possible_location_values[k]:
                    x = self.possible_location_values[k]
                if position[1] <= self.possible_location_values[k]:
                    y = self.possible_location_values[k]
            position = x, y
        self.agents.append(agent)
        self.initial_positions.append(position)

    def _get_random_position(self):
        x = random.sample(self.possible_location_values, 1)[0]
        y = random.sample(self.possible_location_values, 1)[0]
        return x, y

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

        if action == 0:  # None
            position = index_x, index_y
        elif action == 1:  # Top
            position = index_x, index_y + 1
        elif action == 2:  # Left
            position = index_x - 1, index_y
        elif action == 3:  # Bottom
            position = index_x, index_y - 1
        else:
            position = index_x + 1, index_y
        if 0 < position[0] or position[0] >= len(self.possible_location_values):
            position = index_x, position[1]
        if 0 < position[1] or position[1] >= len(self.possible_location_values):
            position = position[0], index_y
        return self.possible_location_values[position[0]], self.possible_location_values[position[1]]

    def _get_state_from_positions(self, positions):
        states = []
        for k in range(len(self.agents)):
            relative_positions = []
            x = positions[2 * k]  # Position of the current agent
            y = positions[2 * k + 1]  # Position of the current agent
            # Compute relative positions
            for i in range(len(self.agents)):
                x_other = positions[2 * i]  # Position of the current agent
                y_other = positions[2 * i + 1]  # Position of the current agent
                relative_positions.append(x - x_other)
                relative_positions.append(y - y_other)
            state = positions[:]
            state.extend(relative_positions)
            states.append(state)
        return states

    def reset(self):
        """
        Returns: State for each agent. Size is (number_agents, 4 * number_agents)
            for each agent, the state is the positions [x_1, y_1, x_2, y_2, ..., x_n, y_n]
            concatenated with the relative position with each other agent:
            [x_i-x_1, y_i - y_1, ..., x_i - x_n, y_i - y_n].
        """
        # Get all positions
        absolute_positions = []
        for k in range(len(self.initial_positions)):
            position = self.initial_positions[k]
            if position is None:  # If random position
                position = self._get_random_position()
            # Absolute positions for the state. List of size num_agents * 2.
            absolute_positions.append(position[0])
            absolute_positions.append(position[1])
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
            position = prev_states[0][2 * k], prev_states[0][2 * k + 1]
            new_position = self._get_position_from_action(position, actions[k])
            if random.random() < self.noise:
                new_position = self._get_random_position()
            positions.append(new_position[0])
            positions.append(new_position[1])
        next_state = self._get_state_from_positions(positions)
        reward = None
        terminal = False
        return next_state, reward, terminal

    def plot(self):
        pass

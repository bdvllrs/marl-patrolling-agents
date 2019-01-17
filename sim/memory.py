import numpy as np


class ReplayMemory:
    def __init__(self, size):
        self.size = size
        self.internal_memory = {
            "states": [],
            "next_states": [],
            "actions": [],
            "rewards": []
        }

    def __len__(self):
        return len(self.internal_memory["states"])

    def add(self, state, next_state, action, reward):
        """
        Add a new entry to the memory
        Args:
            state:
            next_state:
            action:
            reward:
        """
        # If too large, we remove the first entries
        if len(self) == self.size:
            self.internal_memory["states"].pop(0)
            self.internal_memory["next_states"].pop(0)
            self.internal_memory["actions"].pop(0)
            self.internal_memory["rewards"].pop(0)
        self.internal_memory["states"].append(state)
        self.internal_memory["next_states"].append(next_state)
        self.internal_memory["actions"].append(action)
        self.internal_memory["rewards"].append(reward)

    def get_batch(self, batch_size, shuffle=True):
        """
        Args:
            batch_size:
            shuffle: If true, returns a random batch in the memory. Defaults to True.

        Returns: (state_batch, next_state_batch, action_batch, reward_batch)
        """
        if len(self) < 10 * batch_size:
            return None
        permutation = np.arange(0, len(self))
        if shuffle:
            np.random.shuffle(permutation)
        batch_mask = permutation[-batch_size:]

        state_batch = np.array([self.internal_memory["states"][k] for k in batch_mask])
        next_state_batch = np.array([self.internal_memory["next_states"][k] for k in batch_mask])
        action_batch = np.array([self.internal_memory["actions"][k] for k in batch_mask])
        reward_batch = np.array([self.internal_memory["rewards"][k] for k in batch_mask])
        return state_batch, next_state_batch, action_batch, reward_batch


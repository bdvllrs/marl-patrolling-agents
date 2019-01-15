
class ReplayMemory:
    def __init__(self, size):
        self.size = size
        self.internal_memory = {}

    def add(self, state, next_state, action, reward):
        pass

    def get_batch(self, batch_size):
        state_batch = None
        next_state_batch = None
        action_batch = None
        reward_batch = None
        return state_batch, next_state_batch, action_batch, reward_batch

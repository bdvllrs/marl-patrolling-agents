
class Env:
    reward_type = "full"
    noise = 0.001

    def __init__(self):
        pass

    def add_agent(self, agent, position):
        pass

    def reset(self):
        state = None
        return state

    def step(self, prev_state, action):
        next_state = None
        reward = None
        terminal = False
        return next_state, reward, terminal

    def plot(self):
        pass

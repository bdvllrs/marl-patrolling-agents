import sim
from utils import sample_batch_history
import matplotlib.pyplot as plt

patrols = [sim.Officer("Officer " + str(k)) for k in range(3)]
target = sim.Target()

env = sim.Env(width=100, height=100)
env.max_iterations = 5

for patrol in patrols:
    env.add_agent(patrol)
env.add_agent(target)

for tries in range(3):
    states = env.reset()
    env.draw_board()
    terminal = False
    while not terminal:
        states, actions, rewards, terminal = env.step()
        env.draw_board()

batch = sample_batch_history(patrols[0], 5)

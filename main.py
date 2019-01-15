import matplotlib.pyplot as plt
import torch
from sim import Env, AgentDQN, ReplayMemory
from utils import Config


plt.ion()

config = Config('config/')

device_type = "cuda" if torch.cuda.is_available() and config.learning.cuda else "cpu"
device = torch.device(device_type)

print("Using", device_type)

number_agents = config.agents.number_predators + config.agents.number_preys
# Definition of the agents
agents = [AgentDQN("predator", "predator-{}".format(k), device, config.agents)
          for k in range(config.agents.number_predators)]
agents += [AgentDQN("prey", "prey-{}".format(k), device, config.agents)
           for k in range(config.agents.number_preys)]

# Definition of the memories and set to device
for agent in agents:
    agent.memory = ReplayMemory(config.replay_memory.size)

env = Env(config.env)

# Add agents to the environment
for agent in agents:
    env.add_agent(agent, position=None)

fig_board = plt.figure(0)
ax_board = fig_board.gca()
plt.show()

for episode in range(config.learning.n_episodes):
    states = env.reset()
    terminal = False
    while not terminal:
        actions = [0 for i in range(number_agents)]  # TODO
        states, next_states, rewards, terminal = env.step(states, actions)
        env.plot(states, ax_board)
        plt.draw()
        plt.pause(0.0001)

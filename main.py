import torch
from sim import Env, AgentDQN, ReplayMemory
from utils import Config

config = Config('config/')

device_type = "cuda" if torch.cuda.is_available() and config.learning.cuda else "cpu"
device = torch.device(device_type)

print("Using", device_type)

# Definition of the agents
agents = [AgentDQN("predator", "predator-{}".format(k), config.agents) for k in range(config.agents.number_predators)]
agents += [AgentDQN("prey", "prey-{}".format(k), config.agents) for k in range(config.agents.number_preys)]

# Definition of the memories and set to device
for agent in agents:
    agent.to(device)
    agent.memory = ReplayMemory(config.replay_memory.size)

env = Env(config.env)

# Add agents to the environment
for agent in agents:
    env.add_agent(agent, position=None)

for episode in range(config.learning.n_episodes):
    states = env.reset()
    print(states)
    break
    terminal = False
    while not terminal:
        actions = None  # TODO
        states, next_states, rewards, terminal = env.step(states, actions)

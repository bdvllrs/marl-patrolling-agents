from sim import Env, AgentDQN, ReplayMemory
from utils import Config

config = Config('config/')

# Definition of the agents
agents = [AgentDQN("predator", "predator-{}".format(k), config.agents) for k in range(config.agents.number_predators)]
agents += [AgentDQN("prey", "prey-{}".format(k), config.agents) for k in range(config.agents.number_preys)]



import os
import shutil
from datetime import datetime
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np
from sim import Env, ReplayMemory
from sim.agents.multiagents import AgentMADDPG
from utils import Config, Metrics, compute_discounted_return, train, test, make_gif

config = Config('config/')

if not config.save_build:
    plt.ion()
else:
    plt.switch_backend('agg')


device_type = "cuda" if torch.cuda.is_available() and config.learning.cuda else "cpu"
device = torch.device(device_type)

if config.save_build:
    name = datetime.today().strftime('%Y-%m-%d %H:%M:%S') if not config.build_name else config.build_name
    root_path = os.path.abspath(config.learning.save_folder + '/' + name)
    model_path = os.path.join(root_path, "models")
    path_figure = os.path.join(root_path, "figs")
    os.makedirs(model_path)
    os.makedirs(path_figure)
    shutil.copytree(os.path.abspath('config/'), os.path.join(root_path, 'config'))

print("Using", device_type)

number_agents = config.agents.number_predators + config.agents.number_preys
# Definition of the agents
agents = [AgentMADDPG("predator", "predator-{}".format(k), device, config.agents)
          for k in range(config.agents.number_predators)]
agents += [AgentMADDPG("prey", "prey-{}".format(k), device, config.agents)
           for k in range(config.agents.number_preys)]

metrics = []
collision_metric = Metrics()
actors_noise = []
# Definition of the memories and set to device
# Define the metrics for all agents
for agent in agents:
    metrics.append(Metrics())

    # If we have to load the pretrained model
    if config.learning.use_model:
        path = os.path.abspath(os.path.join(config.learning.model_path, agent.id + ".pth"))
        agent.load(path)

env = Env(config.env, config)
shared_memory = ReplayMemory(config.replay_memory.size)
# Add agents to the environment
for k in range(len(agents)):
    env.add_agent(agents[k], position=None)
    agents[k].add_agents(agents, k)

fig_board = plt.figure(0, figsize=(10, 10))
if config.env.world_3D:
    ax_board = fig_board.gca(projection="3d")
else:
    ax_board = fig_board.gca()

fig_losses_returns, ((ax_losses, ax_losses_actor), (ax_returns, ax_collisions)) = plt.subplots(2, 2, figsize=(20, 10))

plt.show()

action_dim = 7 if config.env.world_3D else 5
start = time.time()
path_figure_episode = None
progress_bar = None
for episode in range(config.learning.n_episodes):
    if not episode % config.learning.plot_episodes_every:
        if progress_bar is not None:
            progress_bar.close()
        progress_bar = tqdm(total=config.learning.plot_episodes_every)

    # Test step
    if not episode % config.learning.test_every:
        for test_episode in range(config.learning.n_episode_in_test):
            test(env, agents, collision_metric, metrics, config)

    # Plot step
    if not episode % config.learning.plot_episodes_every or not episode % config.learning.save_episodes_every:
        all_states, all_rewards, all_types = test(env, agents, collision_metric, metrics, config)

        # Make path for episode images
        if not episode % config.learning.save_episodes_every and config.save_build:
            path_figure_episode = os.path.join(path_figure, "episode-{}".format(episode))
            os.mkdir(path_figure_episode)

        # Plot last test episode
        for k, (states, rewards, types) in enumerate(zip(all_states, all_rewards, all_types)):
            # Plot environment
            ax_board.cla()
            env.plot(states, types, rewards, ax_board)
            plt.draw()
            if not episode % config.learning.save_episodes_every and config.save_build:
                fig_board.savefig(os.path.join(path_figure_episode, "frame-{}.jpg".format(k)))
                fig_losses_returns.savefig(os.path.join(path_figure, "losses.eps"), dpi=1000, format="eps")
            if not episode % config.learning.plot_episodes_every:
                plt.pause(0.001)

    all_states, all_next_states, all_rewards, all_actions, _ = train(env, agents, shared_memory,
                                                                     metrics, action_dim, config, agents_type="maddpg")

    # Plot learning curves
    if not episode % config.learning.plot_curves_every:
        print("Episode", episode)
        print("Time :", time.time() - start)
        ax_losses.cla()
        ax_returns.cla()
        ax_losses_actor.cla()
        ax_collisions.cla()
        for k in range(len(agents)):
            metrics[k].compute_averages()

            metrics[k].plot_losses(episode, ax_losses, legend=agents[k].id)
            metrics[k].plot_returns(episode, ax_returns, legend=agents[k].id)
            metrics[k].plot_losses_actor(episode, ax_losses_actor, legend=agents[k].id)
            ax_losses.set_title("Losses critic")
            ax_losses.legend()
            ax_returns.set_title("Returns")
            ax_returns.legend()
            ax_losses_actor.set_title("Losses actor")
            ax_losses_actor.legend()
        collision_metric.compute_averages()
        collision_metric.plot_collision_counts(episode, ax_collisions)
        ax_collisions.set_title("Number of collisions")

        plt.draw()
        plt.pause(0.0001)

    # Save models
    if config.save_build:
        for agent in agents:
            path = os.path.join(model_path, agent.id + ".pth")
            agent.save(path)

    progress_bar.update(1)
progress_bar.close()

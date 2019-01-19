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
from sim.agents.agents import AgentDQN
from utils import Config, Metrics, compute_discounted_return

plt.ion()

config = Config('config/')

device_type = "cuda" if torch.cuda.is_available() and config.learning.cuda else "cpu"
device = torch.device(device_type)

print("Using", device_type)

if config.save_build:
    name = datetime.today().strftime('%Y-%m-%d %H:%M:%S') if not config.build_name else config.build_name
    root_path = os.path.abspath(config.learning.save_folder + '/' + name)
    model_path = os.path.join(root_path, "models")
    path_figure = os.path.join(root_path, "figs")
    os.makedirs(model_path)
    os.makedirs(path_figure)
    shutil.copytree(os.path.abspath('config/'), os.path.join(root_path, 'config'))


number_agents = config.agents.number_predators + config.agents.number_preys
# Definition of the agents
agents = [AgentDQN("predator", "predator-{}".format(k), device, config.agents)
          for k in range(config.agents.number_predators)]
agents += [AgentDQN("prey", "prey-{}".format(k), device, config.agents)
           for k in range(config.agents.number_preys)]

metrics = []
collision_metric = Metrics()

# Definition of the memories and set to device
# Define the metrics for all agents
for agent in agents:

    agent.memory = ReplayMemory(config.replay_memory.size)
    metrics.append(Metrics())

    # If we have to load the pretrained model
    if config.learning.use_model:
        path = os.path.abspath(os.path.join(config.learning.model_path, agent.id + ".pth"))
        agent.load(path)

env = Env(config.env, config)

# Add agents to the environment
for agent in agents:
    env.add_agent(agent, position=None)

fig_board = plt.figure(0)
if config.env.world_3D:
    ax_board = fig_board.gca(projection="3d")
else:
    ax_board = fig_board.gca()


fig_losses_returns, (ax_losses, ax_returns, ax_collisions) = plt.subplots(3, 1)

plt.show()

start = time.time()
path_figure_episode = None

progress_bar = None
for episode in range(config.learning.n_episodes):
    if not episode % config.learning.plot_episodes_every:
        if progress_bar is not None:
            progress_bar.close()
        progress_bar = tqdm(total=config.learning.plot_episodes_every)

    test_step = False
    if not episode % config.learning.plot_episodes_every:
        test_step = True
    if not episode % config.learning.save_episodes_every and config.save_build:
        path_figure_episode = os.path.join(path_figure, "episode-{}".format(episode))
        os.mkdir(path_figure_episode)
    all_rewards = []
    states = env.reset()
    terminal = False
    step_k = 0
    while not terminal:
        actions = [agents[i].draw_action(states[i], no_exploration=test_step) for i in range(len(agents))]
        next_states, rewards, terminal, n_collisions = env.step(states, actions)
        all_rewards.append(rewards)

        if not episode % config.learning.plot_episodes_every or not episode % config.learning.save_episodes_every:
            # Plot environment
            ax_board.cla()
            env.plot(next_states, rewards, ax_board)
            plt.draw()
            if not episode % config.learning.save_episodes_every and config.save_build:
                fig_board.savefig(os.path.join(path_figure_episode, "frame-{}.jpg".format(step_k)))
                fig_losses_returns.savefig(os.path.join(path_figure, "losses.eps"), dpi=1000, format="eps")
            if not episode % config.learning.plot_episodes_every:
                plt.pause(0.001)

        step_k += 1

        collision_metric.add_collision_count(n_collisions)
        # Learning Step
        if not test_step:
            for k in range(len(agents)):
                # Add to agent memory
                agents[k].memory.add(states[k], next_states[k], actions[k], rewards[k])
                # Get batch for learning
                batch = agents[k].memory.get_batch(config.learning.batch_size, shuffle=config.replay_memory.shuffle)
                # Learn
                if batch is not None:
                    loss = agents[k].learn(batch)
                    metrics[k].add_loss(loss)

        states = next_states

    # Compute discounted return of the episode
    for k in range(len(agents)):
        reward = [all_rewards[i][k] for i in range(len(all_rewards))]
        discounted_return = compute_discounted_return(config.agents.gamma, reward)
        metrics[k].add_return(discounted_return)

    # Plot learning curves
    if not episode % config.learning.plot_curves_every:
        print("Episode", episode)
        print("Time :", time.time()-start)

        ax_losses.cla()
        ax_returns.cla()
        for k in range(len(agents)):
            # Compute average of losses of all learning step in episode and add it to the list of losses
            metrics[k].compute_averages()

            metrics[k].plot_losses(episode, ax_losses, legend=agents[k].id)
            metrics[k].plot_returns(episode, ax_returns, legend=agents[k].id)
            ax_losses.set_title("Losses")
            ax_losses.legend()
            ax_returns.set_title("Returns")
            ax_returns.legend()
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

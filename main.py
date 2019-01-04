import sim
from model.optimizer import optimize_model
import time
import numpy as np
from utils.utils import draw_result, random_position_around_point

import matplotlib.pyplot as plt

plt.show()

number_officers = 3
reward_type = "full"
width = height = 100
env = sim.Env(width=width, height=height, reward_type=reward_type)
n_episodes = 10000
batch_size = 64
plot_episode_every = 30
env.max_length_episode = 100  # time to go across the board and do a pursuit
print_every = 30
increase_spawn_circle_every = [1]
spawn_distance = [100]
spawn_index = 0
plot_loss = []

officers = [sim.Officer("Officer " + str(k)) for k in range(number_officers)]
target = sim.Target()  # One target

x_target, y_target = np.random.randint(0, width), np.random.randint(0, height)
env.add_agent(target)

for officer in officers:
    position = random_position_around_point((x_target, y_target), spawn_distance[spawn_index], (width, height))
    env.add_agent(officer, position)

start = time.time()

for episode in range(1, n_episodes + 1):
    states = env.reset()
    # Draw the board
    if not episode % plot_episode_every:
        env.draw_board()
    terminal = False
    # Do an episode
    while not terminal:
        states, actions, rewards, terminal = env.step()
        if not episode % plot_episode_every:
            env.draw_board()

    # Learning step
    optimize_model(env, batch_size, episode)

    if episode % print_every == 0:
        print("Episode Num ", episode)
        print("Time : ", time.time() - start)
        draw_result(env)
        print("Save fig")

    # Update positions
    if not episode % increase_spawn_circle_every[spawn_index] and spawn_index < len(spawn_distance) - 1:
        print('Increasing spawn distance to', spawn_distance[spawn_index])
        spawn_index += 1

    x_target, y_target = np.random.randint(0, width), np.random.randint(0, height)
    env.set_position(target, (x_target, y_target))
    for officer in officers:
        position = random_position_around_point((x_target, y_target), spawn_distance[spawn_index], (width, height))
        env.set_position(officer, position)

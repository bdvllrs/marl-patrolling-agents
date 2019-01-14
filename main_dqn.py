import sim
from model.optimizer_dqn import optimize_model
import time
import numpy as np
from utils.utils import draw_result, random_position_around_point
from utils.rewards import compute_discounted_return

import matplotlib.pyplot as plt

plt.show()



number_officers = 3
reward_type = "full"
width = height = 15
env = sim.Env(width=width, height=height, reward_type=reward_type)
n_epochs = 50000
n_episodes = 30
n_learning = 10
batch_size = 128
plot_episode_every = 100

env.max_length_episode = 50  # time to go across the board and do a pursuit
print_every = 10
increase_spawn_circle_every = [1]
spawn_distance = [10]
spawn_index = 0
k_spawn = 0
plot_loss = []

officers = [sim.Officer("Officer " + str(k)) for k in range(number_officers)]
target = sim.Target("Target")  # One target

x_target, y_target = np.random.randint(0, width), np.random.randint(0, height)
env.add_agent(target, (x_target, y_target))

for officer in officers:
    position = random_position_around_point((x_target, y_target), spawn_distance[spawn_index], (width, height))
    env.add_agent(officer, position)

meter = {
    "losses": {agent.name: [] for agent in env.agents if agent.can_learn},
    "returns": {agent.name: [] for agent in env.agents if agent.can_learn},
    "episodes": []
}

start = time.time()

for epoch in range(n_epochs):
    for episode in range(1, n_episodes + 1):
        states = env.reset()
        # Draw the board
        if not epoch % plot_episode_every and episode == 1:
            env.draw_board()
        terminal = False
        all_rewards = []
        # Do an episode
        while not terminal:
            states, actions, rewards, terminal = env.step()
            all_rewards.append(rewards)
            if not epoch % plot_episode_every and episode == 1:
                env.draw_board()
        all_rewards = np.array(all_rewards)

        # Update meter
        if not epoch % print_every:
            meter["episodes"].append(epoch * n_episodes + episode)
            for k, agent in enumerate(env.agents):
                if agent.can_learn:
                    meter["losses"][agent.name].append(np.mean(agent.loss_values))
                    meter["returns"][agent.name].append(compute_discounted_return(all_rewards[:, k], agent.gamma))

    x_target, y_target = np.random.randint(0, width), np.random.randint(0, height)
    env.set_position(target, (x_target, y_target))
    for officer in officers:
        position = random_position_around_point((x_target, y_target), spawn_distance[spawn_index], (width, height))
        env.set_position(officer, position)

    k_spawn += 1

    # Learning step
    for k in range(n_learning):
        optimize_model(env, batch_size, epoch)

    if epoch % print_every == 0:
        print("Episode Num ", epoch)
        print("Time: ", time.time() - start)
        draw_result(meter)

    # Update positions
    if k_spawn == increase_spawn_circle_every[spawn_index] and spawn_index < len(spawn_distance) - 1:
        k_spawn = 0
        spawn_index += 1
        print('Increasing spawn distance to', spawn_distance[spawn_index])

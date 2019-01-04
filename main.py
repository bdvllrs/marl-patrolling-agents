import sim
from model.optimizer import optimize_model
import time
from utils.utils import draw_result

import matplotlib.pyplot as plt

plt.show()

number_officers = 5
reward_type = "full"
env = sim.Env(width=100, height=100, reward_type=reward_type)
n_episodes = 10000
batch_size = 64
plot_episode_every = 1000
env.max_length_episode = 100  # time to go across the board and do a pursuit
print_every = 1
plot_loss = []

officers = [sim.Officer("Officer " + str(k)) for k in range(number_officers)]
target = sim.Target()  # One target


for officer in officers:
    env.add_agent(officer)
env.add_agent(target)

start = time.time()

for episode in range(1, n_episodes):
    states = env.reset()
    # Draw the board
    if episode % plot_episode_every:
        env.draw_board()
    terminal = False
    # Do an episode
    while not terminal:
        states, actions, rewards, terminal = env.step()
        if episode % plot_episode_every:
            env.draw_board()

    # Learning step
    optimize_model(env, batch_size, episode)

    if episode % print_every == 0:
        print("Episode Num ", episode)
        print("Time : ", time.time()-start)
        for agent in env.agents:
            if agent.can_learn:
                draw_result(agent.loss_values, "loss " + str(agent.name))
                draw_result(agent.reward_values, "reward "+ str(agent.name))
                print("Save fig, ", agent.name)

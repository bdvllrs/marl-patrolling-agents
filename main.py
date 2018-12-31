import sim
from model.optimizer import optimize_model
from utils import sample_batch_history


number_officers = 5
reward_type = "full"
env = sim.Env(width=100, height=100, reward_type=reward_type)
n_episodes = 10000
batch_size = 64
plot_episode_every = 1000
env.max_length_episode = 200  # time to go across the board and do a pursuit

officers = [sim.Officer("Officer " + str(k)) for k in range(number_officers)]
target = sim.Target()  # One target

for officer in officers:
    env.add_agent(officer)
env.add_agent(target)

for episode in range(n_episodes):
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




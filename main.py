import sim
from utils import sample_batch_history


number_officers = 5
env = sim.Env(width=100, height=100)
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
    for agent in env.agents:
        if agent.can_learn:
            batch = sample_batch_history(agent, batch_size)
            # Do some teaching

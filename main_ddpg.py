from torch.autograd import Variable

import sim
from model import DDPG
from model.Buffer import ReplayBuffer
from model.optimizer_ddpg import *
import time
import numpy as np
from utils.utils import draw_result, random_position_around_point
from utils.rewards import compute_discounted_return

import matplotlib.pyplot as plt

plt.show()

number_officers = 3
reward_type = "full"
width = height = 100
env = sim.Env(width=width, height=height, reward_type=reward_type)
n_epochs = 50000
n_episodes = 10
n_learning = 10
batch_size = 1024
plot_episode_every = 100
final_noise_scale = 0.0
init_noise_scale = 0.1
buffer_length = 100
env.max_length_episode = 50  # time to go across the board and do a pursuit
print_every = 10
increase_spawn_circle_every = [1]
spawn_distance = [10]
spawn_index = 0
k_spawn = 0
n_rollout_threads = 1
plot_loss = []
USE_CUDA = True
steps_per_update = 100


ddpg = Trainer.init_from_env(env)

replay_buffer = ReplayBuffer(buffer_length, ddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.n for acsp in env.action_space])

officers = [sim.Officer("Officer " + str(k)) for k in range(number_officers)]
target = sim.Target()  # One target

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

trainer = Trainer(100*100, 9, 9, env)

start = time.time()

t = 0

for epoch in range(n_epochs):
    for episode in range(1, n_episodes + 1):
        states = env.reset()

        explr_pct_remaining = max(0, n_episodes - episode) / n_episodes
        ddpg.scale_noise(
            final_noise_scale + (init_noise_scale - final_noise_scale) * explr_pct_remaining)
        ddpg.reset_noise()


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
        # rearrange observations to be per agent, and convert to torch Variable
        torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                              requires_grad=False)
                     for i in range(ddpg.nagents)]
        # get actions as torch Variables
        torch_agent_actions = ddpg.step(torch_obs, explore=True)
        # convert actions to numpy arrays
        agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
        # rearrange actions to be per environment
        actions = [[ac[i] for ac in agent_actions] for i in range(n_rollout_threads)]
        next_obs, rewards, dones, infos = env.step(actions)
        replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
        obs = next_obs
        t += n_rollout_threads
        if (len(replay_buffer) >= batch_size and
                    (t % steps_per_update) < n_rollout_threads):
            if USE_CUDA:
                ddpg.prep_training(device='gpu')
            else:
                ddpg.prep_training(device='cpu')
            for u_i in range(config.n_rollout_threads):
                for a_i in range(ddpg.nagents):
                    sample = replay_buffer.sample(config.batch_size,
                                                  to_gpu=USE_CUDA)
                    ddpg.update(sample, a_i, logger=logger)
                ddpg.update_all_targets()
            ddpg.prep_rollouts(device='cpu')
    ep_rews = replay_buffer.get_average_rewards(
        n_learning * n_rollout_threads)
    for a_i, a_ep_rew in enumerate(ep_rews):
        logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

    if ep_i % config.save_interval < config.n_rollout_threads:
        os.makedirs(run_dir / 'incremental', exist_ok=True)
        ddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
        ddpg.save(run_dir / 'model.pt')






    if epoch % print_every == 0:
        print("Episode Num ", epoch)
        print("Time: ", time.time() - start)
        draw_result(meter)

    # Update positions
    if k_spawn == increase_spawn_circle_every[spawn_index] and spawn_index < len(spawn_distance) - 1:
        k_spawn = 0
        spawn_index += 1
        print('Increasing spawn distance to', spawn_distance[spawn_index])

ddpg.save(run_dir / 'model.pt')
env.close()
logger.export_scalars_to_json(str(log_dir / 'summary.json'))
logger.close()

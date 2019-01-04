from utils import sample_batch_history
import torch
import torch.nn.functional as F

# if gpu is to be used

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def optimize_model(env, batch_size, episode):
    for agent in env.agents:
        if agent.can_learn:
            batch = sample_batch_history(agent, batch_size)

            if batch is None:
                # Not enough data in histories
                return

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch["next_states"])), device=device, dtype=torch.uint8)
            h = env.height
            BNS = [s for s in batch["next_states"] if s is not None]
            non_final_next_states = torch.FloatTensor(BNS, device=device).reshape(len(BNS), h * h)

            state_batch = torch.FloatTensor(batch["states"], device=device)
            action_batch = torch.FloatTensor(batch["actions"], device=device).max(1)[1].unsqueeze(1)
            reward_batch = torch.FloatTensor(batch["rewards"], device=device)
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_batch = state_batch.reshape(batch_size, h * h).cuda()
            state_action_values = agent.policy_net(state_batch).gather(1, action_batch.long().cuda())
            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(batch_size, device=device)
            next_state_values[non_final_mask] = agent.target_net(non_final_next_states.cuda()).max(1)[0].detach()
            # Compute the expected Q values
            expected_state_action_values = (next_state_values.cuda() * agent.gamma) + reward_batch.cuda()

            # Compute Huber loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
            agent.loss_values.append(loss.item())
            agent.reward_values.append(reward_batch.mean())
            # Optimize the model
            agent.optimizer.zero_grad()
            loss.backward()
            for param in agent.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            agent.optimizer.step()

            if episode % 100 == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

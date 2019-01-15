from utils import sample_batch_history
import torch
import torch.nn.functional as F

# if gpu is to be used

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)

double = True


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
            # non_final_next_states = torch.FloatTensor(BNS, device=device).reshape(len(BNS), h, h)
            non_final_next_states = torch.FloatTensor(BNS, device=device)

            state_batch = torch.FloatTensor(batch["states"], device=device)
            action_batch = torch.FloatTensor(batch["actions"], device=device).max(1)[1].unsqueeze(1)
            reward_batch = torch.FloatTensor(batch["rewards"], device=device)
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            # state_batch = state_batch.reshape(batch_size, h, h).to(device)
            state_batch = state_batch.to(device)

            state_action_values = agent.policy_net(state_batch).gather(1, action_batch.long().to(device))
            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(batch_size, device=device)
            if not double:
                next_state_values[non_final_mask] = agent.target_net(non_final_next_states.to(device)).max(1)[
                    0].detach()
            else:
                Qsa_prime_actions = agent.policy_net(non_final_next_states.to(device)).max(1)[1].detach()
                test = agent.target_net(non_final_next_states.to(device))
                test2 = test.gather(1, Qsa_prime_actions.view(-1, 1))
                next_state_values[non_final_mask] = test2.squeeze(1)

            # Compute the expected Q values
            expected_state_action_values = (next_state_values.to(device) * agent.gamma) + reward_batch.to(device)

            agent.optimizer.zero_grad()
            # Compute Huber loss
            loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
            loss.backward()
            agent.optimizer.step()

            soft_update(agent.policy_net, agent.target_net)

            agent.loss_values.append(loss.item())
            agent.reward_values.append(reward_batch.mean())
            # Optimize the model


def soft_update(local_model, target_model, tau=2e-3):
    """
    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

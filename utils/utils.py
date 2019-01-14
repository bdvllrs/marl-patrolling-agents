import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import shutil
import os
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable


plt.ion()


def choice(l):
    """
    Returns a random element from the list
    """
    item = np.random.randint(0, len(l))
    return l[item]


def random_position_around_point(ref_point, max_distance, limits):
    """
    Gives a random position at in a circle around position
    Args:
        ref_point: central position
        max_distance: maximum distance from position
        limits: limits of the board
    """
    possible_positions = []
    x, y = ref_point
    for i in range(-max_distance, max_distance + 1):
        for j in range(-max_distance, max_distance + 1):
            if 0 <= x + i < limits[0] and 0 <= y + j < limits[1]:
                possible_positions.append((x + i, y + j))
    return choice(possible_positions)


def possible_directions(limit_board, position):
    """
    Gives the possible moves allowed for the agent
    Args:
        limit_board: limits of the board
        position: position of the agent
    Returns: list of possible positions
    """
    lim_x, lim_y = limit_board
    x, y = position
    possible_direction = ['none']
    if x > 0:
        possible_direction.append('left')
        if y > 0:
            possible_direction.append('bottom-left')
        if y < lim_y - 1:
            possible_direction.append('top-left')
    if y > 0:
        possible_direction.append('bottom')
    if x < lim_x - 1:
        possible_direction.append('right')
        if y > 0:
            possible_direction.append('bottom-right')
        if y < lim_y - 1:
            possible_direction.append('top-right')
    if y < lim_y - 1:
        possible_direction.append('top')
    return possible_direction


def position_from_direction(current_position, direction):
    """
    Gives position on the board from a direction
    Args:
        current_position: Position of the agent (x_cur, y_cur)
        direction: one of the 8 possible directions (none, top, left, ...)
    Returns: couple of position (x_new, y_new)
    """
    assert direction in ['none', 'top', 'left', 'right', 'bottom', 'top-left', 'top-right', 'bottom-right',
                         'bottom-left'], "The direction is unknown."
    x, y = current_position
    if direction == 'top':
        return x, y + 1
    elif direction == 'bottom':
        return x, y - 1
    elif direction == 'left':
        return x - 1, y
    elif direction == 'right':
        return x + 1, y
    elif direction == 'top-left':
        return x - 1, y + 1
    elif direction == 'top-right':
        return x + 1, y + 1
    elif direction == 'bottom-left':
        return x - 1, y - 1
    elif direction == 'bottom-right':
        return x + 1, y - 1
    else:  # none
        return x, y


def get_distance_between(limit_board, position_1, position_2):
    """
    Returns the distance between the two positions
    Args:
        limit_board: limits of the board
        position_1: position 1
        position_2: position 2
    """
    positions = [position_from_direction(position_2, direction) for direction in
                 possible_directions(limit_board, position_2)]
    if position_1 in positions:
        return 1
    x, y = position_1
    x_2, y_2 = position_2
    x_new, y_new = position_1
    if x < x_2:
        if y < y_2:
            x_new, y_new = x + 1, y + 1
        elif y == y_2:
            x_new = x + 1
        else:
            x_new, y_new = x + 1, y - 1
    elif x == x_2:
        if y < y_2:
            y_new = y + 1
        elif y > y_2:
            y_new = y - 1
    else:
        if y < y_2:
            x_new, y_new = x - 1, y + 1
        elif y == y_2:
            x_new = x - 1
        else:
            x_new, y_new = x - 1, y - 1
    return 1 + get_distance_between(limit_board, (x_new, y_new), position_2)


def distance_enemies_around(agent, agents, max_distance=None):
    """
    Returns the distance to all enemies around one agent
    Args:
        agent: reference agent
        agents: other agents
        max_distance: max distance. Default agent's field of view
    Returns: distances
    """
    max_distance = agent.view_radius if max_distance is None else max_distance
    enemies_around = []
    for other_agent in agents:
        if ((agent.type == "target" and other_agent.type == 'officer') or
                (agent.type == "officer" and other_agent.type == 'target')):
            dist_agents = get_distance_between(agent.limit_board, agent.position, other_agent.position)
            if dist_agents <= max_distance:
                enemies_around.append(dist_agents)
    return enemies_around


def state_from_observation(agent, position, observation):
    """
    Get state board from observation.
    It is a board of the field of view of the agent where:
    - -1 itself
    - 0.5 is friend (as well as itself, which is always in the center)
    - 1 is enemy
    Args:
        position: position of the agent
        agent: Agent receiving the observation
        observation:
    Returns: array of shape (2 * agent.view_radius + 1, 2 * agent.view_radius + 1)
    """
    side_length = agent.limit_board
    # Defaults to zero (not accessible)
    board = [[0 for i in range(side_length[0])] for j in range(side_length[1])]
    x_a, y_a = position
    # Set to zero every position in field of view
    for obs in observation:
        x, y = obs.position
        if 0 <= x < side_length[0] and 0 <= y < side_length[1]:
            if obs.type == agent.type:
                board[y][x] = 0.5
            else:
                board[y][x] = 1
    if 0 <= x_a < side_length[0] and 0 <= y_a < side_length[1]:
        board[y_a][x_a] = -1
    return np.array(board)


def sample_batch_history(agent, batch_size, memory=10000):
    """
    Samples the history of the given agent.
    Args:
        agent: agent to sample
        batch_size:
        memory: how far we can go back (max) in the history
    Returns: None or dictionary of keys "states", "next_states", "actions" and "rewards" and values a list of the values in
        the history. *states* corresponds to prev_position, *next_states* to position, *actions* to action and
        *rewards* to reward. position is set to None if it is the last position (terminal).
        Returns None if there are not enough elements in the history to fill a full batch.
    """
    actions_to_onehot = {
        "none": [1, 0, 0, 0, 0, 0, 0, 0, 0],
        "top": [0, 1, 0, 0, 0, 0, 0, 0, 0],
        "bottom": [0, 0, 1, 0, 0, 0, 0, 0, 0],
        "left": [0, 0, 0, 1, 0, 0, 0, 0, 0],
        "right": [0, 0, 0, 0, 1, 0, 0, 0, 0]
        #"top-right": [0, 0, 0, 0, 0, 1, 0, 0, 0],
        #"bottom-right": [0, 0, 0, 0, 0, 0, 1, 0, 0],
        #"top-left": [0, 0, 0, 0, 0, 0, 0, 1, 0],
        #"bottom-left": [0, 0, 0, 0, 0, 0, 0, 0, 1],
    }
    # remove first position and keep only memory elements
    history = list(filter(lambda x: "prev_state" in x.keys(), agent.histories))[-memory:]
    if len(history) >= 10*batch_size:
        batch = random.sample(history, batch_size)
        # Transforms into a convenient form
        return {
            "states": list(map(lambda x: x["prev_state"], batch)),
            "next_states": list(map(lambda x: x["state"] if not x["terminal"] else None, batch)),
            "actions": list(map(lambda x: actions_to_onehot[x["action"]], batch)),
            "rewards": list(map(lambda x: x["reward"], batch)),
        }
    return None


def draw_result(meter):
    plt.figure(1)
    plt.clf()
    plt.subplot('121')
    for _, losses in sorted(meter["losses"].items()):
        plt.plot(meter["episodes"], losses)
    plt.title('Losses')
    plt.subplot('122')
    for _, returns in sorted(meter["returns"].items()):
        plt.plot(meter["episodes"], returns)
    plt.title('Returns')

    plt.xlabel("n iteration")
    plt.legend([name for name, _ in sorted(meter['losses'].items())], loc='upper left')
    plt.savefig("fig/losses.eps", format="eps", dpi=1000)  # should before show method
    plt.draw()
    plt.pause(0.0001)
    # save image

def soft_update(target, source, tau):
    """
    Copies the parameters from source network (x) to target network (y) using the below update
    y = TAU*x + (1 - TAU)*y
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    """
    Copies the parameters from source network to target network
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


def save_training_checkpoint(state, is_best, episode_count):
    """
    Saves the models, with all training parameters intact
    :param state:
    :param is_best:
    :param filename:
    :return:
    """
    filename = str(episode_count) + 'checkpoint.path.rar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')



# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size

# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y

import numpy as np


def thompson_sampling(length, agent):
    """
    Generalized Thompson Sampling. Inspired by Agrawal and Goyal, 2012.
    """
    S = np.zeros(agent.number_arms, dtype=float)
    F = np.zeros(agent.number_arms, dtype=float)
    rewards = np.zeros(length)
    draws = np.zeros(length, dtype=int)
    for k in range(length):
        theta = np.random.beta(S + 1, F + 1)
        draws[k] = np.argmax(theta)
        reward = float(MAB[draws[k]].sample())
        # rewards[k] = reward
        bernoulli_trial = float(np.random.rand() < reward)  # Generalized version
        rewards[k] = bernoulli_trial
        S[draws[k]] += bernoulli_trial
        F[draws[k]] += 1 - bernoulli_trial
    return rewards, draws

import numpy as np
from collections import deque
import random
from typing import Optional
import torch as th
from scipy import stats

# base from https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/utils.py
def quantile_huber_loss(
    current_quantiles: th.Tensor,
    target_quantiles: th.Tensor,
    cum_prob: Optional[th.Tensor] = None,
    sum_over_quantiles: bool = True,
) -> th.Tensor:
    """
    The quantile-regression loss, as described in the QR-DQN and TQC papers.
    Partially taken from https://github.com/bayesgroup/tqc_pytorch.

    :param current_quantiles: current estimate of quantiles, must be either
        (batch_size, n_quantiles) or (batch_size, n_critics, n_quantiles)
    :param target_quantiles: target of quantiles, must be either (batch_size, n_target_quantiles),
        (batch_size, 1, n_target_quantiles), or (batch_size, n_critics, n_target_quantiles)
    :param cum_prob: cumulative probabilities to calculate quantiles (also called midpoints in QR-DQN paper),
        must be either (batch_size, n_quantiles), (batch_size, 1, n_quantiles), or (batch_size, n_critics, n_quantiles).
        (if None, calculating unit quantiles)
    :param sum_over_quantiles: if summing over the quantile dimension or not
    :return: the loss
    """
    if current_quantiles.ndim != target_quantiles.ndim:
        raise ValueError(
            f"Error: The dimension of curremt_quantile ({current_quantiles.ndim}) needs to match "
            f"the dimension of target_quantiles ({target_quantiles.ndim})."
        )
    if current_quantiles.shape[0] != target_quantiles.shape[0]:
        raise ValueError(
            f"Error: The batch size of curremt_quantile ({current_quantiles.shape[0]}) needs to match "
            f"the batch size of target_quantiles ({target_quantiles.shape[0]})."
        )
    if current_quantiles.ndim  != 2:
        raise ValueError(f"Error: The dimension of current_quantiles ({current_quantiles.ndim}) needs to be 2.")

    if cum_prob is None:
        n_quantiles = current_quantiles.shape[-1]
        # Cumulative probabilities to calculate quantiles.
        cum_prob = (th.arange(n_quantiles, device=current_quantiles.device, dtype=th.float) + 0.5) / n_quantiles
        if current_quantiles.ndim == 2:
            # For QR-DQN, current_quantiles have a shape (batch_size, n_quantiles), and make cum_prob
            # broadcastable to (batch_size, n_quantiles, n_target_quantiles)
            cum_prob = cum_prob.view(1, -1, 1)
    else:
        cum_prob = cum_prob.view(1, -1, 1)
            
    pairwise_delta = target_quantiles.unsqueeze(-2) - current_quantiles.unsqueeze(-1)
    abs_pairwise_delta = th.abs(pairwise_delta)
    huber_loss = th.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5)
    loss = th.abs(cum_prob - (pairwise_delta.detach() < 0).float()) * huber_loss
    if sum_over_quantiles:
        loss = loss.sum(dim=-2).mean()
    else:
        loss = loss.mean()
    return loss




# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.15, min_sigma=0, decay_period=500):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        if hasattr(action_space, "shape"):
            self.action_dim   = action_space.shape[0]
        else:
            self.action_dim   = action_space
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        self.step = 0
        
    def get_noise(self):
        if self.step > self.decay_period:
            print('Warning: current step > decay_period in OUNoise noise generator.')
        
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, self.step / self.decay_period)
        self.step += 1
        return self.state
    
    
class metaNoise:
    def __init__(self, noise_list):
        self.noise_list = noise_list
    def reset(self):
        for n in self.noise_list:
            n.reset()
    def get_noise(self):
        noise = []
        for n in self.noise_list:
            noise.append(n.get_noise())
        n_len = np.array([len(x) for x in noise])
        max_len = np.max(n_len)
        out_noise = np.zeros(max_len)
        for n in noise:
            out_noise += n
        return out_noise
    
class noNoise:
    def __init__(self):
        pass
    def reset(self):
        pass
    def get_noise(self):
        return [0]


# base from https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, done):
        experience = (state, action, np.array([reward]), done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, done_batch

    def __len__(self):
        return len(self.buffer)
    
   
    
class Memory_constriant:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, constraint):
        
        constraint = np.array(constraint) if hasattr(constraint, "__len__") else np.array([constraint])
        experience = (state, action, np.array([reward]), constraint)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        constraint_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, constraint = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            constraint_batch.append(constraint)
        
        return state_batch, action_batch, reward_batch, constraint_batch

    def __len__(self):
        return len(self.buffer)
    

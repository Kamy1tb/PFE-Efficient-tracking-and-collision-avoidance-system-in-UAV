__version__ = '1.0'
__author__ = 'Miloud Bagaa'
__author_emails__ = 'miloud.bagaa@uqtr.ca, bagmoul@gmail.com'

import random
import numpy as np
from collections import namedtuple
import torch.nn.functional as F


class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.next_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = random.sample(range(max_mem), batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, next_states, dones


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = batch_size if batch_size < len(self.memory) else len(self.memory)
        return random.sample(self.memory, batch)

    def __len__(self):
        return len(self.memory)

def activation_function(x=None, activation_name=None):
    # Other functions can be added later. 
    if activation_name == "relu":
        return F.relu(x)
    if activation_name == "logsigmoid":
        return F.logsigmoid(x)
    if activation_name == "logsigmoid":
        return F.logsigmoid(x)
    if activation_name == "tanhshrink":
        return F.tanhshrink(x)
    if activation_name == "softsign":
        return F.softsign(x)
    if activation_name == "softplus":
        return F.softplus(x)
    if activation_name == "softmin":
        return F.softmin(x)
    if activation_name == "softmax":
        return F.softmax(x)
    if activation_name == "tanh":
        return F.tanh(x)
    if activation_name == "sigmoid":
        return F.sigmoid(x)
    if activation_name == "hardsigmoid":
        return F.hardsigmoid(x)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.4):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * (np.random.rand(*x.shape) - 0.5)
        self.state = x + dx
        return self.state

class OuActionNoise():
    def __init__(self, mu=0, sigma=0.2, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
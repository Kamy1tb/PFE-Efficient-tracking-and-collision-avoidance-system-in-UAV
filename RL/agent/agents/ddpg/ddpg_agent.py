#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# This script is licensed under GNU GPL version 2.0 or above
# (c) 2021 Sihem Ouahouah & Miloud Bagaa

__version__ = '1.0'
__authors__ = 'Sihem Ouahouah & Miloud Bagaa'
__author_emails__ = 'sihem.ouahouah@aalto.fi & miloud.bagaa@aalto.fi'

import sys, os
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agent.helpers.rl_agent_template import *
from agent.helpers.agent_helpers import ReplayBuffer, OuActionNoise
from agent.agents.ddpg.model import ActorNetwork, CriticNetwork

NUM_AGENTS = 20  # How many agents are there in the environment
random_seed = 2

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma=0.9, 
                 max_size=1000000, fc1_dims=400, fc2_dims=300, fc3_dims=100, batch_size=64, fc_name=""):
        self.eval_mode = False
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.fc_name = fc_name

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.noise = OuActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims, fc3_dims,
                                  n_actions=n_actions, name='actor' + self.fc_name)

        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions=n_actions, name='critic' + self.fc_name)

        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims, fc3_dims,
                                  n_actions=n_actions, name='target_actor' + self.fc_name)

        self.target_actor.eval()

        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions=n_actions, name='target_critic' + self.fc_name)

        self.target_critic.eval()

        self.update_network_parameters(tau=self.tau)

    def choose_action(self, observation):
        #Due to the use of L2 Norm to prevent saving the statistics
        self.actor.eval()
        state = T.tensor(np.array([observation]), dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        if self.eval_mode:
            mu_prime = mu
        else:
            mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
            self.actor.train()

        return mu_prime.cpu().detach().numpy()[0]

    def set_eval_mode(self):
        self.eval_mode = True
        self.target_critic.eval()
        self.target_actor.eval()
        self.actor.eval()
        self.critic.eval()

    def train_mode(self):
        self.eval_mode = False
        self.target_critic.eval()
        self.target_actor.eval()
        self.critic.train()
        self.actor.train()

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
        
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, next_states, dones = \
            self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        next_states = T.tensor(next_states, dtype=T.float).to(self.actor.device)
        dones = T.tensor(dones).to(self.actor.device)

        critic_value = self.critic.forward(states, actions)

        target_actions = self.target_actor.forward(next_states)
        critic_value_next = self.target_critic.forward(next_states, target_actions)

        critic_value_next[dones] = 0.0
        #view used instead of shape to ensure that tensor is duplicated. So, self.target_critic will be not affected.
        critic_value_next = critic_value_next.view(-1)

        target_critic_value = rewards + self.gamma * critic_value_next
        target_critic_value = target_critic_value.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target_critic_value, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        #It works as we did not run self.critic.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):

        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_state_dict = dict(target_critic_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                      (1 - tau) * target_critic_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                      (1 - tau) * target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

        #In case batch normalization is used to load mu and segma
        #self.target_critic.load_state_dict(critic_state_dict, strict=False)
        #self.target_actor.load_state_dict(actor_state_dict, strict=False)

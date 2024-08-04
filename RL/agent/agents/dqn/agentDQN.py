__version__ = '1.0'
__author__ = 'Miloud Bagaa'
__author_emails__ = 'miloud.bagaa@uqtr.ca, bagmoul@gmail.com'

import numpy as np
from collections import namedtuple
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from agent.helpers.agent_helpers import Transition, ReplayMemory
from agent.agents.dqn.model import MODELDQN
#from environment.environment_settings import *
from agent.helpers.rl_agent_template import *

class Agent(RLAgent):
    def __init__(self):
        self.number_episodes = 0

    def configure(self, params=None):
        self.eval_mode = False
        self.number_episodes = 0
        self.lr = params["lr"]
        self.epsilon = params["eps_start"]
        self.eps_end = params["eps_end"]
        self.eps_dec = params["eps_dec"]
        self.action_space = params["action_space"]
        self.state_space_dim = len(params["state_space"])
        self.n_actions = len(self.action_space)
        self.hidden_size = params["hidden_size"]
        self.replay_buffer_size = params["replay_buffer_size"]
        self.batch_size = params["batch_size"]
        self.target_update = params["target_update"]
        self.gamma = params["gamma"]
        self.network_spec = params["network_spec"]
        self.memory = ReplayMemory(self.replay_buffer_size)
        self.policy_net =  MODELDQN(lr=self.lr, state_space_dim=self.state_space_dim, action_space_dim=self.n_actions, network_spec=self.network_spec)
        self.target_net =  MODELDQN(lr=self.lr, state_space_dim=self.state_space_dim, action_space_dim=self.n_actions, network_spec=self.network_spec)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def step_learn(self, updates=1):
        for _ in range(updates):
            self._do_network_update()

    def episode_learn(self):
        self.number_episodes += 1
        if  self.number_episodes % self.target_update == 0:
            self.update_target_network()

    def _do_network_update(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = 1-T.tensor(batch.done, dtype=T.uint8)
        non_final_mask = non_final_mask.type(T.bool)
        non_final_next_states = [s for nonfinal,s in zip(non_final_mask,
                                     batch.next_state) if nonfinal > 0]
        non_final_next_states = T.stack(non_final_next_states).to(self.policy_net.device)
        state_batch = T.stack(batch.state).to(self.policy_net.device)
        action_batch = T.cat(batch.action).to(self.policy_net.device)
        reward_batch = T.cat(batch.reward).to(self.policy_net.device)
        # print("state_batch = {}".format(state_batch))
        # print("self.policy_net(state_batch) = {}".format(self.policy_net(state_batch)))
        # print("action_batch = {}".format(action_batch))
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = T.zeros(self.batch_size).to(self.policy_net.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch 


        loss = F.mse_loss(state_action_values.squeeze(), expected_state_action_values).to(self.policy_net.device)

        self.policy_net.optimizer.zero_grad()
        loss.backward()
        
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1e-1, 1e-1)

        self.policy_net.optimizer.step()

    def update_learn_params(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_end else self.eps_end

    def choose_action(self, state):
        if self.eval_mode:
            with T.no_grad():
                state = T.tensor(state, dtype=T.float).to(self.policy_net.device)
                q_values = self.policy_net(state)
                action = T.argmax(q_values).item()
                #print("action = {}".format(action))
                return action

        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            with T.no_grad():
                state = T.tensor(state, dtype=T.float).to(self.policy_net.device)
                q_values = self.policy_net(state)
                return T.argmax(q_values).item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, state=None, action=None, next_state=None, reward=None, done=None):
        if state != None and action != None and next_state != None and reward != None and done != None:
            state = T.tensor(state, dtype=T.float).to(self.policy_net.device)
            next_state = T.tensor(next_state, dtype=T.float).to(self.policy_net.device)
            action = T.tensor([[action]]).long().to(self.policy_net.device)
            reward = T.tensor([reward], dtype=T.float).to(self.policy_net.device)
            self.memory.push(state, action, next_state, reward, done)

    def save_output(self, path=""):
        print(self.memory.memory)

    def save_model(self, path=""):
        print(" ... saving checkpoint ...")
        T.save(self.policy_net.state_dict(), path)

    def load_model(self, path=""):
        print(" ... loading checkpoint ...")
        self.policy_net.load_state_dict(T.load(path))


    def eval(self):
        self.eval_mode = True
        self.policy_net.eval()

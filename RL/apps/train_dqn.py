
__version__ = '1.0'
__authors__ = 'Miloud Bagaa'
__author_emails__ = 'miloud.bagaa@uqtr.ca, bagmoul@gmail.com'

import sys, os
from unipath import Path

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   
sys.path.append(BASE_PATH)
sys.path.append(Path(BASE_PATH).parent)
import json
import torch
from agent.agent import Agent
from agent.helpers.rl_agent_template import *
from env.environment import *
from helpers.utils import *

def main():

    n_games = 5000                   
    complete_check = False
    eps_dec = 1./n_games                 
    environment = Environment()

    # Number of layers and activation functions.
    network_spec = [
        dict(type='dense', size=500, activation='relu'),  # 256         
        dict(type='dense', size=300, activation='relu'),  # 128
        dict(type='dense', size=100, activation='relu')    # 64
    ]

    params = {
            "lr": 1e-2,
            "gamma": 0.9,
            "action_space": environment.get_actions(),
            "state_space": environment.get_states(),
            "eps_start": 1.0,
            "eps_end": 0.01,
            "eps_dec": eps_dec,
            "replay_buffer_size": 50000,
            "batch_size": 256,
            "hidden_size": 40,
            "network_spec": network_spec,
            "target_update": 8,
            }


    agent = Agent("DQN")

    scores = []

    for episode in range(n_games):
        state = environment.reset()
        done = False
        score = 0
    
        while not done:
            actions = agent.choose_action(state)
            next_state, reward, done, info = environment.step(action)
            agent.store_transition(state, action[i], next_state, reward, done)   
            agent.step_learn() 
            state = next_state
            score += reward   

        if episode % 2 == 0:
            print('episode ', episode, 'score %.1f' % score, 'Final_soc %.1f' % Reward_Final_soc , 'cycle %.1f' %  environment.cycle)
        
        agent.episode_learn()
        agent.update_learn_params()
        scores.append(score)


    fname = 'DQN_' + 'Reward' +\
        '_' + str(n_games) + 'games'

    figure_file  = 'RL/output/plots/'  + fname  + '.png'

    x = [i+1 for i in range(n_games)]

    with open("RL/output/log/scores", "w") as f:
         f.write(str(scores))

    plot_learning_curve(x, scores, figure_file)

if __name__ == '__main__':
    main()

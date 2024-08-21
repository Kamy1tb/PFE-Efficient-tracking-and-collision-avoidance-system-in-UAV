
__version__ = '1.0'
__authors__ = 'Miloud Bagaa'
__author_emails__ = 'miloud.bagaa@uqtr.ca, bagmoul@gmail.com'

import sys, os
# Add the 'env' directory to the Python path
from unipath import Path

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   
sys.path.append(BASE_PATH)
sys.path.append(Path(BASE_PATH).parent)
import json
import torch
from agent.agent import Agent
from agent.helpers.rl_agent_template import *
from env.environment import Environment
from helpers.utils import *
import threading


velocity = 1  # Vitesse en m/s
duration = 1  # Durée de chaque mouvement en secondes
waypoint_distance = 10  # Distance entre chaque waypoint en mètres
threads = []

waypoints_track = [
    (waypoint_distance, 0, 0),  # Premier waypoint (droit)
    (waypoint_distance, 2, 0),  # Deuxième waypoint (esquive à gauche)
    (2 * waypoint_distance, 2, 0),  # Troisième waypoint (droit)
    (2 * waypoint_distance, -4, 0),  # Quatrième waypoint (esquive à droite)
    (7 * waypoint_distance, 0, 0),  # cinquième waypoint (droit)
    (waypoint_distance, 2, 0),  # Deuxième waypoint (esquive à gauche)
    (2 * waypoint_distance, 2, 0),  # Troisième waypoint (droit)
    (2 * waypoint_distance, -4, 0),  # Quatrième waypoint (esquive à droite)
    (7 * waypoint_distance, 0, 0)  # cinquième waypoint (droit)
    ]


def control_drone_target(client, waypoints,duration):
        client.drone.takeoff(-4,True)
        for waypoint in waypoints:
            x, y, z = waypoint
            client.drone.move_by_velocity(x, y, z, duration,True)


def main():

    n_games = 5                   
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
            action = agent.choose_action(state)
            next_state, reward, done, info = environment.step(action)
            agent.store_transition(state,action, next_state, reward, done)   
            agent.step_learn() 
            state = next_state
            score += reward   

        if episode % 2 == 0:
            print('episode ', episode, 'score %.1f' % score)
        
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

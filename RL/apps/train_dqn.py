
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
from env.DroneClass import AirSimClientDrone
from helpers.utils import *
import threading
import time

velocity = 1  # Vitesse en m/s
duration = 2  # Durée de chaque mouvement en secondes
waypoint_distance = 3  # Distance entre chaque waypoint en mètres
threads = []

waypoints_track = [
    (waypoint_distance, 0, 0),  # Premier waypoint (droit)
    (waypoint_distance, 2, 0),  # Deuxième waypoint (esquive à gauche)
    (2 * waypoint_distance, 0, 0),  # Troisième waypoint (droit)
    (2 * waypoint_distance, -4, 0),  # Quatrième waypoint (esquive à droite)
    (4 * waypoint_distance, 0, 0),  # cinquième waypoint (droit)
    (waypoint_distance, 2, 0),  # Deuxième waypoint (esquive à gauche)
    (2 * waypoint_distance, 0, 0),  # Troisième waypoint (droit)
    (2 * waypoint_distance, -4, 0),  # Quatrième waypoint (esquive à droite)
    (3 * waypoint_distance, 0, 0),  # cinquième waypoint (droit)
    (0,0,0)

    ]


def control_drone_target(env, waypoints,duration):
        for waypoint in waypoints:
            print("drone 2 ",waypoint)
            x, y, z = waypoint
            if env.drone_target.detect_collision():
                print("collision detected in drone target")
                break
            env.drone_target.move_by_velocity(x, y, z, duration,True)    

        env.set_done()
        

def train_drone(agent,env,state,score):
    while not env.is_done():
        action = agent.choose_action(state)
        print("action ",action)
        next_state, reward, env.done, info = env.step(action)
        print("next state ",next_state)
        agent.store_transition(state,action, next_state, reward, env.done)   
        agent.step_learn() 
        state = next_state
        score[0] += reward

def main():

    n_games = 10000        
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
    agent.configure(params=params)
    agent.load_model("./output/models/best_model.pth")

    scores = []
    best_score = -np.inf
    avg_score = -np.inf
    for episode in range(n_games):
        print("episode ", episode)
        environment.done = 0  
        state = environment.reset()
        environment.drone.takeoff(-4, True)
        environment.drone_target.takeoff(-4, True)

        environment.drone.move_by_velocity(1, 0, 0, 2, False)
        
        score = [0]
        print("before threads ",environment.done)
        # Create a new thread for drone control
        drone_thread = threading.Thread(target=control_drone_target, args=(environment, waypoints_track, duration))
        train_thread = threading.Thread(target=train_drone, args=(agent,environment,state,score))  
        drone_thread.start()
        train_thread.start()
        drone_thread.join()
        print("after thread target ",environment.done)
        train_thread.join()
        print("after thread drone ",environment.done)
        
        if episode % 2 == 0:
            print('episode ', episode, 'score %.1f' % score[0])
        
        agent.episode_learn()
        agent.update_learn_params()
        scores.append(score[0])
        
        if episode % 50 == 0:
            avg_score = np.mean(scores[-50:])
            if avg_score > best_score:
                best_score = avg_score
                agent.save_model("./output/models/best_model2.pth")
        
    print("scores ",scores)


    fname = 'DQN_' + 'Reward' +\
        '_' + str(n_games) + 'games'

    figure_file  = './output/plots/'  + fname  + '.png'

    x = [i+1 for i in range(n_games)]

    with open("./output/log/scores", "w") as f:
         f.write(str(scores))

    plot_learning_curve(x, scores, figure_file)

if __name__ == '__main__':
    main()

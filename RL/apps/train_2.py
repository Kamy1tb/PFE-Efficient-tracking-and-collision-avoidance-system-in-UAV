

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
from env.env2 import Environment1
from env.env3 import Environment2
from env.DroneClass import AirSimClientDrone
from helpers.utils import *
import threading
import time
from dotenv import load_dotenv
import neptune
from random import choice


velocity = 1  # Vitesse en m/s
duration = 2  # Durée de chaque mouvement en secondes
velocity_waypoint = 3  # Distance entre chaque waypoint en mètres
threads = []

waypoints_track = [
    (velocity_waypoint, 0, 0),  # Premier waypoint (droit)
    (velocity_waypoint, 2, 0),  # Deuxième waypoint (esquive à gauche)
    (velocity_waypoint, 0, 0),  # Troisième waypoint (droit)
    (velocity_waypoint, 0, 0),  # Troisième waypoint (droit)
    (velocity_waypoint, -4, 0),  # Quatrième waypoint (esquive à droite)
    (velocity_waypoint, -4, 0),  # Quatrième waypoint (esquive à droite)
    (velocity_waypoint, 0, 0),  # cinquième waypoint (droit)
    (velocity_waypoint, 0, 0),  # cinquième waypoint (droit)
    (velocity_waypoint, 0, 0),  # cinquième waypoint (droit)
    (velocity_waypoint, 0, 0),  # cinquième waypoint (droit)
    (velocity_waypoint, 2, 0),  # Deuxième waypoint (esquive à gauche)
    (velocity_waypoint, 0, 0),  # Troisième waypoint (droit)
    (velocity_waypoint, 0, 0),  # Troisième waypoint (droit)
    (velocity_waypoint, -4, 0),  # Quatrième waypoint (esquive à droite)
    (velocity_waypoint, -4, 0),  # Quatrième waypoint (esquive à droite)
    (velocity_waypoint, 0, 0),  # cinquième waypoint (droit)

    

    ]

def generate_waypoints():
    waypoints_track = []
    for i in range(1, 16):  
        waypoints_track.append((np.random.uniform(3,4),choice([np.random.uniform(3,4),0,np.random.uniform(-4,-3)]), 0))
    return waypoints_track

def control_drone_target(env,duration,waypoints):
        
        for waypoint in waypoints:
            print("drone 2 ",waypoint)
            x, y, z = waypoint
            env.drone_target.move_by_velocity(x, y, -4, duration,True)
            if env.done:
                break       

        env.drone_target.move_by_velocity(0, 0, -4, 1,True)
        if env.done:
            return
        time.sleep(2)
        env.arrived = 1
         

def train_drone(agent,env,state,score,run):
    ac = 1
    while not env.done:
        action, entropy = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state,action, next_state, reward, done)   
        agent.step_learn() 
        state = next_state
        score[0] += reward
        print("action ",action)
        run["reward_per_step"].append(reward)
        run["entropy"].append(entropy)
        ac +=1
        if env.arrived : 
            env.done = 1
    run["Success"].append(env.nbSuccess)
    run["collisions"].append(env.nbCollision)
    run["n_actions"].append(ac)



def main():
    load_dotenv()
    run = neptune.init_run(
    project="dqdqdq/dqn",
    api_token= os.getenv('API_SECRET_KEY'),
    )
    n_games = 3000       
    eps_dec = 0.005                
    environment = Environment2()

    # Number of layers and activation functions.
    network_spec = [
        dict(type='dense', size=512, activation='relu'),  # 256         
        dict(type='dense', size=512, activation='relu'),  # 128
        dict(type='dense', size=128, activation='relu'),  # 128
    ]

    params = {
            "lr": 0.001,
            "gamma": 0.9,
            "action_space": environment.get_actions(),
            "state_space": environment.get_states(),
            "eps_start": 1,
            "eps_end": 0.01,
            "eps_dec": eps_dec,
            "replay_buffer_size": 50000,
            "batch_size": 256,
            "hidden_size": 40,
            "network_spec": network_spec,
            "target_update": 8,
            }
    with open("./output/log/scores", "w") as f:
         f.write("")

    agent = Agent("DQN")
    agent.configure(params=params)
    #agent.load_model("./output/models/best_model_last.pth")

    scores = []
    best_score = -np.inf
    avg_score = -np.inf
    for episode in range(n_games):
        print("episode ", episode)
        environment.arrived = 0
        environment.done = 0  
        state = environment.reset()
        environment.drone.takeoff(-4, True)
        environment.drone_target.takeoff(-4, True)
        state = environment.generate_state()
        
        score = [0]
        
        print("before threads ",environment.done)
        # Create a new thread for drone control
        drone_thread = threading.Thread(target=control_drone_target, args=(environment,duration,waypoints_track))
        train_thread = threading.Thread(target=train_drone, args=(agent,environment,state,score,run))  
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
        with open("./output/log/scores", "a") as f:
            f.write(f"{episode} , {score[0]} , {agent.epsilon} \n")
        run["reward"].append(score[0])
        run["epsilon"].append(agent.epsilon)

        if episode % 50 == 0:
            avg_score = np.mean(scores[-50:])
            if avg_score > best_score:
                best_score = avg_score
                agent.save_model(f"./output/models/best_model{episode}_2.pth")
        agent.save_model("./output/models/best_model_last2.pth")
        
    print("scores ",scores)
    run.stop()

    """ fname = 'DQN_' + 'Reward' +
        '_' + str(n_games) + 'games'

    figure_file  = './output/plots/'  + fname  + '.png'

    x = [i+1 for i in range(n_games)]

    with open("./output/log/scores", "w") as f:
         f.write(str(scores))

    plot_learning_curve(x, scores, figure_file)"""
        

if __name__ == '__main__':
    main()
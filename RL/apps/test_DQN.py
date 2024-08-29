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
from env.DroneClass import AirSimClientDrone
from helpers.utils import *
import threading
import time
from dotenv import load_dotenv
import neptune
from random import choice

velocity = 1  # Vitesse en m/s
duration = 2  # Durée de chaque mouvement en secondes
velocity_waypoint = 4  # Distance entre chaque waypoint en mètres
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

waypoints_track2 = [
    (velocity_waypoint, 0, 0),  # Premier waypoint (droit)
    (velocity_waypoint, 0, 0),  # Deuxième waypoint (esquive à gauche)
    (velocity_waypoint, 0, 0),  # Troisième waypoint (droit)
    (velocity_waypoint, 0, 0),  # Troisième waypoint (droit)
    (velocity_waypoint, 0, 0),  # Quatrième waypoint (esquive à droite)
    (velocity_waypoint, 0, 0),  # Quatrième waypoint (esquive à droite)
    (velocity_waypoint, 0, 0),  # cinquième waypoint (droit)
    (velocity_waypoint, 0, 0),  # cinquième waypoint (droit)
    (velocity_waypoint, 0, 0),  # cinquième waypoint (droit)
    (velocity_waypoint, 0, 0),  # cinquième waypoint (droit)
    (velocity_waypoint, 0, 0),  # Deuxième waypoint (esquive à gauche)
    (velocity_waypoint, 0, 0),  # Troisième waypoint (droit)
    (velocity_waypoint, 0, 0),  # Troisième waypoint (droit)
    (velocity_waypoint, 0, 0),  # Quatrième waypoint (esquive à droite)
    (velocity_waypoint, 0, 0),  # Quatrième waypoint (esquive à droite)
    (velocity_waypoint, 0, 0),  # cinquième waypoint (droit)

    

    ]


def generate_waypoints():
    waypoints_track = []
    for i in range(1, 16):  
        waypoints_track.append((np.random.uniform(3,4),choice([np.random.uniform(3,4),0,np.random.uniform(-4,-3)]), 0))
    return waypoints_track


def control_drone_target(env,duration):
        waypoints = generate_waypoints()
        for waypoint in waypoints:
            print("drone 2 ",waypoint)
            x, y, z = waypoint
            env.drone_target.move_by_velocity(x, y, -4, duration,True)    

        env.drone_target.move_by_velocity(0, 0, -4, 1,True)
        time.sleep(2)
        env.arrived = 1
         

def test_drone(agent,env,state,score,run):
    ac = 0
    while not env.done:
        action, entropy = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        score[0] += reward
        run["reward_per_step"].append(reward)
        run["entropy"].append(entropy)
        ac +=1
        if env.arrived : 
            env.done = 1
    run["Success"].append(env.nbSuccess)
    run["n_actions"].append(ac)

def main():
    load_dotenv()
    print(os.getenv('API_SECRET_KEY'))
    run = neptune.init_run(
    project="dqdqdq/dqn",
    api_token= os.getenv('API_SECRET_KEY'),
    )
    n_games = 100       
    eps_dec = 0.008                
    environment = Environment1()

    # Number of layers and activation functions.
    network_spec = [
        dict(type='dense', size=1024, activation='relu'),  # 256         
        dict(type='dense', size=1024, activation='relu'),  # 128
        dict(type='dense', size=1024, activation='relu'),
    ]

    params = {
            "lr": 0.00025,
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
    with open("./output/log/scores", "w") as f:
         f.write("")

    agent = Agent("DQN")
    agent.configure(params=params)
    agent.load_model("./output/models/best_model6.pth")
    agent.eval()

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

        environment.drone.move_by_velocity(1, 0, -4, 2, True)
        
        score = [0]
        print("before threads ",environment.done)
        # Create a new thread for drone control
        drone_thread = threading.Thread(target=control_drone_target, args=(environment, duration))
        train_thread = threading.Thread(target=test_drone, args=(agent,environment,state,score,run))  
        drone_thread.start()
        train_thread.start()
        drone_thread.join()
        print("after thread target ",environment.done)
        train_thread.join()
        print("after thread drone ",environment.done)
        
       
        print('episode ', episode, 'score %.1f' % score[0])
        scores.append(score[0])
        run["reward"].append(score[0])
        run["epsilon"].append(agent.epsilon)

        
    print("scores ",scores)
    run.stop()

if __name__ == '__main__':
    main()

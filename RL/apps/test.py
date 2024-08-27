import sys, os
# Add the 'env' directory to the Python path
from unipath import Path

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   
sys.path.append(BASE_PATH)
sys.path.append(Path(BASE_PATH).parent)
from env.environment import Environment
from env.DroneClass import AirSimClientDrone
import threading
import numpy as np
import time
def control_drone(client, waypoints,duration):
            
            client.takeoff(-3,True)
            for waypoint in waypoints:
                x, y, z = waypoint
                client.move_by_velocity(x, y, z, duration,True)
                quad_vel = client.get_velocity()
                drone_state = client.client.getMultirotorState()
                vp = drone_state.kinematics_estimated.linear_velocity
                print("velocity 1: ",quad_vel) 
                print("velocity 2: ",vp)

def interpret_action( step_length ,action):
        if action == 0:
            quad_offset = (step_length, 0, 0)
        elif action == 1:
            quad_offset = (0, step_length, 0)
        elif action == 2:
            quad_offset = (-step_length, 0, 0)
        elif action == 3:
            quad_offset = (0, -step_length, 0)
        else:
            quad_offset = (0, 0, 0)

        return quad_offset

def do_action(drone, action):
        quad_offset = interpret_action(3,action)
        quad_vel = drone.get_velocity()
        
        drone.move_by_velocity(
            quad_offset[0],
            quad_offset[1],
            0,
            2,
            True
        ) 
        drone.move_by_velocity(0, 0, 0, 0.5,True)

if __name__ == "__main__":
    env = Environment("Drone1","Drone2",10)
    drone1 = env.drone
    drone2 = env.drone_target
    pos = drone2.get_position()
    velocity = 1  # Vitesse en m/s
    duration = 2  # Durée de chaque mouvement en secondes
    waypoint_distance = 4  # Distance entre chaque waypoint en mètres
    threads = []

    # Waypoints : ici, nous simulerons des arbres à des positions spécifiques
    waypoints_track = [
    (waypoint_distance, 0, 0),  # Premier waypoint (droit)
    (waypoint_distance, 2, 0),  # Deuxième waypoint (esquive à gauche)
    (waypoint_distance, 0, 0),  # Troisième waypoint (droit)
    (waypoint_distance, 0, 0),  # Troisième waypoint (droit)
    (waypoint_distance, -4, 0),  # Quatrième waypoint (esquive à droite)
    (waypoint_distance, -4, 0),  # Quatrième waypoint (esquive à droite)
    (waypoint_distance, 0, 0),  # cinquième waypoint (droit)
    (waypoint_distance, 0, 0),  # cinquième waypoint (droit)
    (waypoint_distance, 0, 0),  # cinquième waypoint (droit)
    (waypoint_distance, 0, 0),  # cinquième waypoint (droit)
    (waypoint_distance, 2, 0),  # Deuxième waypoint (esquive à gauche)
    (waypoint_distance, 0, 0),  # Troisième waypoint (droit)
    (waypoint_distance, 0, 0),  # Troisième waypoint (droit)
    (waypoint_distance, -4, 0),  # Quatrième waypoint (esquive à droite)
    (waypoint_distance, -4, 0),  # Quatrième waypoint (esquive à droite)
    (waypoint_distance, 0, 0),  # cinquième waypoint (droit)

    

    ]

    waypoints2 = [
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
    drone1.takeoff(-3,True)
        
    while True:
        print("initial position : ",drone1.get_position())
        #ac = np.random.randint(0,4)
        #do_action(drone1,ac)
        # Création et démarrage des threads
        thread1 = threading.Thread(target=control_drone, args=(drone1, waypoints_track,duration))
        #thread2 = threading.Thread(target=control_drone, args=(drone2, waypoints2, duration))
        thread1.start()
        #thread2.start()

        # Attendre la fin des threads
        thread1.join()
        #thread2.join()
          
        env.reset()
        #drone2.client.enableApiControl(True)
        #drone2.client.armDisarm(True)
    #drone1 = AirSimClientDrone("Drone1")

    #print(drone1.get_position())
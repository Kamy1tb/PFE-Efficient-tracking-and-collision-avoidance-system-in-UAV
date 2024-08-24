import sys, os
# Add the 'env' directory to the Python path
from unipath import Path

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   
sys.path.append(BASE_PATH)
sys.path.append(Path(BASE_PATH).parent)
from env.environment import Environment
from env.DroneClass import AirSimClientDrone
import threading

import time
def control_drone(client, waypoints,duration):
            client.takeoff(-3,True)
            for waypoint in waypoints:
                x, y, z = waypoint
                client.move_by_velocity(x, y, z, duration,True)

if __name__ == "__main__":
    env = Environment("Drone1","Drone2",10)
    drone1 = env.drone
    drone2 = env.drone_target
    pos = drone2.get_position()
    velocity = 1  # Vitesse en m/s
    duration = 1  # Durée de chaque mouvement en secondes
    waypoint_distance = 10  # Distance entre chaque waypoint en mètres
    threads = []

    # Waypoints : ici, nous simulerons des arbres à des positions spécifiques
    waypoints1 = [
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


    print("initial position : ",drone1.get_position())

    # Création et démarrage des threads
    thread1 = threading.Thread(target=control_drone, args=(drone1, waypoints1,duration))
    thread2 = threading.Thread(target=control_drone, args=(drone2, waypoints2, duration))
    thread1.start()
    thread2.start()

    # Attendre la fin des threads
    thread1.join()
    thread2.join()
    time.sleep(10)
    print("final position : ",drone1.get_position())
    drone2.client.reset()
    drone2.client.enableApiControl(True)
    drone2.client.armDisarm(True)
    #drone1 = AirSimClientDrone("Drone1")

    #print(drone1.get_position())
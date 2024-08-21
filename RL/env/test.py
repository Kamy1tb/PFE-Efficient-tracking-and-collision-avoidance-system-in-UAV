from DroneClass import AirSimClientDrone
from environment import Environment
import threading

import time
def control_drone(client, waypoints,duration,num_client):
        if (num_client == 1):
            client.drone.takeoff(-6,True)
            for waypoint in waypoints:
                x, y, z = waypoint
                client.drone.move_by_velocity(x, y, z, duration,True)
        else:
            client.takeoff(-6,True)
            for waypoint in waypoints:
                x, y, z = waypoint
                client.move_by_velocity(x, y, z, duration,True)

if __name__ == "__main__":
    drone1 = Environment("Drone1",10)
    drone2 = AirSimClientDrone("Drone2")
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


    

    # Création et démarrage des threads
    thread1 = threading.Thread(target=control_drone, args=(drone1, waypoints1,duration,1))
    thread2 = threading.Thread(target=control_drone, args=(drone2, waypoints2, duration,2))
    thread1.start()
    thread2.start()

    # Attendre la fin des threads
    thread1.join()
    thread2.join()
    
        
    drone2.client.reset()
    drone2.client.enableApiControl(True)
    drone2.client.armDisarm(True)
    #drone1 = AirSimClientDrone("Drone1")

    #print(drone1.get_position())
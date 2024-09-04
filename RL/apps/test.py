import sys, os
# Add the 'env' directory to the Python path
from unipath import Path
from gym import spaces
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   
sys.path.append(BASE_PATH)
sys.path.append(Path(BASE_PATH).parent)
from env.environment import Environment
from env.DroneClass import AirSimClientDrone
import threading
import numpy as np
from env.airsim.types import Vector3r, DrivetrainType, YawMode
import time

duration = 2  # Durée de chaque mouvement en secondes
velocity_waypoint = 3 
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
def control_drone(client, waypoints,duration):
            
            client.takeoff(-4,True)
            for waypoint in waypoints:
                x, y, z = waypoint
                client.move_by_velocity(x, y, -4, duration,True)
                



if __name__ == "__main__":
    drone = AirSimClientDrone("Drone1")
    control_drone(drone,waypoints_track,duration)
    print(drone.get_position())

    #insert move method here
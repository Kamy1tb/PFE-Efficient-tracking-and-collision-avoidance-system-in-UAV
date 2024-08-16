import setup_path
from airsim.types import Vector3r

import airsim
import numpy as np
import math
import time
from DroneClass import AirSimClientDrone
from distance import Distance

if __name__ == "__main__":

    D = 1
    dist = Distance()

    drone = AirSimClientDrone("Drone1")
    position = drone.get_gps_position()
    
    print(position) 
    drone2 = AirSimClientDrone("Drone2")
    position2 = drone2.get_gps_position()
    print(position2)
    lat1, lon1, alt1, lat2, lon2, alt2 = position.latitude, position.longitude, position.altitude, position2.latitude, position2.longitude, position2.altitude
    distance = dist.distance_between_drones(lat1, lon1, alt1, lat2, lon2, alt2)
    Z = distance
    raw = drone.take_raw_photo("high_res")
    png, cylinders = drone.take_box_photo(["Drone2"],"high_res",raw)
    list,info = drone.box_info(cylinders,640,360)
    drone.save_photo(png,"100",".")

    print(f"distance between drones : {distance}")
    print(f"info : {info[0][2] * 640}")

    print(f"calculated distance :  {336.3610833984375 / (info[0][2] * 640)}")



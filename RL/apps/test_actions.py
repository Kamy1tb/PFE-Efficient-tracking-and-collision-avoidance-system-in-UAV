import sys, os
# Add the 'env' directory to the Python path
from unipath import Path

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   
sys.path.append(BASE_PATH)
sys.path.append(Path(BASE_PATH).parent)
from env.airsim.types import Vector3r
import env.setup_path
import env.airsim
import numpy as np
import math
import time
from env.DroneClass import AirSimClientDrone
from gym import spaces
import threading
from env.env3 import Environment2
def _do_action(env, action):
        speed = interpret_action(env,action)
        rotation = env.prev_angle 
        val_x = speed[0] * np.cos(rotation) - speed[1] * np.sin(rotation)
        val_y = speed[0] * np.sin(rotation) + speed[1] * np.cos(rotation)
        if val_x > 5 :
            val_x = 5
        if val_y > 5 : 
            val_y = 5
        if val_x < -5 :
            val_x = -5
        if val_y < -5 : 
            val_y = -5

        env.prev_angle += speed[2]
        print("Speed: ", val_x, val_y)
        env.drone.move_by_velocity(
            val_x,
            val_y,
            -4,
            1,
            True
        )

def interpret_action(env, action):
        if action == 0:
            quad_offset = (env.step_length, 0, 0) #forward
        elif action == 1:
            quad_offset = (0, env.step_length, np.pi/2) #right
        elif action == 2:
            quad_offset = (env.step_length, env.step_length, np.pi/4) # 45 degrees
        elif action == 3:
            quad_offset = (0, -env.step_length, -np.pi/2) #left
        elif action == 4:
            quad_offset = (env.step_length, -env.step_length, -np.pi/4) #-45 degrees
        elif action == 5:
            quad_offset = (env.step_length * 2, 0, 0) #accelerate forward5
        elif action == 6:
            quad_offset = (0, env.step_length *2, np.pi/2) #accelerate right
        elif action == 7:
            quad_offset = (env.step_length * 2, env.step_length * 2, np.pi/4) #accelerate 45 degrees
        elif action == 8:
            quad_offset = (0, -env.step_length *2, -np.pi/2) #accelerate left
        elif action == 9:
            quad_offset = (env.step_length *2, -env.step_length *2, -np.pi/4) #accelerate -45 degrees
        elif action == 10:
            quad_offset = (env.step_length * np.cos(22.5 * np.pi / 180), env.step_length * np.sin(22.5 * np.pi / 180), 22.5 * np.pi/180) #30 degrees
        elif action == 11:
            quad_offset = (env.step_length * np.cos(22.5 * np.pi / 180), -env.step_length * np.sin(22.5 * np.pi / 180), -22.5 * np.pi/180) #-30 degrees
        elif action == 12:
            quad_offset = (env.step_length *np.cos(67.5 * np.pi / 180), env.step_length * np.sin(67.5 * np.pi / 180), 67.5 * np.pi/180) #60 degrees
        elif action == 13:
            quad_offset = (env.step_length *np.cos(67.5 * np.pi / 180), -env.step_length * np.sin(67.5 * np.pi / 180), -67.5 * np.pi/180) #-60 degrees
        elif action == 14:
            quad_offset = (env.step_length * 2 *  np.cos(22.5 * np.pi / 180), env.step_length * 2 * np.sin(22.5 * np.pi / 180), 22.5 * np.pi/180) #accelerate 30 degrees
        elif action == 15:
            quad_offset = (env.step_length * 2 * np.cos(22.5 * np.pi / 180), -env.step_length * 2 * np.sin(22.5 * np.pi / 180), -22.5 * np.pi/180) #accelerate -30 degrees
        elif action == 16:
            quad_offset = (env.step_length * 2 * np.cos(67.5 * np.pi / 180), env.step_length * 2 * np.sin(67.5 * np.pi / 180), 67.5 *np.pi/180) #accelerate 60 degrees
        elif action == 17:
            quad_offset = (env.step_length * 2 * np.cos(67.5 * np.pi / 180), -env.step_length * 2 * np.sin(67.5 * np.pi / 180), - 67.5 *np.pi/180) #accelerate 30 degrees
        else:
            quad_offset = (0, 0, 0) #stop

        return quad_offset
drone_controller = Environment2("Drone1", "Drone2")
drone_controller.drone.takeoff(-4)
while True:
    
    try:
        action = int(input("Enter action (0-19): "))
        if action < 0 :
            print("Invalid action. Please enter a value between 0 and 19.")
            continue
        time.sleep(3)
        _do_action(drone_controller, action)
    except ValueError:
        print("Please enter a valid number.")
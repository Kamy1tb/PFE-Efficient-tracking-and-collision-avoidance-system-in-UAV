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



def interpret_action( action,prev_angle):

        if action == 0:
            quad_offset = (3, 0, 0) #forward
        elif action == 1:
            quad_offset = (0, 3, np.pi/2) #right
        elif action == 2:
            quad_offset = (3, 3, np.pi/4) # 45 degrees
        elif action == 3:
            quad_offset = (0, -3, -np.pi/2) #left
        elif action == 4:
            quad_offset = (3, -3, -np.pi/4) #-45 degrees
        elif action == 5:
            quad_offset = (3 * 2, 0, 0) #accelerate forward
        elif action == 6:
            quad_offset = (0, 3 *2, np.pi/2) #accelerate right
        elif action == 7:
            quad_offset = (3 * 2, 3 * 2, np.pi/4) #accelerate 45 degrees
        elif action == 8:
            quad_offset = (0, -3 *2, -np.pi/2) #accelerate left
        elif action == 9:
            quad_offset = (3 *2, -3 *2, -np.pi/4) #accelerate -45 degrees
        else:
            quad_offset = (0, 0, prev_angle) #stop

        return quad_offset






if __name__ == "__main__":
    drone = AirSimClientDrone("Drone1")
    drone.takeoff(-4,True)

    actions = spaces.Discrete(11)
    prev_angle = 0
    action = 1
    speed = interpret_action(action, prev_angle)

    rotation = prev_angle 
    val_x = speed[0] * np.cos(rotation) - speed[1] * np.sin(rotation)
    val_y = speed[0] * np.sin(rotation) + speed[1] * np.cos(rotation)
    if val_x > 6 :
        val_x = 6
    if val_y > 6 : 
        val_y = 6
    if val_x < -6 :
        val_x = -6
    if val_y < -6 : 
        val_y = -6
    print(f"The speed vector from action = {speed}")
    print(f"Vx = {val_x} ,Vy = {val_y}")

    prev_angle += speed[2]
    drone.client.moveByVelocityZAsync(val_x,val_y,-4,2,DrivetrainType.ForwardOnly,yaw_mode= YawMode(is_rate=False)).join()

    action = 0
    speed = interpret_action(action, prev_angle)

    rotation = prev_angle 
    val_x = speed[0] * np.cos(rotation) - speed[1] * np.sin(rotation)
    val_y = speed[0] * np.sin(rotation) + speed[1] * np.cos(rotation)
    if val_x > 6 :
        val_x = 6
    if val_y > 6 : 
        val_y = 6
    if val_x < -6 :
        val_x = -6
    if val_y < -6 : 
        val_y = -6
    print(f"The speed vector from action = {speed}")
    print(f"Vx = {val_x} ,Vy = {val_y}")

    prev_angle += speed[2]
    drone.client.moveByVelocityZAsync(val_x,val_y,-4,2,DrivetrainType.ForwardOnly,yaw_mode= YawMode(is_rate=False)).join()

    #insert move method here
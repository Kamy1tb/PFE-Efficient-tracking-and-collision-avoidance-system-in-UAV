from env.airsim.types import Vector3r
import env.setup_path
import env.airsim
import numpy as np
import math
import time
from env.DroneClass import AirSimClientDrone
from gym import spaces
import threading


__version__ = '1.0'
__authors__ = 'Sihem Ouahouah & Miloud Bagaa'
__author_emails__ = 'sihem.ouahouah@aalto.fi & miloud.bagaa@aalto.fi'

class Environment2(object):
    def __init__(self, drone_name="Drone1",drone_target="Drone2", step_length=10, areaSideSize=None, observableAccessPoints=None, observableEvents=None):
        self.drone_name = drone_name
        self.drone_target = AirSimClientDrone(drone_target)
        self.drone = AirSimClientDrone(drone_name)
        self.step_length = step_length
        self.camera_name = "high_res"
        self.distances = self.drone.get_distance_all()
        self.render = False
        print(self.distances)
        # Initialize state
        self.state= [0 for i in range(23)]
        self.generate_state()
        self.done = 0 
        self.lock = threading.Lock()
        # Define discrete action space
        self.action_space = spaces.Discrete(21)
        self.step_length = 3
        # Initialize environment properties
        self.areaSideSize = areaSideSize
        self.observableAccessPoints = observableAccessPoints
        self.observableEvents = observableEvents
        self.steps = 0
        self.nbCollision = 0
        self.nbSuccess = 0
        self.arrived = 0
        self.prev_angle = 0
        self.observation_space = self.generate_state()

    def is_done(self):
        with self.lock:
            return self.done
    
    def set_done(self):
        with self.lock:
            self.done = 1

    def get_actions(self):
        return self.action_space

    def get_states(self):
        return self.state
    
    def set_render(self):
        self.render = True

    def unset_render(self):
        self.render = False

    def reset(self):
        self.drone.turn_off_api()

        self.drone.client.reset()

        self.drone.turn_on_api()
        self.drone_target.turn_on_api()
        
        self.state= [0 for i in range(23)]
        self.generate_state()
        self.steps = 0
        self.nbCollision = 0
          
        return self.state

    def _compute_reward(self):
        collision = self.drone.detect_collision()
        rc = 0
        if collision:
            self.done = 1
            self.nbCollision += 1
            return -1000, self.done
        r_ang = 0
        dist = np.linalg.norm([ self.state[6] - 0.5 , self.state[7] - 0.5 ]) #distance from center
        if self.state[6] <0 :
            r_ang = -2
        else:
            r_ang = -dist   # 0 - 1

        rd = 0
        if self.state[6] >0 :  #target in FOV
            if self.state[4] < 2/120 : 
                rd = -2
            elif self.state[4] > 7/120 :    
                rd = -self.state[4] # 0 - 1
            else : 
                rd = 2
        if self.arrived : 
            self.done = 1
        if self.done and self.state[6] != -1 and self.state[4] > 0  and self.state[4] < 15/120:
            self.nbSuccess += 1

        print(f"r_center = {r_ang} , rd = {rd}")
        return rd + r_ang , self.done
                
    

    def _do_action(self, action):
        speed = self.interpret_action(action)
        rotation = self.prev_angle 
        if len(speed) == 4 :
            self.prev_angle += speed[2] / 180 * np.pi
            self.drone.rotate(speed[2], 0.1)
            
        else:
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

            self.prev_angle += speed[2]
            self.drone.move_by_velocity(
                val_x,
                val_y,
                -4,
                1.5,
                True
            )

    def interpret_action(self, action):
        if action == 0:
            quad_offset = (self.step_length, 0, 0) #forward
        elif action == 1:
            quad_offset = (0, self.step_length, np.pi/2) #right
        elif action == 2:
            quad_offset = (self.step_length, self.step_length, np.pi/4) # 45 degrees
        elif action == 3:
            quad_offset = (0, -self.step_length, -np.pi/2) #left
        elif action == 4:
            quad_offset = (self.step_length, -self.step_length, -np.pi/4) #-45 degrees
        elif action == 5:
            quad_offset = (self.step_length * 2, 0, 0) #accelerate forward
        elif action == 6:
            quad_offset = (0, self.step_length *2, np.pi/2) #accelerate right
        elif action == 7:
            quad_offset = (self.step_length * 2, self.step_length * 2, np.pi/4) #accelerate 45 degrees
        elif action == 8:
            quad_offset = (0, -self.step_length *2, -np.pi/2) #accelerate left
        elif action == 9:
            quad_offset = (self.step_length *2, -self.step_length *2, -np.pi/4) #accelerate -45 degrees
        elif action == 10:
            quad_offset = (self.step_length, self.step_length, np.pi/6) #30 degrees
        elif action == 11:
            quad_offset = (self.step_length, -self.step_length, -np.pi/6) #-30 degrees
        elif action == 12:
            quad_offset = (self.step_length, self.step_length, np.pi/3) #60 degrees
        elif action == 13:
            quad_offset = (self.step_length, -self.step_length, -np.pi/3) #-60 degrees
        elif action == 14:
            quad_offset = (self.step_length * 2, self.step_length * 2, np.pi/6) #accelerate 30 degrees
        elif action == 15:
            quad_offset = (self.step_length * 2, -self.step_length * 2, -np.pi/6) #accelerate -30 degrees
        elif action == 16:
            quad_offset = (self.step_length * 2, self.step_length * 2, np.pi/3) #accelerate 60 degrees
        elif action == 17:
            quad_offset = (self.step_length * 2, -self.step_length * 2, -np.pi/3) #accelerate 30 degrees
        elif action == 18:
            quad_offset = (0,0,100, True) #Rotation right
        elif action == 19:
            quad_offset = (0,0,-100, True) #Rotation left
        else:
            quad_offset = (0, 0, 0) #stop

        return quad_offset


    def step(self, action_agent):       
        self._do_action(action_agent)
        next_state = self.generate_state()
        reward, done= self._compute_reward()
        info = []
        return next_state, reward, done, info

    def generate_state(self):
        self.distances = self.drone.get_distance_all()
        try:
            raw = self.drone.take_raw_photo("high_res")
            png,cylinders = self.drone.take_box_photo(["Drone2"],"high_res",raw)
            _ , info = self.drone.box_info(cylinders,640,360)
            if (self.drone.get_estimated_distance(1,info[0][2],336.3610833984375,640) <= 35):
                self.state = [
            self.drone.get_position()[0] / 120,  # position xt                                                       0
            self.drone.get_position()[1] / 120 ,  # position yt                                                      1
            self.state[0],  # position xt-1                                                                          2
            self.state[1],  # position yt-1                                                                          3
            self.drone.get_estimated_distance(1,info[0][2],336.3610833984375,640) / 120, # distance target           4
            self.state[4], # distance target t-1                                                                     5
            info[0][0] ,#position target xt                                                                          6
            info[0][1] , #position target yt                                                                         7
            self.state[6],  # position target xt-1                                                                   8
            self.state[7],  # position target yt-1                                                                   9
            self.drone.get_velocity().x_val,#                                                                        10
            self.drone.get_velocity().y_val,#                                                                        11
            self.state[10], #                                                                                        12
            self.state[11], #                                                                                        13
        ] + self.distances
            else : 
                self.state = [
            self.drone.get_position()[0] / 120,  # position xt                                                       0
            self.drone.get_position()[1] / 120 ,  # position yt                                                      1
            self.state[0],  # position xt-1                                                                          2
            self.state[1],  # position yt-1                                                                          3
            -1, # distance target                                                                                    4
            self.state[4], # distance target t-1                                                                     5
            -1 ,#position target xt                                                                                  6
            -1 , #position target yt                                                                                 7
            self.state[6],  # position target xt-1                                                                   8
            self.state[7],  # position target yt-1                                                                   9
            self.drone.get_velocity().x_val,#                                                                        10
            self.drone.get_velocity().y_val,#                                                                        11
            self.state[10], #                                                                                        12
            self.state[11], #                                                                                        13
        ] + self.distances

            
        except:
            self.state = [
            self.drone.get_position()[0] / 120,  # position xt                                                       0
            self.drone.get_position()[1] / 120 ,  # position yt                                                      1
            self.state[0],  # position xt-1                                                                          2
            self.state[1],  # position yt-1                                                                          3
            -1, # distance target                                                                                    4
            self.state[4], # distance target t-1                                                                     5
            -1 ,#position target xt                                                                                  6
            -1 , #position target yt                                                                                 7
            self.state[6],  # position target xt-1                                                                   8
            self.state[7],  # position target yt-1                                                                   9
            self.drone.get_velocity().x_val,#                                                                        10
            self.drone.get_velocity().y_val,#                                                                        11
            self.state[10], #                                                                                        12
            self.state[11], #                                                                                        13
        ] + self.distances        

        return self.state
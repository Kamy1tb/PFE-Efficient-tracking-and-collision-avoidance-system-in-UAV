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

class Environment1(object):
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
        self.action_space = spaces.Discrete(5)
        self.step_length = 3
        # Initialize environment properties
        self.areaSideSize = areaSideSize
        self.observableAccessPoints = observableAccessPoints
        self.observableEvents = observableEvents
        self.steps = 0
        self.nbCollision = 0
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
        print("reset")
        '''
        
        time.sleep(5)
        drone_target_velocity = self.drone_target.get_velocity()
        drone_velocity = self.drone.get_velocity()
        while drone_target_velocity.x_val != 0 or drone_target_velocity.y_val != 0 or drone_target_velocity.z_val != 0 or drone_velocity.x_val != 0 or drone_velocity.y_val != 0 or drone_velocity.z_val != 0:
            drone_target_velocity = self.drone_target.get_velocity()
            drone_velocity = self.drone.get_velocity()
             self.drone.client.simSetVehiclePose(Vector3r(-5, 0, 0), True, "Drone1")
        self.drone_target.client.simSetVehiclePose(Vector3r(5, 0, 0), True,"Drone2")
        self.drone.land()
        self.drone_target.land()
        #self.drone_target.turn_off_api()
        self.drone.client.reset()
        self.drone_target.turn_on_api()
        self.drone.turn_on_api

            time.sleep(1)'''
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
        s = 0
        if collision:
            rc = -40
            self.done = 1
        else:
            rc = 0
        if self.state[8] == -1:
            rfov = -5
        else:
            rfov = 0
        if self.done and not collision and self.state[8] != -1 and self.state[6] <= 20/120:
            rf = 20
            s = 1
        elif self.done and not collision and (self.state[8] == -1 or self.state[6] > 20/120):
            rf = -20
        else:
            rf = 0
        
        if self.state[7] > 0 and self.state[6] > 0 :
            r_track = self.state[7] - self.state[6]
        else:
            r_track = 0
        if (self.state[6] >= 20/120) and self.state[6] >0 :
            r_dis = -5            
        elif (self.state[6] < 20/120) and self.state[6] > 0 :
            r_dis = +5
        else :
            r_dis = 0
        rd = (r_track * 10 + r_dis) * 0.5
        quad_vel = self.drone.get_velocity()
        vp = np.sqrt(quad_vel.x_val**2 + quad_vel.y_val**2 + quad_vel.z_val**2)
        if vp <= 0.5: 
            r_v = -0.2 * (0.5 - vp)
        else : 
            r_v = 0
        
        rdir = 0.2 * ((self.state[18] - 10) / 10)
        r = 0
        for i in self.state[14:]:
            r += (1/i - 1/10)
        r *= -1.5

        if r < 0 and r > -0.5:
            robs = r
        elif r < -0.5:
            robs = -0.5
        else:
            robs = 0
        r_ang = 0
        dist = np.linalg.norm([ self.state[8] - 0.5 , self.state[9] - 0.5 ])
        if self.state[8] <0 :
            r_ang = 0
        else:
            if dist <= 0.6 : 
                r_ang += (0.6-dist) * 5
            else:
                r_ang -= (dist - 0.6) * 5
        reward = rc + rfov + rd + rf + rdir + robs + r_v + r_ang
        print(f"rc={rc} , rfov={rfov} , rd={rd} , rdir = {rdir} , robs= {robs} , r_v={r_v} , r_ang= {r_ang} , rf={rf} ")
        return reward, self.done,s,rf
    

    def _do_action(self, action):
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.get_velocity()

        val_x = quad_vel.x_val + quad_offset[0]
        val_y = quad_vel.y_val + quad_offset[1]

        if quad_vel.x_val + quad_offset[0] > 6  :
            val_x = 6
        if quad_vel.y_val + quad_offset[1] > 6  :
            val_y = 6
        if quad_vel.x_val + quad_offset[0] < -6  :
            val_x = -6
        if quad_vel.y_val + quad_offset[1] < -6  :
            val_y = -6

        self.drone.move_by_velocity(
            val_x,
            val_y,
            0,
            2,
            True
        ) 

    def interpret_action(self, action):
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
        elif action == 2:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 3:
            quad_offset = (0, -self.step_length, 0)
        else:
            quad_offset = (0, 0, 0)

        return quad_offset


    def step(self, action_agent):
        state = self.generate_state()
        self._do_action(action_agent)
        reward, done, success ,rf= self._compute_reward()
        info = []
        return state, reward, done, info,success,rf

    def generate_state(self):
        self.distances = self.drone.get_distance_all()
        try:
            raw = self.drone.take_raw_photo("high_res")
            png,cylinders = self.drone.take_box_photo(["Drone2"],"high_res",raw)
            list,info = self.drone.box_info(cylinders,640,360)
            if (self.drone.get_estimated_distance(1,info[0][2],336.3610833984375,640) <= 30):
                self.state = [
            self.drone.get_position()[0] / 120,  # position xt
            self.drone.get_position()[1] / 120,  # position yt
            self.drone.get_position()[2] / 120,  # position zt
            self.state[0],  # position xt-1
            self.state[1],  # position yt-1
            self.state[2],  # position zt-1
            self.drone.get_estimated_distance(1,info[0][2],336.3610833984375,640) / 120, # distance target
            self.state[6], # distance target t-1
            (info[0][0]) ,info[0][1] , #position target xt, yt
            self.state[8],  # position target xt-1
            self.state[9],  # position target yt-1
            self.state[10],  # position target xt-2
            self.state[11],  # position target yt-2
        ] + self.distances
            else : 
                self.state = [
            self.drone.get_position()[0] / 120,  # position xt
            self.drone.get_position()[1] / 120,  # position yt
            self.drone.get_position()[2] / 120,  # position zt
            self.state[0],  # position xt-1
            self.state[1],  # position yt-1
            self.state[2],  # position zt-1
            -1, # distance target
            self.state[6], # distance target t-1
            -1 ,-1 , #position target xt, yt
            self.state[8],  # position target xt-1
            self.state[9],  # position target yt-1
            self.state[10],  # position target xt-2
            self.state[11],  # position target yt-2
        ] + self.distances

            
        except:
            self.state = [
            self.drone.get_position()[0] / 120,  # position xt
            self.drone.get_position()[1] / 120,  # position yt
            self.drone.get_position()[2] / 120,  # position zt
            self.state[0],  # position xt-1
            self.state[1],  # position yt-1
            self.state[2],  # position zt-1
            -1, # distance target
            self.state[6], # distance target t-1
            -1,-1, #position target xt, yt
            self.state[8],  # position target xt-1
            self.state[9],  # position target yt-1
            self.state[10],  # position target xt-2
            self.state[11],  # position target yt-2
        ] + self.distances

        print(self.state[6],self.state[8],self.state[9])            
        

        return self.state

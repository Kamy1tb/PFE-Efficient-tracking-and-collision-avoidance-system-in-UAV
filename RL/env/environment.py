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

class Environment(object):
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
        self.num_bins = 10
        self.time_seconds = 3
        self.discretized_values = np.linspace(-1, 1, self.num_bins)
        self.action_space = spaces.Discrete(self.num_bins * self.num_bins * self.time_seconds)

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
        

        self.generate_state()
        self.steps = 0
        self.nbCollision = 0
          
        return self.state
    
    def _compute_reward(self):
        collision = self.drone.detect_collision()
        if collision:
            rc = -50
            self.done = 1
        else:
            rc = 0
        if self.state[8] == -1:
            rfov = -10
        else:
            rfov = 0
        if self.done and not collision and self.state[8] != -1 and self.state[6] <= 20/120:
            rf = 50
        elif self.state[8] == -1:
            rf = -50
        else:
            rf = 0
        
        if self.state[7] > 0 and self.state[6] > 0 :
            r_track = self.state[7] - self.state[6]
        else:
            r_track = 0
        if (self.state[6] >= 20/120) and self.state[6] >0 :
            r_dis = -10            
        elif self.state[6] > 0 :
            r_dis = +5
        else :
            r_dis = 0
        rd = (r_track + r_dis) * 0.2
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
        dist = np.linalg.norm([ self.state[8] - 0.5 , self.state[9] - 0.5) ])
        if self.state[8] <0 :
            r_ang = 0
        else:
            if dist <= 0.6 : 
                r_ang += dist * 10
            else:
                r_ang -= (dist - 0.6) * 10
        reward = rc + rfov + rd + rf + rdir + robs + r_v + r_ang
        print(f"rc={rc} , rfov={rfov} , rd={rd} , rdir = {rdir} , robs= {robs} , r_v={r_v} , r_ang= {r_ang}")
        return reward, self.done
    

    def _do_action(self, action):
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.get_velocity()
        vp = np.sqrt(quad_vel.x_val**2 + quad_vel.y_val**2 + quad_vel.z_val**2)
        a_vp = (vp + quad_offset[0]) / vp
        print("quad_offset = ",quad_offset)
        print("a_vp = ",a_vp)
        new_x = np.cos(quad_offset[1] )* quad_vel.x_val - np.sin(quad_offset[1] )* quad_vel.y_val
        new_y = np.sin(quad_offset[1] ) * quad_vel.x_val+ np.cos(quad_offset[1] )* quad_vel.y_val
        print("new_x = ",new_x)
        print("new_y = ",new_y)
        self.drone.move_by_velocity(
            new_x * a_vp,
            new_y * a_vp,
            0,
            quad_offset[2],
            True
        )

    def interpret_action(self, action):
        duration = action // (self.num_bins ** 2) + 1
        x_index = (action % (self.num_bins**2)) // self.num_bins
        y_index = action % self.num_bins
        x = self.discretized_values[x_index]
        y = self.discretized_values[y_index]
        delta_speed = 3 * x
        delta_angle = 30 * y
        return [delta_speed, delta_angle,duration]


    def step(self, action_agent):
        state = self.generate_state()
        self._do_action(action_agent)
        reward, done = self._compute_reward()
        info = []
        return state, reward, done, info

    def generate_state(self):
        self.distances = self.drone.get_distance_all()
        try:
            raw = self.drone.take_raw_photo("high_res")
            png,cylinders = self.drone.take_box_photo(["Drone2"],"high_res",raw)
            list,info = self.drone.box_info(cylinders,640,360)
            self.state = [
            self.drone.get_position()[0] / 120,  # position xt
            self.drone.get_position()[1] / 120,  # position yt
            self.drone.get_position()[2] / 120,  # position zt
            self.state[0],  # position xt-1
            self.state[1],  # position yt-1
            self.state[2],  # position zt-1
            self.drone.get_estimated_distance(1,info[0][2],336.3610833984375,640) / 120, # distance target
            self.state[6], # distance target t-1
            info[0][0] / 640 ,info[0][1] / 360 , #position target xt, yt
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

            
        

        return self.state

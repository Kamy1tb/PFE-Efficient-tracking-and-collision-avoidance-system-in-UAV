from DroneClass import AirSimClientDrone
from environment import Environment
import time
if __name__ == "__main__":
    #Creating Target drone and setting waypoints
    drone2 = AirSimClientDrone("Drone2")
    pos = drone2.get_position()
    pos = pos.tolist()
    waypoints = [(5,2,0,5),(7,2,3,5)]
    # Creating environment of the tracking drone 
    env = Environment("Drone1",10)
    print("env created")
    time.sleep(2)

    #Takeoff the drones
    env.drone.takeoff(-6,True)
    drone2.takeoff(-6,True)
  
    #Move the drones to the waypoints
    for waypoint in waypoints:
        drone2.move_by_velocity(waypoint[0],waypoint[1],waypoint[2],waypoint[3])
        env.drone.move_by_velocity(waypoint[0],waypoint[1],waypoint[2],waypoint[3])
        
        time.sleep(2)


    time.sleep(2)
    obs = env.reset()
    print(obs)
    #
    #waypoints2 = [(position2.x_val+50,position2.y_val,position2.z_val),(position2.x_val+50, position2.y_val+50, position2.z_val-10)]

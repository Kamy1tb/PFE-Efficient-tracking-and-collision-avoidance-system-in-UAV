from DroneClass import AirSimClientDrone
from environment import Environment
import time
if __name__ == "__main__":
    #Creating Target drone and setting waypoints
    drone2 = AirSimClientDrone("Drone2")
    pos = drone2.get_position()
    pos = pos.tolist()
    waypoints = [(3,0,0,5),(1,-1,1,5)]
    # Creating environment of the tracking drone 
    env = Environment("Drone1",10)
    print("env created")

    #Takeoff the drones
    env.drone.takeoff(-6,True)
   # drone2.takeoff(-6,True)
    """
    #Move the drones to the waypoints
    for waypoint in waypoints:
        #while not (drone2._calculate_distance(drone2.get_position(),(waypoint[0],waypoint[1],waypoint[2])) < 0.5 and env.drone._calculate_distance(env.drone.get_position(),(waypoint[0],waypoint[1],waypoint[2])) < 0.5 ):
        drone2.move_by_velocity(waypoint[0],waypoint[1],waypoint[2],waypoint[3])
        env.drone.move_by_velocity(waypoint[0],waypoint[1],waypoint[2],waypoint[3])
        
        time.sleep(2)
    """
    env.drone.move(10,0,-6,5,True)
    obs = env.reset()
    print(obs)
    print(env.drone.get_position())
    #
    #waypoints2 = [(position2.x_val+50,position2.y_val,position2.z_val),(position2.x_val+50, position2.y_val+50, position2.z_val-10)]

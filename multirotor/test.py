import setup_path
import airsim
import cv2
import pprint
import os
import time
import json
class AirSimClientDrone:
    def __init__(self, drone_name):
        self.drone_name = drone_name
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, drone_name)
        self.client.armDisarm(True, drone_name)
        self.initial_position = self.client.getMultirotorState(self.drone_name).kinematics_estimated.position
    def return_client(self):
        return self.client
    
    def takeoff(self,z,join):
        if join==False:
            self.client.takeoffAsync(vehicle_name=self.drone_name)
            self.client.moveToPositionAsync(0,0, z,4,vehicle_name=self.drone_name)
        else:
            self.client.takeoffAsync(vehicle_name=self.drone_name).join()
            self.client.moveToPositionAsync(0,0, z,4,vehicle_name=self.drone_name).join()

    def get_position(self):
        return self.client.getMultirotorState(self.drone_name).kinematics_estimated.position
    
    
    def take_raw_photo(self,camera_name):
        raw_image = self.client.simGetImage(camera_name, airsim.ImageType.Scene)
        png = cv2.imdecode(airsim.string_to_uint8_array(raw_image), cv2.IMREAD_UNCHANGED)
        return png
    
    def take_box_photo(self, meshe,camera_name,photo):
        self.client.simSetDetectionFilterRadius(camera_name, airsim.ImageType.Scene, 200 * 100) 
        self.client.simAddDetectionFilterMeshName(camera_name, airsim.ImageType.Scene, meshe) 
        png = photo
        cylinders = self.client.simGetDetections(camera_name,airsim.ImageType.Scene)
        print("Found %d cylinders in image" % len(cylinders))
        for cylinder in cylinders:
                s = pprint.pformat(cylinder)
                print("test: %s" % s)
                cv2.rectangle(png,(int(cylinder.box2D.min.x_val),int(cylinder.box2D.min.y_val)),(int(cylinder.box2D.max.x_val),int(cylinder.box2D.max.y_val)),(255,0,0),2)
                cv2.putText( png, cylinder.name, (int(cylinder.box2D.min.x_val),int(cylinder.box2D.min.y_val - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12))
        return png, cylinders

    def save_photo(self,image,filename,folder):
        image_path = os.path.join(folder, f"train_{filename}.png")
        cv2.imwrite(image_path, image)
        print(f"Image saved to {image_path}")
         

    def move(self, x, y, z,speed):
        self.client.moveToPositionAsync(x, y, z,speed,vehicle_name=self.drone_name)

    def rotate(self, yaw,rotation_duration):
        self.client.rotateByYawRateAsync(yaw, rotation_duration,vehicle_name=self.drone_name)

    def land(self):
        self.client.landAsync(vehicle_name=self.drone_name).join()

    def do_random_path(self):
        print("Random path")

    def box_info(self,class_label,min_x,min_y,max_x,max_y,width_photo,height_photo,folder,filename):
        center_x = (max_x+min_x)/2 * (1/width_photo)
        center_y = (max_y+min_y)/2 * (1/height_photo)
        width = (max_x-min_x) * (1/width_photo)
        height = (max_y-min_y) * (1/height_photo)

        filepath = os.path.join(folder, f"train_{filename}.txt")
        with open(filepath, 'w') as file:
            file.write(f"{class_label} {center_x} {center_y} {width} {height}")


if __name__ == "__main__":

    drone = AirSimClientDrone("Drone1")
    drone2 = AirSimClientDrone("Drone2")
    position = drone.get_position()
    position2 = drone2.get_position()
    print("postion = {}".format(position))
    print("postion2 = {}".format(position2))
    drone.takeoff(-10,True) 
    drone2.takeoff(-10,True) 
    position = drone.get_position()
    position2 = drone2.get_position()
    print("postion = {}".format(position))
    print("postion2 = {}".format(position2))

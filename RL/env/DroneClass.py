import env.setup_path
import env.airsim as airsim
import cv2
import pprint
import os
import math
from env.model import YOLOModel
from env.airsim.types import Vector3r, DrivetrainType, YawMode
import time 
import numpy as np

classes = [
    {"index": 0, "nom": "Drone2", "type": "target"},
    {"index": 1, "nom": "tree", "type": "obstacle"}
]

MIN_DEPTH_METERS = 0
MAX_DEPTH_METERS = 100

class AirSimClientDrone:
    def __init__(self,drone_name):
        self.drone_name = drone_name
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, drone_name)
        self.client.armDisarm(True, drone_name)
        self.initial_position = self.client.getMultirotorState(self.drone_name).kinematics_estimated.position
        print(f"{self.drone_name} is set !")

    def return_client(self):
        return self.client
    
    def takeoff(self,z,join=False):
        if join==False:
            self.client.takeoffAsync(vehicle_name=self.drone_name)
            self.client.moveToPositionAsync(0,0, z,4,vehicle_name=self.drone_name)
        else:
            self.client.takeoffAsync(vehicle_name=self.drone_name).join()
            self.client.moveToPositionAsync(0,0, z,4,vehicle_name=self.drone_name).join()

    def get_position(self):       
        return self.client.getMultirotorState(self.drone_name).kinematics_estimated.position.to_numpy_array()
    
    
    def get_velocity(self):
        return self.client.getMultirotorState(self.drone_name).kinematics_estimated.linear_velocity
    
    
    def take_raw_photo(self,camera_name):
        raw_image = self.client.simGetImage(camera_name, airsim.ImageType.Scene)
        png = cv2.imdecode(airsim.string_to_uint8_array(raw_image), cv2.IMREAD_UNCHANGED)
        return png
    
    def get_gps_position(self):
        return self.client.getMultirotorState(self.drone_name).gps_location
    
    def take_depth_photo(self, camera_name):
        start_time = time.time()
    # Utilisation de simGetImages pour une seule demande d'image
        response, = self.client.simGetImages(
        [
            airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, True, False),
        ]
        )

        print(f"Time taken for image request : {time.time()-start_time}")
    
    # Conversion de la liste en tableau 2D directement
        start_time = time.time()
        depth_img_in_meters = np.array(response.image_data_float, dtype=np.float64).reshape(response.height, response.width)

        print(f"Time taken for depth in meters: {time.time()-start_time}")
    
    # Lerp 0..100m to 0..255 gray values
        start_time = time.time()
        depth_8bit_lerped = np.interp(depth_img_in_meters, (MIN_DEPTH_METERS, MAX_DEPTH_METERS), (0, 255)).astype('uint8')
        cv2.imwrite("depth_visualization.png", depth_8bit_lerped)

        print(f"Time taken for image 8bit lerped: {time.time()-start_time}")

    # Convert depth_img to millimeters and clamp large values
        start_time = time.time()
        depth_img_in_millimeters = np.clip(depth_img_in_meters * 1000, 0, 65535).astype('uint16')
        cv2.imwrite("depth_16bit.png", depth_img_in_millimeters)

        print(f"Time taken for depth in milimeters: {time.time()-start_time}")
    
        return depth_img_in_millimeters

    def take_box_photo(self, meshes,camera_name,photo):
        self.client.simSetDetectionFilterRadius(camera_name, airsim.ImageType.Scene, 200 * 100) 
        for mesh in meshes:
            self.client.simAddDetectionFilterMeshName(camera_name, airsim.ImageType.Scene, mesh)
        png = photo
        cylinders = self.client.simGetDetections(camera_name,airsim.ImageType.Scene)
        print("Found %d cylinders in image" % len(cylinders))
        for cylinder in cylinders:
                s = pprint.pformat(cylinder)
                cv2.rectangle(png,(int(cylinder.box2D.min.x_val),int(cylinder.box2D.min.y_val)),(int(cylinder.box2D.max.x_val),int(cylinder.box2D.max.y_val)),(255,0,0),2)
                cv2.putText( png, cylinder.name, (int(cylinder.box2D.min.x_val),int(cylinder.box2D.min.y_val - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12))
        return png, cylinders

    def save_photo(self,image,filename,folder):
        image_path = os.path.join(folder, f"train_{filename}.png")
        cv2.imwrite(image_path, image)
        print(f"Image saved to {image_path}")
         

    def move(self, x, y, z,speed,join=False):
        if join:
            self.client.moveToPositionAsync(x, y, z,speed,vehicle_name=self.drone_name).join()
        else:
            self.client.moveToPositionAsync(x, y, z,speed,vehicle_name=self.drone_name)
        print(f"{self.drone_name} moved to {x}, {y}, {z}")

    def move_by_velocity(self,vx,vy,z,time,join=False):
        if join:
            self.client.moveByVelocityZAsync(vx,vy,z,time,DrivetrainType.ForwardOnly,yaw_mode= YawMode(is_rate=False),vehicle_name=self.drone_name).join()
        else:
            self.client.moveByVelocityZAsync(vx,vy,z,time,DrivetrainType.ForwardOnly,yaw_mode= YawMode(is_rate=False),vehicle_name=self.drone_name)
        print(f"{self.drone_name} moved with speed vector [{vx}, {vy}, {z}] with duration {time}")
    

    def rotate(self, yaw,rotation_duration):
        self.client.rotateByYawRateAsync(yaw, rotation_duration,vehicle_name=self.drone_name).join()

     

    def land(self):
        self.client.landAsync(10,vehicle_name=self.drone_name).join()

    def do_random_path(self):
        print("Random path")





    def box_info(self,cylinders,width_photo,height_photo):
        
        liste = []
        info = []
        for cylinder in cylinders :
            center_x = (cylinder.box2D.max.x_val+cylinder.box2D.min.x_val)/2 * (1/width_photo)
            center_y = (cylinder.box2D.max.y_val+cylinder.box2D.min.y_val)/2 * (1/height_photo)
            width = (cylinder.box2D.max.x_val-cylinder.box2D.min.x_val) * (1/width_photo)
            height = (cylinder.box2D.max.y_val-cylinder.box2D.min.y_val) * (1/height_photo)
            class_label = cylinder.name
            liste.append(f"{class_label} {center_x} {center_y} {width} {height}")
            info.append([center_x,center_y,width,height])
            return liste,info
        
    def box_info_2(self,cylinders,width_photo,height_photo):
        info = []
        for cylinder in cylinders :
            center_x = (cylinder.box2D.max.x_val+cylinder.box2D.min.x_val)/2 * (1/width_photo)
            center_y = (cylinder.box2D.max.y_val+cylinder.box2D.min.y_val)/2 * (1/height_photo)
            width = (cylinder.box2D.max.x_val-cylinder.box2D.min.x_val) * (1/width_photo)
            height = (cylinder.box2D.max.y_val-cylinder.box2D.min.y_val) * (1/height_photo)
            class_label = cylinder.name
            info.append([center_x,center_y,width,height])
        return info
    
    def save_box_info_2(self,info,folder,filename):
            liste = []
            liste.append(f"{info[0]} {info[1]} {info[2]} {info[3]} {info[4]}")
            filepath = os.path.join(folder, f"train_{filename}.txt")
            with open(filepath, 'w') as file:
                for line in liste:
                    file.write(line+"\n")

    def save_box_info(self,liste,folder,filename):

        filepath = os.path.join(folder, f"train_{filename}.txt")
        with open(filepath, 'w') as file:
            for line in liste:
                file.write(line+"\n")
                
    def detect_collision(self):
        collision_info = self.client.simGetCollisionInfo(self.drone_name)
        if collision_info.has_collided and collision_info.object_id != -1 and collision_info.object_name != "beam_metal_window_divider_4x1129" and collision_info.object_name != "Floor136":
            print("Collision detected for drone %s" % self.drone_name)
            print(f"Nom de l'objet: {collision_info.object_name}")
            return True
        else:
            return False
    def action_collision(self,object_name):
        if object_name == "Drone2":
            self.rotate(180,1)
            self.move(self.client.getMultirotorState(self.drone_name).kinematics_estimated.position.x_val -10,self.client.getMultirotorState(self.drone_name).kinematics_estimated.position.y_val, self.client.getMultirotorState(self.drone_name).kinematics_estimated.position.z_val,4)
            

        else:
           self.rotate(180,1)
    def _calculate_distance(self,point1, point2):
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        squared_diff_x = (x2 - x1) ** 2
        squared_diff_y = (y2 - y1) ** 2
        squared_diff_z = (z2 - z1) ** 2
        distance = math.sqrt(squared_diff_x + squared_diff_y + squared_diff_z)

        return distance
    def set_camerapose(self,x,y,z,roll,pitch,yaw):
        pose = airsim.Pose(airsim.Vector3r(x,y,z), airsim.to_quaternion(roll,pitch,yaw))
        self.client.simSetCameraPose("high_res",pose)

    
    def get_distance(self,num_sensor):
        distance = self.client.getDistanceSensorData(f"Distance{num_sensor}")
        return distance
    

    
    def get_estimated_distance(self,real_width,pixel_width,focal_length,camera_width):
        return focal_length * real_width / (pixel_width* camera_width)



    def near_collision(self):
        for i in range(1,9):
            distance = self.client.getDistanceSensorData(f"Distance{i}")
            print("distance "+f"{i}"+": "+f"{distance.distance}")
            if distance.distance < 10 :
                return True
        
        return False
    
    def get_distance_all(self): 
        distances = []
        for i in range(1,10):
            distance = self.client.getDistanceSensorData(f"Distance{i}")
            distances.append(distance.distance)
        return distances

    def get_lidar(self):
        for i in range(1,5):
            lidarData = self.client.getLidarData()
            if (len(lidarData.point_cloud) < 3):
                print("\tNo points received from Lidar data")
                print(lidarData.point_cloud)
            else:
                points = self.parse_lidarData(lidarData)
                print("\tReading %d: time_stamp: %d number_of_points: %d" % (i, lidarData.time_stamp, len(points)))
                print("\t\tlidar position: %s" % (pprint.pformat(lidarData.pose.position)))
                print("\t\tlidar orientation: %s" % (pprint.pformat(lidarData.pose.orientation)))
            time.sleep(5)


    def parse_lidarData(self, data):

        # reshape array of floats to array of [X,Y,Z]
        points = np.array(data.point_cloud, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0]/3), 3))
       
        return points
    
    def predict_yolo(self,photo):
            model = YOLOModel('./best.pt')
            raw = photo
            raw = cv2.cvtColor(raw, cv2.COLOR_RGBA2RGB)
            photo = raw.copy()
            start_time = time.time()
            predict = model.predict(photo)
            print(predict)
            print(f"Time taken: {time.time()-start_time}")
            return predict

    def turn_on_api(self):
        self.client.enableApiControl(True, self.drone_name)
        self.client.armDisarm(True, self.drone_name)
    def turn_off_api(self):
        self.client.enableApiControl(False, self.drone_name)
        self.client.armDisarm(False, self.drone_name)
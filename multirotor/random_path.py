import setup_path
import airsim
import random
import os
import cv2
import numpy as np
from PIL import Image
import time
import io
import pprint

# Connect to AirSim
client = airsim.MultirotorClient()

client.confirmConnection()
client.enableApiControl(True, "Drone1")
client.enableApiControl(True, "Drone2")
client.armDisarm(True, "Drone1")
client.armDisarm(True, "Drone2")

# Takeoff
f1 = client.takeoffAsync(vehicle_name="Drone1")
f2 = client.takeoffAsync(vehicle_name="Drone2")
f1.join()
f2.join()

# Set initial position
initial_position_drone1 = client.getMultirotorState("Drone1").kinematics_estimated.position
print(initial_position_drone1)
current_position = initial_position_drone1

# Create folder for saving images
image_folder = r"C:\Users\tkamy\Documents\Unreal Projects\Airsim_Taibi\PythonClient\multirotor\randompics"
os.makedirs(image_folder, exist_ok=True)

yaw_angle = 90
rotation_duration = 1
airsim.ImageType.Scene
client.simSetDetectionFilterRadius("high_res", airsim.ImageType.Scene, 200 * 100) 
# add desired object name to detect in wild card/regex format
client.simAddDetectionFilterMeshName("high_res", airsim.ImageType.Scene, "Drone2") 

# Move in random path
for i in range(10):  # Adjust the number of iterations as needed
    # Generate random movement
    if i == 5: 
        client.rotateByYawRateAsync(90, rotation_duration,vehicle_name="Drone1")
        client.rotateByYawRateAsync(90, rotation_duration,vehicle_name="Drone2").join()
        dx = 0 #random.uniform(-10, 10)
        dy = 10 #random.uniform(-10, 10)
    elif i > 5:
        dx = 0 #random.uniform(-10, 10)
        dy = 10 #random.uniform(-10, 10)
        dz = random.uniform(-20, 0)
    else:
        dx = 10
        dy = 0
        
        

    # Update current position
    current_position.x_val += dx
    current_position.y_val += dy
    #current_position.z_val += dz

    # Move to new position
    client.moveToPositionAsync(current_position.x_val, current_position.y_val, current_position.z_val, 4, vehicle_name="Drone1")
    client.moveToPositionAsync(current_position.x_val, current_position.y_val, current_position.z_val, 4, vehicle_name="Drone2").join()

    # Take picture
    #image_response = client.simGetImages([airsim.ImageRequest("high_res", airsim.ImageType.Scene, False, False)])
    #print(image_response)
    #image_data = Image.open(io.BytesIO(image_response))
    
    #image_data.save(image_path)


    response = client.simGetImage("high_res", airsim.ImageType.Scene)
    png = cv2.imdecode(airsim.string_to_uint8_array(response), cv2.IMREAD_UNCHANGED)
    cylinders = client.simGetDetections("high_res", airsim.ImageType.Scene)
    while not cylinders:
        for cylinder in cylinders:
            s = pprint.pformat(cylinder)
            print("test: %s" % s)

            cv2.rectangle(png,(int(cylinder.box2D.min.x_val),int(cylinder.box2D.min.y_val)),(int(cylinder.box2D.max.x_val),int(cylinder.box2D.max.y_val)),(255,0,0),2)
            cv2.putText(png, cylinder.name, (int(cylinder.box2D.min.x_val),int(cylinder.box2D.min.y_val - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12))
        image_path = os.path.join(image_folder, f"image_{i}.png")
        cv2.imwrite(image_path, png)
        print(f"Image saved to {image_path}")

    # Enregistrer les images capturées
    '''for idx, response in enumerate(responses):
        image_path = os.path.join(image_folder, f"image_{idx*10+i}.png")
        if response.pixels_as_float:
            # Convertir les pixels en valeurs de 0 à 255 (pour OpenCV)
            img1d = np.array(response.image_data_float, dtype=np.float32) 

            img1d = img1d * 255 / np.max(img1d)
            img2d = np.reshape(img1d, (response.height, response.width))
            image = np.array(img2d, dtype=np.uint8)
        else:
            # Décompresser les pixels JPEG
            image = cv2.imdecode(airsim.string_to_uint8_array(response.image_data_uint8), cv2.IMREAD_UNCHANGED)

        # Enregistrer l'image
        cv2.imwrite(image_path, image)
        print(f"Image saved to {image_path}")'''


    # Get drone state
    drone_state = client.getMultirotorState()
    drone_distance = client.getDistanceSensorData(vehicle_name="Drone1")
    print(f"Distance sensor data: drone 1: {drone_distance.distance}")

    '''# Check drone state and move accordingly
    if drone_state == "state1":
        client.moveToPositionAsync(state1_position.x_val, state1_position.y_val, state1_position.z_val, 1).join()
    elif drone_state == "state2":
        client.moveToPositionAsync(state2_position.x_val, state2_position.y_val, state2_position.z_val, 1).join()
    elif drone_state == "state3":
        client.moveToPositionAsync(state3_position.x_val, state3_position.y_val, state3_position.z_val, 1).join()
    '''

client.reset()
client.armDisarm(False)

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)
# yolo_model.py
from ultralytics import YOLO
from PIL import Image
import cv2
import os
class YOLOModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def predict(self, image,i):
        results = self.model(image,save = True,name = f"folder{i}")
        # Extraire les informations des objets détectés
        predictions = []
        for result in results:
            for obj in result.boxes:
                print(obj)
                class_name = self.model.names[int(obj.cls)]  # Nom de la classe (par exemple, SM_PineTree16)
                confidence = obj.conf.item()  # Confiance de la prédiction
                x_center, y_center, width, height = obj.xywh[0].tolist()  # Coordonnées de la bounding box en format (centre_x, centre_y, largeur, hauteur)

                predictions.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height
                })

        return results
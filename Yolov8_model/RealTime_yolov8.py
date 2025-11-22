from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor # ->  from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

model = YOLO("Mouse-Com.pt")

result = model.predict(source="0", show=True) # accepts all formats - img/floder/vid.* ()
print(result)
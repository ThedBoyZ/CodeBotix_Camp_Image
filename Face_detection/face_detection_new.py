import cv2
import numpy as np
import requests

xml_model_url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
xml_path = 'haarcascade_frontalface_default.xml'

def download_xml(url, path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded {path} successfully.")
    else:
        raise Exception(f"Failed to download {url}, status code: {response.status_code}")

# Download haarCascade XML file
download_xml(xml_model_url, xml_path)

cascade = cv2.CascadeClassifier(xml_path)

if cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')


# Video Capture Start 'Camera' ---> Source 0
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    objects = cascade.detectMultiScale(
        gray, 
        scaleFactor=1.3,
        minNeighbors=5, 
        minSize=(30, 30)
    )
    
    for (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    cv2.imshow('Face Object Detection', frame)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import urllib.request
import os

# Auto-download model on first run
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Done!")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Hand skeleton connections
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

webcam = cv2.VideoCapture(0)
timestamp = 0

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, image = webcam.read()
        if not success:
            break

        timestamp += 1
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        results = landmarker.detect_for_video(mp_image, timestamp)

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Draw connections
                for start, end in CONNECTIONS:
                    x1 = int(hand_landmarks[start].x * w)
                    y1 = int(hand_landmarks[start].y * h)
                    x2 = int(hand_landmarks[end].x * w)
                    y2 = int(hand_landmarks[end].y * h)
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw landmark dots
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (cx, cy), 6, (255, 0, 0), -1)

        cv2.imshow("Hand Tracking", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

webcam.release()
cv2.destroyAllWindows()

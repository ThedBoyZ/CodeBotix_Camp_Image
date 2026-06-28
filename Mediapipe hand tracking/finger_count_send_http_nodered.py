import cv2
import mediapipe as mp
import requests
import urllib.request
import os

url = 'http://127.0.0.1:1880/data'

# ─── Auto-download model 
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )
    print("Model ready!")

# ─── Colors per finger (BGR) 
FINGER_COLORS = {
    'thumb':  (255,  50,  50),   # blue
    'index':  ( 50, 255,  50),   # green
    'middle': (  0, 200, 255),   # orange
    'ring':   (200,  50, 255),   # purple
    'pinky':  ( 50,  50, 255),   # red
    'palm':   (200, 200, 200),   # light gray
}

# ─── Which landmarks belong to which finger
LANDMARK_FINGER = {
     0: 'palm',
     1: 'thumb',  2: 'thumb',  3: 'thumb',  4: 'thumb',
     5: 'index',  6: 'index',  7: 'index',  8: 'index',
     9: 'middle', 10: 'middle', 11: 'middle', 12: 'middle',
    13: 'ring',   14: 'ring',   15: 'ring',   16: 'ring',
    17: 'pinky',  18: 'pinky',  19: 'pinky',  20: 'pinky',
}

# ─── Connections grouped by finger
FINGER_CONNECTIONS = {
    'thumb':  [(0,1),(1,2),(2,3),(3,4)],
    'index':  [(0,5),(5,6),(6,7),(7,8)],
    'middle': [(0,9),(9,10),(10,11),(11,12)],
    'ring':   [(0,13),(13,14),(14,15),(15,16)],
    'pinky':  [(0,17),(17,18),(18,19),(19,20)],
    'palm':   [(5,9),(9,13),(13,17)],
}

def draw_landmarks(image, landmarks, h, w):
    # Draw colored lines per finger
    for finger, connections in FINGER_CONNECTIONS.items():
        color = FINGER_COLORS[finger]
        for start, end in connections:
            x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
            x2, y2 = int(landmarks[end].x * w),   int(landmarks[end].y * h)
            cv2.line(image, (x1, y1), (x2, y2), color, 2)

    # Draw colored dots per landmark
    for i, lm in enumerate(landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)
        color  = FINGER_COLORS[LANDMARK_FINGER[i]]
        cv2.circle(image, (cx, cy), 7, color, -1)            # filled dot
        cv2.circle(image, (cx, cy), 7, (255, 255, 255), 1)   # white outline

# ─── Setup Tasks API
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ─── Main loop 
capture   = cv2.VideoCapture(0)
timestamp = 0

with HandLandmarker.create_from_options(options) as landmarker:
    while capture.isOpened():
        success, image = capture.read()
        if not success:
            print("Ignored empty webcam's frame")
            continue

        timestamp += 1
        h, w = image.shape[:2]

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results   = landmarker.detect_for_video(mp_image, timestamp)

        fingerCount = 0

        if results.hand_landmarks:
            for handIndex, hand_landmarks in enumerate(results.hand_landmarks):
                handLabel = results.handedness[handIndex][0].category_name

                lm = [[p.x, p.y] for p in hand_landmarks]

                # Thumb
                if handLabel == "Left"  and lm[4][0] > lm[3][0]: fingerCount += 1
                elif handLabel == "Right" and lm[4][0] < lm[3][0]: fingerCount += 1

                # Four fingers
                if lm[8][1]  < lm[6][1]:  fingerCount += 1
                if lm[12][1] < lm[10][1]: fingerCount += 1
                if lm[16][1] < lm[14][1]: fingerCount += 1
                if lm[20][1] < lm[18][1]: fingerCount += 1

                draw_landmarks(image, hand_landmarks, h, w)

        cv2.putText(image, str(fingerCount),
                    (50, 450), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 255, 0), 10)
        cv2.imshow('FingerCounting Apps', image)

        key = cv2.waitKey(1)
        if key == 27: # ESC to exit
            break
        if key & 0xFF == ord('c'): # c -> Send finger count to Node-RED
            try:
                response = requests.post(url, data=str(fingerCount), timeout=5)
                print(f"Sent: {fingerCount} fingers" if response.status_code == 200
                      else f"Error: {response.text}")
            except requests.Timeout:
                print("Request timed out.")
            except requests.RequestException as e:
                print("Error:", e)

capture.release()
cv2.destroyAllWindows()

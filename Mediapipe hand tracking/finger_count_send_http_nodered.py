import cv2
import mediapipe as mp
import requests
url = 'http://127.0.0.1:1880/receive-data'

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

capture = cv2.VideoCapture(0)
with mp_hands.Hands(
  model_complexity=0,
  min_detection_confidence=0.5,
  min_tracking_confidence=0.5) as hands:
  while capture.isOpened():
    success, image = capture.read()
    if not success:
      print('Ignored empty webcam\'s frame')
      continue
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fingerCount = 0

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        handIndex = results.multi_hand_landmarks.index(hand_landmarks)
        handLabel = results.multi_handedness[handIndex].classification[0].label

        handLandmarks = []

        for landmarks in hand_landmarks.landmark:
          handLandmarks.append([landmarks.x, landmarks.y])
        
        if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
          fingerCount = fingerCount + 1
        elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
          fingerCount = fingerCount + 1

        if handLandmarks[8][1] < handLandmarks[6][1]:
          fingerCount = fingerCount + 1
        if handLandmarks[12][1] < handLandmarks[10][1]:
          fingerCount = fingerCount + 1
        if handLandmarks[16][1] < handLandmarks[14][1]:
          fingerCount = fingerCount + 1
        if handLandmarks[20][1] < handLandmarks[18][1]:
          fingerCount = fingerCount + 1

        mp_drawing.draw_landmarks(
          image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style()
        )

    cv2.putText(image, str(fingerCount), (50,450), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255,0,0), 10)
    cv2.imshow('FingerCounting Apps',image)

    if cv2.waitKey(1) == 27: # Check if the ASCII value of the pressed key is 27 (ESC key)
        break
    if cv2.waitKey(1) & 0xFF == ord('c'): # Check if the ASCII value of the pressed key is 99 (C key) // 32 is (backspace key)
      # send http buffer string to http in Node-red not recall.     
      try:
          response = requests.post(url, data=str(fingerCount), timeout=5)  # Set timeout to 5 seconds
          if response.status_code == 200:
              print('Data sent successfully')
          else:
              print('Error sending data:', response.text)
      except requests.Timeout:
          print('Request timed out. Server did not respond in time.')
      except requests.RequestException as e:
          print('An error occurred:', e)
  capture.release()

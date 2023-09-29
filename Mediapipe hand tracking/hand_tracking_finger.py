import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
webcam = cv2.VideoCapture(0)

# Initialize the Hand model
hands = mp_hands.Hands()

while True:
    # Read a frame from the webcam
    success, image = webcam.read()

    # Convert the frame from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    results = hands.process(image_rgb)

    # Print the detected hand landmarks' positions
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)            


    cv2.imshow("Webcam", image)
    cv2.waitKey(1)


    # webcam.release()
    # cv2.destroyAllWindows()
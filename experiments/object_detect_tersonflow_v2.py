import cv2
import numpy as np
import tensorflow as tf
model = tf.keras.applications.MobileNetV2()
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

image_path = "image.jpg"
image = cv2.imread(image_path)
preprocessed_image = preprocess_image(image)
predictions = model.predict(preprocessed_image)
results = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]

for result in results:
    _, label, confidence = result
    print(f"{label}: {confidence * 100:.2f}%")
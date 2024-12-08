import tensorflow as tf
import numpy as np
from src.utils import preprocess_image
import cv2
# Load the trained model
model = tf.keras.models.load_model('models/model.keras')

def predict_image(img_path):
    img = cv2.imread(img_path)
    resized_image = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    resized_image = np.expand_dims(resized_image, axis=0)  
    predictions = model_01.predict(resized_image)
    if  predictions[0][0] < 0.5:
        return "NO"

    else:
        return "YES"

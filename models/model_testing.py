# models/model_testing.py
import tensorflow as tf
import numpy as np
import cv2

def predict_image(image_path, model_path='models/trained_model.h5'):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Preprocess the input image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Perform prediction
    prediction = model.predict(img)
    result = 'Autistic' if prediction[0][0] > 0.5 else 'Non-Autistic'
    return result

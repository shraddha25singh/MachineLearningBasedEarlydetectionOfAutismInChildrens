# models/model_training.py
'''import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2

def load_data(data_path):
    images, labels = [], []
    for label, class_dir in enumerate(['Non-Autistic', 'Autistic']):
        class_path = os.path.join(data_path, class_dir)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (128, 128))  # Resize to match input size
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load and preprocess the data
data_path = "dataset"  # Local dataset path
X, y = load_data(data_path)
X = X / 255.0  # Normalize pixel values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
model.save('models/trained_model.h5')
'''


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2

def load_data(data_path):
    """
    Load images and labels from the dataset directory.

    Args:
    - data_path (str): Path to the dataset directory.

    Returns:
    - Tuple (np.array, np.array): Images and labels as numpy arrays.
    """
    images, labels = [], []
    for label, class_dir in enumerate(['Non-Autistic', 'Autistic']):
        class_path = os.path.join(data_path, class_dir)
        if not os.path.exists(class_path):
            raise FileNotFoundError(f"Directory not found: {class_path}")
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: Unable to read image: {img_path}")
                continue
            img = cv2.resize(img, (128, 128))  # Resize to match input size
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Path to the dataset
#data_path = "C:\Users\Shraddha singh\Desktop\autism_detection\dataset"  # Replace with the correct path if different
data_path = r"C:\Users\Shraddha singh\Desktop\autism_detection - Webapp\dataset"

try:
    # Load and preprocess the data
    X, y = load_data(data_path)
    X = X / 255.0  # Normalize pixel values to [0, 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model definition
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

    # Save the model
    model.save('models/trained_model.h5')
    print("Model training complete and saved as 'models/trained_model.h5'.")

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

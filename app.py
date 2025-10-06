import os
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import UserMixin, login_user, login_required, logout_user, current_user, LoginManager
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import cv2
import tensorflow as tf
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Bcrypt for password hashing
bcrypt = Bcrypt(app)

# Login manager configuration
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# File upload configuration
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Ensure the uploads folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# User loader for login manager
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load the trained model
try:
    model = tf.keras.models.load_model('models/trained_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Helper function to preprocess images
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 128))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize pixel values
    return np.expand_dims(img_array, axis=0)

# Function to generate confusion matrix as base64 image
def generate_confusion_matrix():
    """Generate and return confusion matrix as base64 image"""
    # Example confusion matrix (replace with actual values)
    cm = np.array([[50, 10], [5, 35]])

    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap="coolwarm")
    plt.colorbar(cax)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), va='center', ha='center', color='black')

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# Function to generate accuracy graph as base64 image
def generate_accuracy_graph():
    """Generate and return accuracy graph as base64 image"""
    epochs = [1, 2, 3, 4, 5]
    accuracy = [0.6, 0.7, 0.75, 0.8, 0.85]  # Example accuracy values (replace with actual data)

    fig, ax = plt.subplots()
    ax.plot(epochs, accuracy, marker='o', linestyle='-', color='b')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Prediction Accuracy Over Time")

    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'danger')
            return redirect(url_for('signup'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Account created successfully!', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Logged in successfully.', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')

    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=current_user.username)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded.', 'danger')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file.', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            if model:
                try:
                    image_data = preprocess_image(file_path)
                    prediction = model.predict(image_data)
                    result = 'Autistic' if prediction[0][0] > 0.5 else 'Non-Autistic'
                    cm_img = generate_confusion_matrix()
                    acc_img = generate_accuracy_graph()
                    return render_template('result.html', result=result, cm_img=cm_img, acc_img=acc_img)
                except Exception as e:
                    flash(f"Error during prediction: {str(e)}", 'danger')
            else:
                flash('Model not loaded. Please check the setup.', 'danger')

    return render_template('upload.html')
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

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Autistic', 'Autistic'],
                yticklabels=['Non-Autistic', 'Autistic'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

def plot_training_history(history):
    """Plot training accuracy and loss graphs."""
    plt.figure(figsize=(12, 5))

    # Accuracy Graph
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label="Train Accuracy")
    plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training vs Validation Accuracy")

    # Loss Graph
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")

    plt.show()

def evaluate_uploaded_image(model, uploaded_image_path):
    """Evaluate a single uploaded image using the trained model."""
    img = cv2.imread(uploaded_image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Unable to read image {uploaded_image_path}")
        return
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Reshape for model input

    prediction = model.predict(img)[0][0]
    predicted_label = "Autistic" if prediction > 0.5 else "Non-Autistic"
    print(f"Uploaded Image Prediction: {predicted_label}")

    # Confusion matrix update
    true_label = int(input("Enter actual label (0 for Non-Autistic, 1 for Autistic): "))
    plot_confusion_matrix([true_label], [int(prediction > 0.5)])

# Path to dataset
data_path = r"C:\Users\Shraddha singh\Desktop\autism_detection - Copy\dataset"

try:
    # Load dataset
    X, y = load_data(data_path)
    X = X / 255.0  # Normalize
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define model
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

    # Train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

    # Save model
    model.save('models/trained_model.h5')
    print("Model saved as 'models/trained_model.h5'.")

    # Plot confusion matrix after training
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    plot_confusion_matrix(y_test, y_pred)

    # Plot training graphs
    plot_training_history(history)

    # Evaluate an uploaded image
    uploaded_image_path = input("Enter path of uploaded image: ")  # User provides image path
    model = load_model('models/trained_model.h5')  # Load trained model
    evaluate_uploaded_image(model, uploaded_image_path)

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

@app.route('/result')
@login_required
def result():
    result = request.args.get('result', 'No result available.')
    cm_img = generate_confusion_matrix()  # Generate confusion matrix image
    acc_img = generate_accuracy_graph()  # Generate accuracy graph image
    return render_template('result.html', result=result, cm_img=cm_img, acc_img=acc_img)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

@login_manager.unauthorized_handler
def unauthorized():
    flash("You need to log in to access this page.", "warning")
    return redirect(url_for('login'))

# Initialize database and run the app
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)



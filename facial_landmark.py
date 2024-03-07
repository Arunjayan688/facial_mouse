import dlib
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Set the paths for the training dataset and the model
dataset_path = "data/300W"
model_path = "model/f_L_68.h5"

# Load dataset and split into training and validation sets
def load_dataset(dataset_path):
    images = []
    landmarks = []
    for foldername in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, foldername)
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                images.append(gray)
                landmark_path = os.path.join(folder_path, filename.split('.')[0] + ".txt")
                with open(landmark_path, 'r') as file:
                    landmark = np.array([int(x) for x in file.read().split()], dtype=np.int32)
                    landmarks.append(landmark)

    images = np.array(images)
    landmarks = np.array(landmarks)

    # Split dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, landmarks, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val

# Data augmentation function
def augment_data(images, landmarks):
    augmented_images = []
    augmented_landmarks = []
    for image, landmark in zip(images, landmarks):
        # Flip horizontally
        flipped_image = cv2.flip(image, 1)
        flipped_landmark = np.copy(landmark)
        for i in range(len(flipped_landmark)):
            if i % 2 == 0:  # X coordinates
                flipped_landmark[i] = image.shape[1] - landmark[i]
        augmented_images.append(flipped_image)
        augmented_landmarks.append(flipped_landmark)
    augmented_images = np.array(augmented_images)
    augmented_landmarks = np.array(augmented_landmarks)
    return np.concatenate((images, augmented_images)), np.concatenate((landmarks, augmented_landmarks))

# Define model architecture
def create_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(68*2)  # Output layer with 68*2 output nodes (x, y coordinates for each of 68 landmarks)
    ])
    return model

# Load dataset
X_train, X_val, y_train, y_val = load_dataset(dataset_path)

# Data augmentation
X_train_augmented, y_train_augmented = augment_data(X_train, y_train)

# Reshape data for CNN
X_train_augmented = X_train_augmented.reshape(-1, X_train_augmented.shape[1], X_train_augmented.shape[2], 1)
X_val = X_val.reshape(-1, X_val.shape[1], X_val.shape[2], 1)

# Normalize input data
X_train_augmented = X_train_augmented / 255.0
X_val = X_val / 255.0

# Create and compile the model
model = create_model(input_shape=X_train_augmented.shape[1:])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_augmented, y_train_augmented, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the trained model
model.save(model_path)

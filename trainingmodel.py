
import cv2
import os
import random
import shutil
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

# Directory to store the collected face images
data_directory = "face_data"

# Directory to store the dataset
dataset_directory = "facerecognition"

# Create the dataset directory if it doesn't exist
if not os.path.exists(dataset_directory):
    os.makedirs(dataset_directory)

# Create train and test directories
train_directory = os.path.join(dataset_directory, "train")
test_directory = os.path.join(dataset_directory, "test")

if not os.path.exists(train_directory):
    os.makedirs(train_directory)

if not os.path.exists(test_directory):
    os.makedirs(test_directory)

# Move the collected face images to train and test directories
for person_name in os.listdir(data_directory):
    person_directory = os.path.join(data_directory, person_name)
    if os.path.isdir(person_directory):
        images = os.listdir(person_directory)
        random.shuffle(images)
        train_images = images[:int(len(images) * 0.8)]  # 80% for training
        test_images = images[int(len(images) * 0.8):]  # 20% for testing

        # Define the class labels
        class_labels = ["sneha", "supraja"]

        # Determine the destination folder (sneha, supraja, or unknown)
        if person_name in class_labels:
            destination_train = os.path.join(train_directory, person_name)
            destination_test = os.path.join(test_directory, person_name)
        else:
            destination_train = os.path.join(train_directory, "unknown")
            destination_test = os.path.join(test_directory, "unknown")

        # Move images to train directory
        for image in train_images:
            source = os.path.join(person_directory, image)
            os.makedirs(destination_train, exist_ok=True)
            shutil.move(source, destination_train)

        # Move images to test directory
        for image in test_images:
            source = os.path.join(person_directory, image)
            os.makedirs(destination_test, exist_ok=True)
            shutil.move(source, destination_test)

# Remove the empty person directories from the data directory
for person_name in os.listdir(data_directory):
    person_directory = os.path.join(data_directory, person_name)
    if os.path.isdir(person_directory) and not os.listdir(person_directory):
        os.rmdir(person_directory)

# VGG16 model
vgg = VGG16(input_shape=(224, 224, 3), weights="imagenet", include_top=False)

# Freeze pre-trained layers
for layer in vgg.layers:
    layer.trainable = False

# Flatten the output layer
x = Flatten()(vgg.output)

# Add a dense layer
x = Dense(1024, activation="relu")(x)

# Add the final classification layer
output = Dense(len(os.listdir(train_directory)), activation="softmax")(x)

# Create the model
model = Model(inputs=vgg.input, outputs=output)

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Data augmentation for training set
train_datagen = ImageDataGenerator(rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# Data augmentation for test set
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Set up the batch size and target image size
batch_size = 32
target_size = (224, 224)

# Generate the training set
train_set = train_datagen.flow_from_directory(train_directory, target_size=target_size, batch_size=batch_size,
                                              class_mode="categorical")

# Generate the test set
test_set = test_datagen.flow_from_directory(test_directory, target_size=target_size, batch_size=batch_size,
                                            class_mode="categorical")

# Train the model
r = model.fit(train_set, validation_data=test_set, epochs=10)

# Save the trained model
model.save("trained_model5.h5")

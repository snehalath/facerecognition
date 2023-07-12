
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Path to the trained model
model_path = "trained_model5.h5"

# Load the trained model
model = load_model(model_path)


# Define the class labels
class_labels = ["sneha", "supraja"]

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess the input image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, (224, 224))  # Resize to match the input size of the model
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add an extra dimension for batch size
    return image

# Function to perform prediction on the input image
def predict_image(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)
    if predicted_class < len(class_labels):
        predicted_label = class_labels[predicted_class]
        confidence = prediction[0][predicted_class] * 100
    else:
        predicted_label = "Unknown"
        confidence = 0.0
    return predicted_label, confidence

# Open the video capture
video_capture = cv2.VideoCapture(0)

# Process frames from the video
while True:
    # Read a frame from the video
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face_image = frame[y:y+h, x:x+w]

        # Perform prediction on the face image
        predicted_label, confidence = predict_image(face_image)

        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Write the predicted label and confidence on the frame
        label = f"{predicted_label}: {confidence:.2f}%"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        #
    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Check for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
video_capture.release()

# Destroy all windows
cv2.destroyAllWindows()

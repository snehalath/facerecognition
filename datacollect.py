import cv2
import os

# Directory to store the collected face images
data_directory = "face_data/supraja"

# Create the data directory if it doesn't exist
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

# Webcam capture
cap = cv2.VideoCapture(0)

# Face classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Number of collected face images per person
num_images_per_person = 10

# Counter to track the number of collected images
image_count = 0

# Input person name
person_name = input("Enter the name of the person: ")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the face image
        face_image = gray[y:y + h, x:x + w]
        image_filename = os.path.join(data_directory, person_name + "_" + str(image_count) + ".jpg")
        cv2.imwrite(image_filename, face_image)

        image_count += 1

        # Break if collected the desired number of images
        if image_count >= num_images_per_person:
            break

    # Display the frame
    cv2.imshow("Collecting Face Data", frame)

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

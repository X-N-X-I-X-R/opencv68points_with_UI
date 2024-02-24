import cv2
import dlib
import numpy as np
import face_recognition
import os

# Load the face landmark predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Directory containing your images
image_dir = "./static/"
# Get the list of image file names
image_files = os.listdir(image_dir)
# Filter out any non-image files
image_files = [f for f in image_files if f.endswith('.png')]

# Load each image and create face encoding
your_face_encodings = []
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    if face_encodings:
        your_face_encodings.extend(face_encodings)

# Function to align faces

def align_face(image):
    # Convert the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = dlib.get_frontal_face_detector()(gray)

    for face in faces:
        # Get the landmarks
        shape = predictor(gray, face)

        # Get the coordinates of the left and right eye
        left_eye = shape.part(36)
        right_eye = shape.part(45)

        # Calculate the angle between the two eyes
        dX = right_eye.x - left_eye.x
        dY = right_eye.y - left_eye.y
        angle = np.degrees(np.arctan2(dY, dX))

        # Compute the desired right eye x-coordinate based on the desired x-coordinate of the left eye
        desired_right_eye_x = 1.0 - 0.35

        # Determine the scale of the new resulting image by taking the ratio of the distance between eyes
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desired_dist = (desired_right_eye_x - 0.35)
        desired_dist *= 256
        scale = desired_dist / dist

        # Compute center (x, y)-coordinates (i.e., the median point) between the two eyes in the input image
        eyes_center = ((left_eye.x + right_eye.x) // 2, (left_eye.y + right_eye.y) // 2)

        # Grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # Update the translation component of the matrix
        tX = 256 * 0.5
        tY = 256 * 0.45
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])

        # Apply the affine transformation
        (w, h) = (256, 256)
        output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        return output

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a single frame
    ret, frame = video_capture.read()

    # Align the frame
    aligned_frame = align_face(frame)

    # Find all face encodings in the current frame
    face_encodings = face_recognition.face_encodings(aligned_frame)

    # If there are any faces in the frame
    if face_encodings:
        # Check if the first face matches any of your faces
        match = face_recognition.compare_faces(your_face_encodings, face_encodings[0])

        if True in match:
            print("It's you!")
        else:
            print("It's not you!")

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture when everything is done
video_capture.release()
cv2.destroyAllWindows()
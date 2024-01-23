import cv2
import dlib

# Load the detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Open webcam
cap = cv2.VideoCapture(0)

frame_count = 0

while True:
    _, frame = cap.read()
    frame_count += 1

    # Skip every 2nd frame to speed up
    if frame_count % 2 == 0:
        continue

    # Resize frame to half its size for faster processing
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        # Print facial landmarks with their corresponding names
        for landmark_index in range(0, 68):
            x = landmarks.part(landmark_index).x
            y = landmarks.part(landmark_index).y

            # Define named constants for better readability
            JAW_THRESHOLD = 17
            REB_THRESHOLD = 22
            LEB_THRESHOLD = 27
            NOSE_THRESHOLD = 36
            REYE_THRESHOLD = 42
            LEYE_THRESHOLD = 48

            if landmark_index < JAW_THRESHOLD:
                print(f"Jaw Point {landmark_index}: ({x}, {y})")
            elif landmark_index < REB_THRESHOLD:
                print(f"Right Eyebrow Point {landmark_index - JAW_THRESHOLD}: ({x}, {y})")
            elif landmark_index < LEB_THRESHOLD:
                print(f"Left Eyebrow Point {landmark_index - REB_THRESHOLD}: ({x}, {y})")
            elif landmark_index < NOSE_THRESHOLD:
                print(f"Nose Point {landmark_index - LEB_THRESHOLD}: ({x}, {y})")
            elif landmark_index < REYE_THRESHOLD:
                print(f"Right Eye Point {landmark_index - NOSE_THRESHOLD}: ({x}, {y})")
            elif landmark_index < LEYE_THRESHOLD:
                print(f"Left Eye Point {landmark_index - REYE_THRESHOLD}: ({x}, {y})")
            else:
                print(f"Mouth Point {landmark_index - LEYE_THRESHOLD}: ({x}, {y})")

        # Draw facial landmarks
        for landmark_index in range(0, 68):
            x = landmarks.part(landmark_index).x
            y = landmarks.part(landmark_index).y
            cv2.circle(frame, (x, y), 1, (144, 238, 144), 1)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

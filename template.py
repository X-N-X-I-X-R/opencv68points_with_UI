from threading import Timer
import webbrowser
from flask import Flask, render_template, Response
import cv2
import dlib
import numpy as np

app = Flask(__name__)

# Load the detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)
def generate_frames():
    frame_count = 0
    while True:
        _, frame = cap.read()
        frame_count += 1

        # Skip every 2nd frame to speed up
        if frame_count % 2 == 0:
            continue

        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)

            # Draw landmarks with a smaller size and different color
            for landmark_index in range(0, 68):
                x = landmarks.part(landmark_index).x
                y = landmarks.part(landmark_index).y
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Green color, smaller size

            # Draw lines with a different color and thickness
            draw_3d_lines(frame, landmarks, color=(0, 0, 255), thickness=1)  # Red color, thinner lines

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def draw_3d_lines(frame, landmarks, color=(0, 0, 255), thickness=1):
    # Connect points to create a 3D effect
    for landmark_index in range(0, 67):
        x1, y1 = landmarks.part(landmark_index).x, landmarks.part(landmark_index).y
        x2, y2 = landmarks.part(landmark_index + 1).x, landmarks.part(landmark_index + 1).y
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

def draw_landmarks(frame, landmarks, start, end, color=(0, 255, 0), radius=2):
    for landmark_index in range(start, end):
        x = landmarks.part(landmark_index).x
        y = landmarks.part(landmark_index).y
        cv2.circle(frame, (x, y), radius, color, -1)  # Smaller size

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    Timer(1, lambda: webbrowser.open("http://127.0.0.1:5000/")).start()
    app.run(debug=True, port=5000)

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

            for landmark_index in range(0, 68):
                x = landmarks.part(landmark_index).x
                y = landmarks.part(landmark_index).y

                if landmark_index == 8:  # Change this to the index of a landmark you want to highlight
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

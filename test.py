from flask import Flask, Response, stream_with_context
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
import requests
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime, timedelta
from threading import Thread
import time

# Initialize Firebase Admin
cred = credentials.Certificate('fb.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'realtimedatabse.firebaseio.com/' #replace with yours
})

app = Flask(__name__)
CORS(app)

# ESP32-CAM video stream URL
ESP32_CAM_URL = "http://192.168.1.114:80/stream"

# Load the pre-trained model (MobileNet-SSD)
model = tf.saved_model.load('ssd_mobilenet_v2_coco_2018_03_29/saved_model')

frame_storage = None
human_detected = False
last_detection_time = None
db.reference('/human_detected').set(False)
def detect_human(frame):
    global human_detected, last_detection_time
    try:
        # Resize frame to model's expected size
        frame_resized = cv2.resize(frame, (300, 300))  # Adjust dimensions if different for your model

        # Convert frame to the format expected by the model
        input_tensor = tf.convert_to_tensor([frame_resized], dtype=tf.uint8)

        # Use the 'serving_default' signature for inference
        detect_fn = model.signatures['serving_default']

        # Perform the detection
        detections = detect_fn(input_tensor)

        # Loop over detections and draw boxes around humans
        for i in range(int(detections['num_detections'])):
            class_id = int(detections['detection_classes'][0][i])
            score = float(detections['detection_scores'][0][i])
            if class_id == 1 and score > 0.5:  # Class 1 is for "person"
                # Get bounding box coordinates
                box = detections['detection_boxes'][0][i].numpy()
                h, w, _ = frame.shape
                box = box * np.array([h, w, h, w])
                y1, x1, y2, x2 = box.astype('int')
                # Draw a rectangle around the detected person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                human_detected = True
                last_detection_time = datetime.now()
                db.reference('/human_detected').set(True)
        return frame
    except Exception as e:
        print(f"Error in detect_human: {e}")
        return frame  # Return the original frame if an error occurs

def reset_human_detected():
    global human_detected, last_detection_time
    while True:
        if human_detected and (datetime.now() - last_detection_time) > timedelta(minutes=1):
            human_detected = False
            db.reference('/human_detected').set(False)
        time.sleep(10)  # Check every 10 seconds

def fetch_stream():
    global frame_storage
    # Connect to the ESP32-CAM video stream
    stream = requests.get(ESP32_CAM_URL, stream=True)

    bytes = b''
    for chunk in stream.iter_content(chunk_size=1024):
        bytes += chunk
        # Check if a JPEG frame is received
        a = bytes.find(b'\xff\xd8')
        b = bytes.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes[a:b+2]  # Extract the JPEG frame
            bytes = bytes[b+2:]  # Remove the processed frame from the byte buffer

            # Decode the JPEG frame
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

            # Detect humans in the frame
            frame = detect_human(frame)

            frame_storage = frame

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    def generate():
        global frame_storage
        while True:
            if frame_storage is not None:
                ret, buffer = cv2.imencode('.jpg', frame_storage)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Concatenate video frame data
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + b'No Frame' + b'\r\n')
    return Response(stream_with_context(generate()), mimetype='multipart/x-mixed-replace; boundary=frame')

def display_video():
    global frame_storage
    while True:
        if frame_storage is not None:
            # Define the desired display size
            display_width = 800
            display_height = 600

            # Resize the frame for display
            display_frame = cv2.resize(frame_storage, (display_width, display_height))

            # Display the resized frame
            cv2.imshow('ESP32-CAM Human Detection', display_frame)

            if cv2.waitKey(1) == 27:  # exit on ESC key
                break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Start background threads
    fetch_thread = Thread(target=fetch_stream)
    display_thread = Thread(target=display_video)
    reset_thread = Thread(target=reset_human_detected)

    fetch_thread.start()
    display_thread.start()
    reset_thread.start()

    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)

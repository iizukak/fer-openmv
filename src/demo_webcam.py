from predictor import Predictor
import config
import dataset
import cv2
import os
import numpy as np

import time

WINDOW_NAME = config.NETWORK + " demo"


def demo(device_num, ext='jpg', delay=1):
    cap = setup_webcam(device_num)
    # cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
    cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    p = Predictor()
    inference_result = None

    while True:
        ret, frame = cap.read()
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        
        # Execute Face Detection
        faces = detect_face(frame, cascade)

        # Draw Detected Face
        for (x,y,w,h) in faces:
            # Inference test data
            face_img = frame[y:y+h, x:x+w]
            face_img, _ = cv2.decolor(face_img)
            face_img = cv2.resize(face_img, dsize=(dataset.IMAGE_SIZE, dataset.IMAGE_SIZE))
            # cv2.imshow(WINDOW_NAME, face_img)
            start = time.time()
            inference_result = p(np.array(face_img, dtype=np.float32))
            end = time.time()
            measured_time = (end - start) * 1000
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        if inference_result is not None:
            expression = dataset.CLASS_NAME[np.argmax(inference_result)]
            if not np.argmax(inference_result) in [0, 3, 5, 6]:
                expression = "None"

            percentage = ": {0:.2f}".format(inference_result[np.argmax(inference_result)] * 100)
            cv2.putText(frame, 
                        expression + ": " + percentage, 
                        (10, 400),
                        cv2.FONT_HERSHEY_PLAIN,
                        2, 
                        (0, 0, 255), 
                        2, 
                        cv2.LINE_AA)

            cv2.putText(frame, 
                       "{0:.2f} ms, {1:.2f} fps".format(measured_time, 1000.0 / measured_time), 
                       (10, 440),
                       cv2.FONT_HERSHEY_PLAIN,
                       2, 
                       (0, 255, 0), 
                       2, 
                       cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)

    cv2.destroyWindow(WINDOW_NAME)


def setup_webcam(device_num):
    cap = cv2.VideoCapture(device_num)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        sys.exit(-1)

    time.sleep(2)
    return cap



def detect_face(frame, cascade):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray_frame, 1.11, 5)
    return faces


if __name__ == '__main__':
    demo(0, "captures")

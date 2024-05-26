import cv2 as cv
import numpy as np

eye_cascade = cv.CascadeClassifier('./detectors/haarcascade_eye.xml')

def detect_eye(image_file):
    image = np.array(image_file.convert('RGB'))
    eyes = eye_cascade.detectMultiScale(image, 1.1, 6)
    for (x, y, w, h) in eyes:
        cv.rectangle(image, (x, y), (x + w, w + h), (255, 255, 0))
    return image, eyes
    
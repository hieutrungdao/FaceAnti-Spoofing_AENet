import cv2
import numpy as np
from detector import Detector

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector = Detector()

# To capture video from webcam.
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('http://192.168.0.100:4747/video')
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')


while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        
        face = img[y:y+h, x:x+w]
        face = np.expand_dims(face, axis=0)
        out = detector.predict(face)
        prediction = float(out[0][1])
        print(prediction)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        text = str()

        if (prediction > 0.1):
            text = "fake"
        else:
            text = "real"

        cv2.putText(img, text+str(prediction), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import pygame

pygame.mixer.init()

sound = pygame.mixer.Sound("pop-up-something-160353.wav") #sound file

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)  # Start the camera capture


is_drinking = False  # Flag to track if drinking coffee

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    faces = face_cascade.detectMultiScale(  
        gray, scaleFactor=1.3, minNeighbors=5)  # it will detect faces in the grayscale frame

    for (x, y, w, h) in faces:

        mouth_roi = gray[y + h//2:y + h, x:x + w]  # roi is region of interest

        _, mouth_thresh = cv2.threshold(mouth_roi, 50, 255, cv2.THRESH_BINARY)  

        white_pixels = np.sum(mouth_thresh == 255)

        if white_pixels > 10000:
            if not is_drinking:
                print("Drinking coffee detected!")
                is_drinking = True
                sound.play()
        else:
            is_drinking = False

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Now It will draw rectangle around detected face

    cv2.imshow('frame', frame)  # Displaying the resulting frame

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

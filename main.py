from calendar import c
import cv2
from random import randrange


# Load some pre-trained data on face from OpenCV
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
# img = cv2.imread('img2.jpg')
webcam = cv2.VideoCapture(0)

# Iterate forever over frames
while True:

    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # Must convert to grayscale(B&W)
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

    # Draw rectange around the face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 4)

    cv2.imshow('Face Detector App', frame)
    key = cv2.waitKey(1)

    # Stop if Q is pressed
    if key == 81 or key == 113:
        break

# Release the VideoCapture Object
webcam.release()


# # Detect Faces
# face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

# # Draw rectange around the face
# for (x, y, w, h) in face_coordinates:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 4)




# # To show the image with the faces
# cv2.imshow("Face Detector App", img)
# cv2.waitKey()

# # Testing if the code's working or not
# print('Code Completed')

import cv2
import os



# Create our body classifier


# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    gray = cv2.cvtColor(cap,cv2.COLOR_BGR2GRAY)
    # Pass frame to our body classifier
    body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')
    
    bodies = body_classifier.detectMultiScale(gray,1,2,3)
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in bodies:
        cv2.rectangle(cap,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('cap',cap)
    cv2.waitkey(0)

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()

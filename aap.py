import numpy as np
import cv2
import time
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
cap = cv2.VideoCapture(0) # for default camera
count = 0
while 1:
    ret, img = cap.read()
    #convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    #First detect face and then look for eyes inside the face.
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # to show image and saving in every 1 min 
    cv2.imshow('img',img)
    if count%600==0:
        t = time.strftime("%Y-%m-%d_%H-%M-%S")
        print("Image "+t+" saved")
        file = 'D:/face_eye/data/'+t+'.jpg'
        cv2.imwrite(file,img)
        count +=1

    k = cv2.waitKey(30) & 0xff
    #Press Esc to stop the video
    if k == 27:      
        break

cap.release()
cv2.destroyAllWindows()
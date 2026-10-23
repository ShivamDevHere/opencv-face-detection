import cv2

# ====Model Imported====
face_cascade = cv2.CascadeClassifier("../model/haarcascade_frontalface_default.xml")

# ====Video Capturing====
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

# ====Converting into Gray====
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1.1: if detects: 1 else: 10% zooms frame everytime it fails, -> 10, 20, 30, etc 
    # 5: test 5 times a frame, majority is choosen, if +: 1, else: skip frame.
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)     
    

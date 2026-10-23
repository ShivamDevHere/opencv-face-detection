import cv2

face_cascade = cv2.CascadeClassifier("../model/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
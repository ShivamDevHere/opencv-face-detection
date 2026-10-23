import cv2
from pathlib import Path
import cv2

model_path = Path(__file__).resolve().parent.parent / "model"

face_cascade = cv2.CascadeClassifier(str(model_path / "haarcascade_frontalface_default.xml"))
eye_cascade = cv2.CascadeClassifier(str(model_path / "haarcascade_eye.xml"))
issmile_cascade = cv2.CascadeClassifier(str(model_path / "haarcascade_smile.xml"))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)     
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        eyes = eye_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(15, 15)
        )

        if len(eyes) > 0:
            cv2.putText(frame, "Eye detected", (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,0), 2)
        # print(eye_cascade.empty())
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

        issmile = issmile_cascade.detectMultiScale(roi_gray, 1.7, 10)
        if len(issmile) > 0:
            cv2.putText(frame, "yes, Smilling hehe", (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,0), 2)
    
    if not ret:
        print("Coudn't Read Frame")
        break

    cv2.imshow("Webcame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exit")
        break

cap.release()
cv2.destroyAllWindows()

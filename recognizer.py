import cv2
import json
import os

def load_names(json_path="user_data.json"):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return {}

def recognize_faces(model_path="trainer.yml", user_map_path="user_data.json"):
    if not os.path.exists(model_path):
        print("❌ Train model first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    user_map = load_names(user_map_path)

    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 4)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            id_, conf = recognizer.predict(face_img)

            name = user_map.get(str(id_), "Unknown")
            label = f"{name} ({round(100 - conf)}%)" if conf < 100 else "Unknown"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow('Recognizing Face - Press Q to exit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

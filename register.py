import cv2
import os
import json

def capture_faces(user_id, user_name, save_dir="dataset"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            filename = f"{save_dir}/User.{user_id}.{count}.jpg"
            cv2.imwrite(filename, face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Register - Press Q to Exit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Captured {count} face samples for {user_name}")
        # Save user name in user_data.json
    user_data_file = "user_data.json"
    user_map = {}
    if os.path.exists(user_data_file):
        with open(user_data_file, "r") as f:
            user_map = json.load(f)

    user_map[str(user_id)] = user_name
    with open(user_data_file, "w") as f:
        json.dump(user_map, f)


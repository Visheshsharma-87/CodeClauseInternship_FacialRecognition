import cv2
import numpy as np
import os

def train_model(dataset_path="dataset", model_save="trainer.yml"):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = [], []

    image_files = [f for f in os.listdir(dataset_path) if f.endswith(".jpg")]

    if len(image_files) == 0:
        print("❌ No images found in dataset/. Please register faces first.")
        return

    for img in image_files:
        path = os.path.join(dataset_path, img)
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"⚠️ Skipping unreadable image: {img}")
            continue
        try:
            id = int(img.split('.')[1])
            faces.append(gray)
            ids.append(id)
        except:
            print(f"⚠️ Skipping bad file name: {img}")

    if len(faces) == 0:
        print("❌ No valid face data to train.")
        return

    recognizer.train(faces, np.array(ids))
    recognizer.save(model_save)
    print(f"✅ Model trained and saved as {model_save}")

if __name__ == "__main__":
    train_model()

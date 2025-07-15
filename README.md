# Facial Recognition System using OpenCV

This is a face recognition system built using Python and OpenCV. It allows users to register their face, train a model, and recognize the face in real-time using webcam input. The system uses LBPH (Local Binary Pattern Histogram) algorithm and displays the person's name with confidence.

---

## Technologies Used

- Python 3
- OpenCV (cv2)
- Tkinter (GUI)
- Pillow (Image Display)
- JSON (to map user ID with name)

---

## Folder Structure

FacialRecognitionSystem/
├── main.py                 → GUI entry point  
├── register.py             → Register new face via webcam  
├── trainer.py              → Train the face recognizer  
├── recognizer.py           → Recognize face using trained model  
├── dataset/                → Captured face samples  
├── trainer.yml             → Trained model file  
├── user_data.json          → Stores user ID and name mapping  
├── output/                 → (Optional: image/video logs)  
├── requirements.txt        → Required Python packages  
└── README.md               → Project documentation  

---

## How to Use

### 1. Install Requirements

### 2. Run GUI

### 3. From GUI:
- Click on **Register Face**  
  - Enter a numeric ID (e.g., 1)  
  - Enter your name (e.g., Vishesh Sharma)  
  - Capture 50 face samples via webcam  
- Click on **Train Model**  
  - Model will be trained and saved as `trainer.yml`  
- Click on **Recognize Face**  
  - Webcam will recognize your face and display your name with confidence percentage

---

## Notes

- `dataset/` folder must contain enough face images for accurate training.
- The `user_data.json` file is automatically created/updated when registering new users.
- Make sure lighting is good during face capture.
- Works completely offline.

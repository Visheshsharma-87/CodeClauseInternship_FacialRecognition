import tkinter as tk
from tkinter import messagebox, simpledialog
from register import capture_faces
from trainer import train_model
from recognizer import recognize_faces

def gui():
    window = tk.Tk()
    window.title("Facial Recognition System")
    window.geometry("400x300")

    def register_user():
        user_id = simpledialog.askstring("Input", "Enter numeric User ID:")
        user_name = simpledialog.askstring("Input", "Enter your name:")
        if user_id and user_name:
            capture_faces(user_id, user_name)
        else:
            messagebox.showerror("Error", "Invalid input")

    def train():
        train_model()
        messagebox.showinfo("Training", "Model trained successfully")

    def recognize():
        recognize_faces()

    tk.Button(window, text="Register Face", command=register_user, width=25, height=2).pack(pady=10)
    tk.Button(window, text="Train Model", command=train, width=25, height=2).pack(pady=10)
    tk.Button(window, text="Recognize Face", command=recognize, width=25, height=2).pack(pady=10)

    window.mainloop()

if __name__ == "__main__":
    gui()

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

# Set up face recognizer & detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Where to save trained data and face images
DATASET_DIR = "face_dataset"
MODEL_PATH = "face_model.xml"
os.makedirs(DATASET_DIR, exist_ok=True)

# Utility: get face from image file
def get_face(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face_img = img[y:y+h, x:x+w]
    return cv2.resize(face_img, (200, 200))

# Utility: load dataset
def load_dataset():
    faces, labels, label_names = [], [], {}
    current_label = 0
    for name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, name)
        if not os.path.isdir(person_dir): continue
        label_names[current_label] = name
        for fname in os.listdir(person_dir):
            fpath = os.path.join(person_dir, fname)
            face = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if face is not None:
                faces.append(face)
                labels.append(current_label)
        current_label += 1
    return faces, np.array(labels), label_names

# Train or load model if possible
def train_model():
    faces, labels, label_names = load_dataset()
    if len(faces) > 0:
        recognizer.train(faces, labels)
        recognizer.save(MODEL_PATH)
    return label_names

def load_model():
    faces, labels, label_names = load_dataset()
    if os.path.exists(MODEL_PATH) and len(faces)>0:
        recognizer.read(MODEL_PATH)
        return label_names
    elif len(faces)>0:
        return train_model()
    else:
        return {}

# --- Tkinter UI starts here ---
class FaceUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenCV Face Recognition Demo")
        self.last_file = None
        self.label_names = load_model()
        self.setup_ui()

    def setup_ui(self):
        self.frame = tk.Frame(self.root)
        self.frame.pack(padx=10, pady=10)

        self.image_label = tk.Label(self.frame)
        self.image_label.grid(row=0, column=0, columnspan=2)

        tk.Button(self.frame, text="Add Person", command=self.add_person).grid(row=1, column=0, pady=10)
        tk.Button(self.frame, text="Recognize", command=self.recognize).grid(row=1, column=1, pady=10)
        self.result_label = tk.Label(self.frame, text="")
        self.result_label.grid(row=2, column=0, columnspan=2, pady=10)

    def show_image(self, file_path):
        from PIL import Image, ImageTk
        img = Image.open(file_path)
        img.thumbnail((200, 200))
        img = ImageTk.PhotoImage(img)
        self.image_label.image = img
        self.image_label.config(image=img)

    def add_person(self):
        file_path = filedialog.askopenfilename(title="Select Face Image")
        if not file_path: return
        face = get_face(file_path)
        if face is None:
            messagebox.showerror("Error", "No face detected in image.")
            return
        name = simpledialog.askstring("Name", "Enter person's name:")
        if not name: return
        person_dir = os.path.join(DATASET_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
        # Save the face
        idx = len(os.listdir(person_dir)) + 1
        face_path = os.path.join(person_dir, f"{idx}.png")
        cv2.imwrite(face_path, face)
        self.label_names = train_model()
        messagebox.showinfo("Success", f"Image added for {name}")

    def recognize(self):
        file_path = filedialog.askopenfilename(title="Select Image to Recognize")
        if not file_path: return
        face = get_face(file_path)
        if face is None:
            messagebox.showerror("Error", "No face detected in image.")
            return
        label, confidence = recognizer.predict(face)
        name = self.label_names.get(label, "Unknown")
        self.result_label.config(text=f"Prediction: {name} (confidence: {confidence:.1f})")
        try: # Show with PIL if available
            self.show_image(file_path)
        except:
            pass

if __name__ == "__main__":
    try:
        from PIL import Image, ImageTk
    except ImportError:
        import sys
        print("Pillow is required for image display in the UI. Install with 'pip install pillow'")
        sys.exit(1)
    root = tk.Tk()
    app = FaceUI(root)
    root.mainloop()
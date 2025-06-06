
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from src.model import EmotionCNN
from torchvision import transforms
import webbrowser

# Emotion classes (make sure they match your training)
emotion_labels = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# Optional: Music URLs per emotion
music_links = {
    "angry": "https://music.youtube.com/search?q=calm+down",
    "disgust": "https://music.youtube.com/search?q=clean+vibes",
    "scared": "https://music.youtube.com/search?q=confidence+music",
    "happy": "https://music.youtube.com/search?q=happy+music",
    "sad": "https://music.youtube.com/search?q=cheer+up+music",
    "surprised": "https://music.youtube.com/search?q=energetic+music",
    "neutral": "https://music.youtube.com/search?q=background+lofi"
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = EmotionCNN().to(device)
model.load_state_dict(torch.load("checkpoints/emotion_cnn.pth", map_location=device))
model.eval()

# Preprocessing transform (must match training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

last_emotion = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (48, 48))
        roi_normalized = transform(roi_resized).unsqueeze(0).to(device)  # shape: [1, 1, 48, 48]

        with torch.no_grad():
            output = model(roi_normalized)
            probs = F.softmax(output, dim=1)
            pred = torch.argmax(probs).item()
            label = emotion_labels[pred]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if label != last_emotion:
            print(f"Detected emotion: {label}")
            webbrowser.open(music_links[label])
            last_emotion = label

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

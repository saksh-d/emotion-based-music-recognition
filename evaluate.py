
import torch
import torch.nn as nn
from src.dataset import get_dataloaders
from src.model import EmotionCNN
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model_path = 'checkpoints/emotion_cnn.pth'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    #load test_loader
    _, _, test_loader = get_dataloaders("data/fer2013.csv")

    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=[
        "angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"
    ]))

    #plot the confusion matrix

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=[
        "angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"
    ], yticklabels=[
        "angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"
    ], cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_model()

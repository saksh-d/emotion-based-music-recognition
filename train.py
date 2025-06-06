
import torch
import torch.nn as nn
import torch.optim as optim
import os
from src.dataset import get_dataloaders
from src.model import EmotionCNN

def train_model(epochs=20, 
                batch_size=32, 
                lr=0.001, 
                model_save_path='checkpoints/emotion_cnn.pth'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_dataloaders("data/fer2013.csv", batch_size=batch_size)

    model = EmotionCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(),
                           lr = lr,
                           weight_decay=1e-4)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            #fwd pass
            y_pred = model(images)

            #loss
            loss = loss_fn(y_pred, labels)
            total_loss += loss.item()

            _, predicted = torch.max(y_pred, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #get avg loss
        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        #validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                y_pred_val = model(images)
                _, predicted = torch.max(y_pred_val, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

            val_acc = val_correct / val_total

            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

            # Save best model
        if val_acc > best_val_acc:
            print("New best model found. Saving...")
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    train_model()

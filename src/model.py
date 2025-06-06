
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchinfo import summary

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()

        #conv block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, 
                      out_channels=32,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, 
                         stride=2) #This will yield 24x24
        )

        #conv block 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, 
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, 
                         stride=2) #This will yield 12x12
        )

        #conv block 3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels = 64, 
                      out_channels=96,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, 
                         stride=2) #This will yield 6x6
        )

        self.dropout = nn.Dropout(0.6)
        self.fc1 = nn.Linear(96*6*6, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        return x

#print model summary using torchinfo.summary
model = EmotionCNN()
batch_size = 32
summary(model, input_size=(batch_size, 1, 48, 48))



import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class FER2013Dataset(Dataset): #important to overwrite __len__ and __getitem__
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pixels = np.fromstring(self.df.iloc[idx]['pixels'], sep=' ', dtype=np.float32)
        image = pixels.reshape(48,48) #FER resolution
        image = np.expand_dims(image, axis=0) #image_size = (1,48,48)

        if self.transform:
            image = self.transform(torch.tensor(image))

        label = int(self.df.iloc[idx]['emotion'])
        return image, label

def get_dataloaders(csv_path, batch_size=64):
    df = pd.read_csv(csv_path)

    #Relabeling to match inference labels
    #label_mapping = {0: 'angry', 1: 'disgust', 2: 'scared', 3: 'happy', 4: 'sad', 5: 'surprised', 6: 'neutral'}
    #df['emotion'] = df['emotion'].map({k: v for k, v in label_mapping.items()})

    # transform = transforms.Compose([
    #     transforms.Normalize(mean=[0.5], std=[0.5])
    # ])

    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

    def df_by_usage(usage_type):
        return df[df['Usage'] == usage_type]

    train_set = FER2013Dataset(df_by_usage("Training"), transform=transform)
    val_set = FER2013Dataset(df_by_usage("PublicTest"), transform=transform)
    test_set = FER2013Dataset(df_by_usage("PrivateTest"), transform=transform)

    #create dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


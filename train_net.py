import os
import cv2
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from torchvision.transforms import ColorJitter, ToTensor, Normalize

# Define custom modules and classes
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.transpose(3, 2).contiguous()
        return x.view(x.size(0), -1)

class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, 3, 1)),
            ('prelu1', nn.PReLU(32)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),
            ('conv2', nn.Conv2d(32, 64, 3, 1)),
            ('prelu2', nn.PReLU(64)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),
            ('conv3', nn.Conv2d(64, 64, 3, 1)),
            ('prelu3', nn.PReLU(64)),
            ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),
            ('conv4', nn.Conv2d(64, 128, 2, 1)),
            ('prelu4', nn.PReLU(128)),
            ('flatten', Flatten()),
            ('conv5', nn.Linear(1152, 256)),
            ('drop5', nn.Dropout(0.25)),
            ('prelu5', nn.PReLU(256)),
        ]))

        # Output layer with 136 features
        self.output_layer = nn.Linear(256, 136)

    def forward(self, x):
        x = self.features(x)
        landmarks = self.output_layer(x)
        return landmarks

class LandmarksDataset(Dataset):
    def __init__(self, root_dir, landmarks_file, transform=None):
        self.root_dir = root_dir
        self.landmarks_file = landmarks_file
        self.transform = transform
        self.landmarks = self._load_landmarks()

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        img_name, landmarks = self.landmarks[idx]
        landmarks = np.array(landmarks).reshape(-1, 2)
        img_path = os.path.join(self.root_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transforms.ToPILImage()(image)
        if self.transform:
            image = self.transform(image)
        else:
            image = ToTensor()(image)
            image = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)
        landmarks = torch.flatten(torch.tensor(landmarks, dtype=torch.float32))
        return image, landmarks

    def _load_landmarks(self):
        landmarks = []
        with open(self.landmarks_file, 'r') as f:
            for line in f:
                data = line.strip().split()
                img_name = data[0]
                landmarks_data = [float(coord) for coord in data[1:]]
                landmarks.append((img_name, landmarks_data))
        return landmarks

# Define loss function
class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, predicted_landmarks, target_landmarks):
        mask = (target_landmarks != -1).float()
        loss = torch.sum((predicted_landmarks - target_landmarks)**2 * mask) / torch.sum(mask)
        return loss

def main(args):
    # Define transforms
    color_transform = transforms.Compose([
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Load datasets
    dataset_train = LandmarksDataset(args.train_root_dir, args.train_landmarks_file, transform=color_transform)
    dataset_test = LandmarksDataset(args.test_root_dir, args.test_landmarks_file, transform=None)

    # Define data loaders
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size_test, shuffle=False)

    # Initialize model
    model = ONet()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = CustomMSELoss()

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        for images, landmarks in tqdm(dataloader_train):
            images, landmarks = images.to(device), landmarks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, landmarks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(dataloader_train)
        print(f'Epoch {epoch+1}/{args.num_epochs}, Training Loss: {train_loss}')

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, landmarks in dataloader_test:
                images, landmarks = images.to(device), landmarks.to(device)
                outputs = model(images)
                test_loss += criterion(outputs, landmarks).item()

        test_loss /= len(dataloader_test)
        print(f'Epoch {epoch+1}/{args.num_epochs}, Test Loss: {test_loss}')

# Define command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Face Landmarks Detection Training")

    # Dataset paths
    parser.add_argument('--train_root_dir', type=str, default='./cropped_faces', help="Path to train dataset root directory")
    parser.add_argument('--train_landmarks_file', type=str, default='landmarks.txt', help="Path to train dataset landmarks file")
    parser.add_argument('--test_root_dir', type=str, default='./cropped_faces_test', help="Path to test dataset root directory")
    parser.add_argument('--test_landmarks_file', type=str, default='landmarks_test.txt', help="Path to test dataset landmarks file")

    # Model hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help="Training batch size")
    parser.add_argument('--batch_size_test', type=int, default=1024, help="Test batch size")
    parser.add_argument('--learning_rate', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of training epochs")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
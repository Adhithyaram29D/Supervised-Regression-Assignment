# Supervised-Regression-Assignment

### Problem Statement:
Using Deep Learning techniques, predict the coordinates (x,y) of a pixel which has a value of
255 for 1 pixel in a given 50x50 pixel grayscale image and all other pixels are 0. The pixel with a
value of 255 is randomly assigned. You may generate a dataset as required for solving the
problem. Please explain your rationale behind dataset choices.

### Approach Overview:
Dataset Generation

A synthetic dataset is created since no real-world data is available.
Each sample is a 50×50 grayscale image with exactly one pixel set to 255 and all remaining pixels set to 0.
The corresponding label is the normalized (x, y) coordinate of the bright pixel.

Model Selection

A Convolutional Neural Network (CNN) is used to learn spatial patterns in the image.
The model takes a 50×50 image as input and outputs two continuous values representing the pixel’s (x, y) coordinates.

Loss Function

Mean Squared Error (MSE) is used as the task involves coordinate regression.

## Program

```python

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class PixelDataset(Dataset):


    def __init__(self, num_samples=5000):
        self.images = []
        self.labels = []

        for _ in range(num_samples):
            image = np.zeros((50, 50), dtype=np.float32)

            x = np.random.randint(0, 50)
            y = np.random.randint(0, 50)

            image[y, x] = 255.0

            
            image /= 255.0

            self.images.append(image)
            self.labels.append([x / 49.0, y / 49.0])

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx]).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


dataset = PixelDataset(num_samples=5000)

train_idx, val_idx = train_test_split(
    np.arange(len(dataset)), test_size=0.2, random_state=42
)

train_data = torch.utils.data.Subset(dataset, train_idx)
val_data = torch.utils.data.Subset(dataset, val_idx)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)


class PixelLocatorCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 12 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PixelLocatorCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 15
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_loss:.6f} "
        f"Val Loss: {val_loss:.6f}"
    )


plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid()
plt.show()


model.eval()

images, labels = next(iter(val_loader))
images = images.to(device)
labels = labels.to(device)

with torch.no_grad():
    predictions = model(images)

labels_px = labels.cpu().numpy() * 49
pred_px = predictions.cpu().numpy() * 49

plt.figure(figsize=(6, 6))
plt.scatter(labels_px[:, 0], labels_px[:, 1], c="green", label="Ground Truth")
plt.scatter(pred_px[:, 0], pred_px[:, 1], c="red", marker="x", label="Predicted")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.title("Ground Truth vs Predicted Pixel Coordinates")
plt.legend()
plt.grid()
plt.show()

```

## Output:
<img width="700" height="470" alt="download" src="https://github.com/user-attachments/assets/10ba67f0-55a8-4e79-b9f7-9056766607ef" />
<img width="531" height="547" alt="download" src="https://github.com/user-attachments/assets/f33eb2cc-f2db-45d4-9ad5-6b15de3ca1ea" />



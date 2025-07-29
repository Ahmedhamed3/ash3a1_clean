import os
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class LandmarkDataset(Dataset):
    def __init__(self, csv_file='dataset/all_19_samples_landmarks_clean.csv', img_dir='dataset/xray_images', img_size=224):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        landmarks = [row['MF_x'], row['MF_y'],
                     row['Apex_of_PremolarMolar_x'], row['Apex_of_PremolarMolar_y'],
                     row['IAC_x'], row['IAC_y'],
                     row['LBM_90_x'], row['LBM_90_y']]
        # convert to positive normalized coordinates
        norm = []
        for i, val in enumerate(landmarks):
            val = -val
            if i % 2 == 0:
                norm.append(val / w)
            else:
                norm.append(val / h)
        image = self.transform(image)
        return image, torch.tensor(norm, dtype=torch.float32)

def split_dataset(dataset, val_ratio=0.2):
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    return torch.utils.data.random_split(dataset, [train_size, val_size])

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, 8)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def train_model(model, loader, device, epochs=20):
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optim.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, targets)
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * imgs.size(0)
        print(f'Epoch {epoch+1}: loss={epoch_loss/len(loader.dataset):.4f}')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = LandmarkDataset()
    train_ds, val_ds = split_dataset(dataset)
    loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    model = SimpleCNN().to(device)
    train_model(model, loader, device)
    torch.save(model.state_dict(), 'landmark_model.pth')
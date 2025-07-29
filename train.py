import json
import base64
import zlib
from io import BytesIO
import os
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import numpy as np

class MandibleDataset(Dataset):
    def __init__(self, img_dir, ann_dir, img_size=(256, 256)):
        self.img_paths = sorted(glob(os.path.join(img_dir, '*.png')), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.ann_paths = [os.path.join(ann_dir, os.path.basename(p) + '.json') for p in self.img_paths]
        self.img_size = img_size
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        with open(self.ann_paths[idx]) as f:
            ann = json.load(f)
        bm = ann['objects'][0]['bitmap']
        submask = Image.open(BytesIO(zlib.decompress(base64.b64decode(bm['data']))))
        fullmask = Image.new('L', (ann['size']['width'], ann['size']['height']), 0)
        fullmask.paste(submask, tuple(bm['origin']))

        img = img.resize(self.img_size, Image.BILINEAR)
        fullmask = fullmask.resize(self.img_size, Image.NEAREST)

        img = self.to_tensor(img)
        mask = torch.from_numpy(np.array(fullmask, dtype=np.int64) // 255)
        return img, mask

def train(model, loader, device):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(2):
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            out = model(imgs)['out']
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1} done')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MandibleDataset('Segmentation1/img', 'Segmentation1/ann')
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = deeplabv3_resnet50(num_classes=2)
    model.to(device)
    train(model, loader, device)
    torch.save(model.state_dict(), 'my_model.pth')

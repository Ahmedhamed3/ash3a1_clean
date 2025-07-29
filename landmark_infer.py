import os
from PIL import Image
import torch
from torchvision import transforms
from landmark_train import SimpleCNN


def load_model(weight_path, device):
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model


def predict(image_path, model, device):
    image = Image.open(image_path).convert('RGB')
    w, h = image.size
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor).squeeze(0).cpu()
    # convert back to original coordinate scale (negative values)
    preds = []
    for i, val in enumerate(out):
        if i % 2 == 0:
            preds.append(-float(val) * w)
        else:
            preds.append(-float(val) * h)
    return preds


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predict landmarks')
    parser.add_argument('image', help='Path to X-ray image')
    parser.add_argument('--weights', default='landmark_model_v2.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.weights, device)
    preds = predict(args.image, model, device)
    names = ['MF_x', 'MF_y', 'Apex_x', 'Apex_y', 'IAC_x', 'IAC_y', 'LBM_90_x', 'LBM_90_y']
    for name, val in zip(names, preds):
        print(f'{name}: {val:.2f}')
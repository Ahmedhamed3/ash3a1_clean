import os
from glob import glob
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model and load weights
    model = deeplabv3_resnet50(num_classes=2)
    model.load_state_dict(torch.load('my_model.pth', map_location=device))
    model.to(device)
    model.eval()

    img_dir = 'Segmentation1/img'
    pred_dir = 'predictions'
    os.makedirs(pred_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
    ])

    to_pil = transforms.ToPILImage()

    img_paths = sorted(glob(os.path.join(img_dir, '*.png')),
                       key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    with torch.no_grad():
        for path in img_paths:
            img = Image.open(path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            output = model(tensor)['out']
            pred = output.argmax(dim=1).squeeze(0).byte() * 255
            pred_img = to_pil(pred.cpu())
            pred_img.save(os.path.join(pred_dir, os.path.basename(path)))


if __name__ == '__main__':
    main()
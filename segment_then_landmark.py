import argparse
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from landmark_train import SimpleCNN


def load_segmentation_model(weight_path, device):
    model = deeplabv3_resnet50(num_classes=2)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_landmark_model(weight_path, device):
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model


def segment_jaw(image, model, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)["out"]
        mask = out.argmax(dim=1).squeeze(0).cpu().byte().numpy()
    mask_img = Image.fromarray(mask * 255).resize(image.size, Image.NEAREST)
    return mask_img


def crop_to_mask(image, mask):
    bbox = mask.getbbox()
    if bbox is None:
        return image, (0, 0), image.size
    cropped = image.crop(bbox)
    return cropped, (bbox[0], bbox[1]), (bbox[2] - bbox[0], bbox[3] - bbox[1])


def predict_landmarks(image, model, device):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor).squeeze(0).cpu()
    return out


def main(image_path, seg_weights, landmark_weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_model = load_segmentation_model(seg_weights, device)
    landmark_model = load_landmark_model(landmark_weights, device)

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    mask = segment_jaw(image, seg_model, device)
    # Debug: save raw segmentation mask
    mask.save("debug_mask.png")

    # Determine crop region and show bbox
    cropped, offset, crop_size = crop_to_mask(image, mask)
    print(f"Mask bounding box: {offset[0], offset[1], offset[0]+crop_size[0], offset[1]+crop_size[1]}")

    # Debug: save cropped region
    cropped.save("debug_cropped.png")

    preds = predict_landmarks(cropped, landmark_model, device)
    print("Normalized landmark predictions:", preds.numpy().tolist())

    preds_np = []
    for i, val in enumerate(preds):
        if i % 2 == 0:
            coord = -float(val) * crop_size[0] + offset[0]
        else:
            coord = -float(val) * crop_size[1] + offset[1]
        preds_np.append(coord)
    print("Absolute landmark predictions:", preds_np)

    names = ["MF_x", "MF_y", "Apex_x", "Apex_y", "IAC_x", "IAC_y", "LBM_90_x", "LBM_90_y"]
    for name, val in zip(names, preds_np):
        print(f"{name}: {val:.2f}")

    # Debug overlay with predicted landmarks
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    for x, y in zip(preds_np[::2], preds_np[1::2]):
        x = -x
        y = -y
        draw.line((x - 5, y - 5, x + 5, y + 5), fill="red", width=2)
        draw.line((x - 5, y + 5, x + 5, y - 5), fill="red", width=2)
    overlay.save("debug_overlay_check.png")
    # Save a final overlay as well
    overlay.save("overlay.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment jaw then predict landmarks")
    parser.add_argument("image", help="Path to input X-ray image")
    parser.add_argument("--seg_weights", default="my_model.pth", help="Path to segmentation weights")
    parser.add_argument("--landmark_weights", default="landmark_model_v2.pth", help="Path to landmark weights")
    args = parser.parse_args()

    main(args.image, args.seg_weights, args.landmark_weights)

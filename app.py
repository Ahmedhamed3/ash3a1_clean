import streamlit as st
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = deeplabv3_resnet50(num_classes=2)
    model.load_state_dict(torch.load("my_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device


def create_overlay(image: Image.Image, mask: np.ndarray, color=(255, 0, 0), alpha=0.5) -> Image.Image:
    image = image.convert("RGB")
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255).resize(image.size, Image.NEAREST)
    img_arr = np.array(image, dtype=np.float32)
    mask_arr = np.array(mask_img)
    overlay_color = np.array(color, dtype=np.float32)
    mask_bool = mask_arr > 127
    img_arr[mask_bool] = (1 - alpha) * img_arr[mask_bool] + alpha * overlay_color
    return Image.fromarray(img_arr.astype(np.uint8))


def main():
    st.title("Dental X-ray Segmentation")
    model, device = load_model()

    file = st.file_uploader("Upload a dental X-ray", type=["png", "jpg", "jpeg"])
    if file is not None:
        image = Image.open(file).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
        ])
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)["out"]
            mask = output.argmax(dim=1).squeeze(0).cpu().byte().numpy()

        overlay = create_overlay(image, mask)

        col1, col2 = st.columns(2)
        with col1:
            st.header("Original")
            st.image(image, use_column_width=True)
        with col2:
            st.header("Prediction")
            st.image(overlay, use_column_width=True)


if __name__ == "__main__":
    main()
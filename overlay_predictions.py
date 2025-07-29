import os
from glob import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse


def create_overlay_image(image, mask, color=(255, 0, 0), alpha=0.5):
    """Return an image with `mask` overlayed on `image` in the given `color`."""
    image = image.convert("RGB")
    mask = mask.convert("L")
    mask = mask.resize(image.size, Image.NEAREST)

    img_arr = np.array(image, dtype=np.float32)
    mask_arr = np.array(mask)

    overlay_color = np.array(color, dtype=np.float32)
    mask_bool = mask_arr > 127

    img_arr[mask_bool] = (
        (1 - alpha) * img_arr[mask_bool] + alpha * overlay_color
    )

    return Image.fromarray(img_arr.astype(np.uint8))


def main(img_dir="Segmentation1/img", pred_dir="predictions", output_dir="overlays", show=False):
    os.makedirs(output_dir, exist_ok=True)

    img_paths = sorted(
        glob(os.path.join(img_dir, "*.png")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )

    for img_path in img_paths:
        pred_path = os.path.join(pred_dir, os.path.basename(img_path))
        if not os.path.exists(pred_path):
            print(f"Prediction not found for {img_path}, skipping")
            continue

        img = Image.open(img_path)
        mask = Image.open(pred_path)
        overlay = create_overlay_image(img, mask)
        out_path = os.path.join(output_dir, os.path.basename(img_path))
        overlay.save(out_path)

        if show:
            plt.figure()
            plt.imshow(overlay)
            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay predicted masks on images")
    parser.add_argument("--show", action="store_true", help="Display overlays instead of only saving")
    parser.add_argument("--img_dir", default="Segmentation1/img", help="Directory with original images")
    parser.add_argument("--pred_dir", default="predictions", help="Directory with predicted masks")
    parser.add_argument("--output_dir", default="overlays", help="Where to save overlay images")
    args = parser.parse_args()

    main(img_dir=args.img_dir, pred_dir=args.pred_dir, output_dir=args.output_dir, show=args.show)
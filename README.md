# Panoramic Dental X-rays With Segmented Mandibles
This repository provides a dataset of panoramic dental X‑ray images with segmentation masks for mandibles. You can use it for tasks such as instance segmentation, semantic segmentation and object detection.

## Dataset structure
- `Segmentation1` and `Segmentation2` contain the same dataset split into two folders.
  - `img/` holds the X‑ray images in PNG format.
  - `ann/` holds the annotation files in Supervisely JSON format.
- `meta.json` describes the dataset classes.

## Quick start
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ASH3A
   ```
2. Install the Python packages needed for the example below:
   ```bash
   pip install pillow matplotlib numpy
   ```
3. Run the following Python snippet to visualize an image with its segmentation mask:
   ```python
   import json, base64, zlib
   from io import BytesIO
   from PIL import Image
   import numpy as np
   import matplotlib.pyplot as plt

   img = Image.open("Segmentation1/img/1.png")
   with open("Segmentation1/ann/1.png.json") as f:
       ann = json.load(f)

   bitmap = ann["objects"][0]["bitmap"]
   mask = Image.open(BytesIO(zlib.decompress(base64.b64decode(bitmap["data"]))))
   mask = mask.resize(img.size)

   plt.figure()
   plt.imshow(img)
   plt.imshow(np.array(mask), alpha=0.5)
   plt.axis("off")
   plt.show()
   ```

The dataset is released under the [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/) license as noted in `LICENSE.md`.


## Visualize predictions
Once you have saved your model outputs in the `predictions/` folder (for example by running `predict.py`), you can overlay them on the original images with:
```bash
python overlay_predictions.py --pred_dir predictions --output_dir overlays
```
The resulting images will be written to `overlays/`.

## Segment then detect landmarks
Use `segment_then_landmark.py` to first segment the mandible and crop the image
before running the landmark model:

```bash
python segment_then_landmark.py path/to/image.jpg \
    --seg_weights my_model.pth --landmark_weights landmark_model_v2.pth
```
The script prints the predicted landmark coordinates relative to the original
image.
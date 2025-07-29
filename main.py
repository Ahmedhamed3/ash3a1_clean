import json, base64, zlib
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = Image.open("Segmentation1/img/10.png")

# Load and decode the annotation
with open("Segmentation1/ann/10.png.json") as f:
    ann = json.load(f)
bitmap = ann["objects"][0]["bitmap"]
mask = Image.open(BytesIO(zlib.decompress(base64.b64decode(bitmap["data"]))))
mask = mask.resize(img.size)

# Display
plt.figure()
plt.imshow(img)
plt.imshow(np.array(mask), alpha=0.5, cmap="Reds")
plt.axis("off")
plt.show()

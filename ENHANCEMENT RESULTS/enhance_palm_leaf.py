import sys
import os
import torch
import cv2
import numpy as np
from torchvision.transforms.functional import to_tensor, to_pil_image

# ✅ Add FSRCNN-Pytorch directory to sys.path to import neuralnet
fsrcnn_path = r"F:\mini project 25\image enhancement\final\FSRCNN-Pytorch"
sys.path.append(fsrcnn_path)

# ✅ Import FSRCNN model
from neuralnet import FSRCNN_model  

# ✅ Define Paths
INPUT_IMAGE_PATH = r"F:\mini project 25\image enhancement\final\input\binarise_t.jpg"  # Change if needed
OUTPUT_IMAGE_PATH = r"F:\mini project 25\image enhancement\final\output\enhanced_t1.jpg"  # Change if needed
MODEL_PATH = r"F:\mini project 25\image enhancement\final\FSRCNN-Pytorch\checkpoint\x4\FSRCNN-x4.pt"  # Using x4 model

# ✅ Check if the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Error: Model file not found at '{MODEL_PATH}'")

# ✅ Load FSRCNN model
model = FSRCNN_model(scale=4)  # Scale factor x4
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

print(f"✅ Using FSRCNN model: {MODEL_PATH} (Scale x4)")

# ✅ Read input image (Convert Grayscale to RGB)
image = cv2.imread(INPUT_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)  # Load as Grayscale
if image is None:
    raise FileNotFoundError(f"❌ Error: Unable to load input image '{INPUT_IMAGE_PATH}'")

# ✅ Convert 1-channel grayscale image to 3-channel RGB
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

# Convert image to tensor
image = to_tensor(image).unsqueeze(0)  # Add batch dimension

# ✅ Apply Super-Resolution
with torch.no_grad():
    sr_image = model(image)

# Convert back to image format
sr_image = sr_image.squeeze(0).clamp(0, 1)  # Remove batch, ensure values are between [0,1]
sr_image = to_pil_image(sr_image)

# ✅ Save Output Image
os.makedirs(os.path.dirname(OUTPUT_IMAGE_PATH), exist_ok=True)  # Ensure output directory exists
sr_image.save(OUTPUT_IMAGE_PATH)

print(f"✅ Enhanced image saved at: {OUTPUT_IMAGE_PATH}")

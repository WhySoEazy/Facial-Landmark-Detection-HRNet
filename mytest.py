import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import lib.models as models
from lib.utils import utils
import matplotlib.pyplot as plt
from lib.config import config, update_config

# Step 1: Load the HRNet model
model = models.get_face_alignment_net(config)
# Load the pre-trained weights (you need to download the weights)
checkpoint = torch.load('HR18-WFLW.pth', map_location=torch.device('cpu'))
model.eval()

# Step 2: Prepare input images
# Example: Load a sample image
image_path = "C:/Users/Asus/Desktop/wife.jpg"
input_image = Image.open(image_path)

# Step 3: Preprocess the input image
transform = transforms.Compose([
    transforms.Resize((256 , 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
input_tensor = transform(input_image).unsqueeze(0)

# Step 4: Perform inference
with torch.no_grad():
    output = model(input_tensor)

# Step 5: Postprocess the output to obtain facial landmarks
landmarks = utils.get_preds(output)[-1].squeeze().cpu().numpy()

# Step 6: Visualize the results
plt.figure(figsize=(8, 8))
plt.imshow(input_image)
plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
plt.axis('off')
plt.show()

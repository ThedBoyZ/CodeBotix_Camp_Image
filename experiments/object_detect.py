import torch
from torchvision import models
from torchvision.transforms import functional as F
from PIL import Image

# Load the pre-trained PyTorch model
model = models.detection.retinanet_resnet50_fpn(pretrained=True)
model.eval()

# Load and preprocess the input image
input_image_path = "image.jpg"
input_image = Image.open(input_image_path)
input_tensor = F.to_tensor(input_image)
input_tensor = input_tensor.unsqueeze(0)

# Perform object detection
with torch.no_grad():
    predictions = model(input_tensor)

# Process and print the detected objects
for i in range(predictions[0]['boxes'].shape[0]):
    bbox = predictions[0]['boxes'][i].tolist()
    score = predictions[0]['scores'][i].item()
    label = predictions[0]['labels'][i].item()
    print("Label:", label, "Score:", score, "Bounding Box:", bbox)
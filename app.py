import torch
import torchvision.models as models
from torchvision.models import (
efficientnet_b0,EfficientNet_B0_Weights,
resnet50, ResNet50_Weights, 
vgg16, VGG16_Weights,
mobilenet_v2, MobileNet_V2_Weights, 
densenet121, DenseNet121_Weights)
import requests
import gradio as gr
from backend import * # Import all functions from backend.py

print("Starting Trustworthy AI Application...")

# --- Global Setup (from Cells 1, 2, 29) ---
print("Setting up device (GPU if available)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading ImageNet labels...")
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
response = requests.get(LABELS_URL)
labels = response.json()
print(f"Loaded {len(labels)} labels.")

print("Loading model zoo (this may take a moment)...")
model_zoo = {}

# 1. ResNet50
print("Loading ResNet50...")
resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT).to(device).eval()
model_zoo['ResNet50'] = resnet50

# 2. VGG16
print("Loading VGG16...")
vgg16 = vgg16(weights=VGG16_Weights.DEFAULT).to(device).eval()
model_zoo['VGG16'] = vgg16

# 3. MobileNetV2
print("Loading MobileNetV2...")
mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).to(device).eval()
model_zoo['MobileNetV2'] = mobilenet

# 4. DenseNet121
print("Loading DenseNet121...")
densenet = densenet121(weights=DenseNet121_Weights.DEFAULT).to(device).eval()
model_zoo['DenseNet121'] = densenet

# 5. EfficientNet-B0
print("Loading EfficientNet-B0...")
try:
    efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).to(device).eval()
    model_zoo['EfficientNet-B0'] = efficientnet
except Exception as e:
    print(f"Could not load EfficientNet-B0, skipping: {e}")

print(f"--- All {len(model_zoo)} models loaded ---")

# --- Main execution ---
if __name__ == "__main__":
    print("Creating Gradio interface...")
    # Pass the loaded models, device, and labels to the interface function
    demo = gradio_interface(model_zoo, device, labels)
    
    print("Launching Gradio app... Access the URL below in your browser.")
    # (Cell 35)
    demo.launch(debug=True, share=True)
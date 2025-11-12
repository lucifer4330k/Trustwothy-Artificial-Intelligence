# Imports from Cell 1, 11, 15, 26, 27, 29, 34
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import requests
from io import BytesIO
import warnings
import json
from datetime import datetime
# from google.colab import files  <- THIS LINE IS NOW REMOVED
from IPython.display import display, HTML
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap
import gradio as gr
from scipy import stats

# --- From Cell 2: Helper Functions ---

# Define preprocessing transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Inverse transform for visualization
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

def load_image(image_path_or_url):
    """Load image from path or URL"""
    try:
        if image_path_or_url.startswith('http'):
            response = requests.get(image_path_or_url, stream=True)
            response.raise_for_status()
            img = Image.open(response.raw).convert('RGB')
        else:
            img = Image.open(image_path_or_url).convert('RGB')
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def predict(img_tensor, model, labels, top_k=5):
    """Make prediction and return top-k classes with probabilities"""
    with torch.no_grad():
        img_tensor = img_tensor.to(next(model.parameters()).device) # Use model's device
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)

    top_probs, top_indices = torch.topk(probabilities, top_k)
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]

    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            'class': labels[idx],
            'probability': prob,
            'index': idx
        })

    return results

def display_prediction(img, predictions, title="Predictions"):
    """Display image with top predictions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Display image
    if isinstance(img, torch.Tensor):
        img = inv_normalize(img.cpu()).squeeze().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(title, fontsize=14, fontweight='bold')

    # Display predictions
    classes = [p['class'] for p in predictions]
    probs = [p['probability'] * 100 for p in predictions]
    colors = sns.color_palette("RdYlGn_r", len(classes))

    ax2.barh(classes, probs, color=colors)
    ax2.set_xlabel('Confidence (%)', fontsize=12)
    ax2.set_title('Top 5 Predictions', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)

    for i, (cls, prob) in enumerate(zip(classes, probs)):
        ax2.text(prob + 2, i, f'{prob:.1f}%', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()

# --- From Cell 3: FGSM Attack ---

def fgsm_attack(image, epsilon, data_grad):
    """
    FGSM Attack: Fast Gradient Sign Method
    ... (docstring from cell) ...
    """
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image,
                                   image.min().item(),
                                   image.max().item())
    return perturbed_image

def generate_fgsm_adversarial(model, image_tensor, target_class, epsilon):
    """
    Generate adversarial example using FGSM
    ... (docstring from cell) ...
    """
    image_tensor.requires_grad = True
    output = model(image_tensor)
    loss = F.cross_entropy(output, torch.tensor([target_class]).to(image_tensor.device))
    model.zero_grad()
    loss.backward()
    data_grad = image_tensor.grad.data
    adversarial_image = fgsm_attack(image_tensor, epsilon, data_grad)
    perturbation = adversarial_image - image_tensor
    return adversarial_image.detach(), perturbation.detach()

# --- From Cell 4: PGD Attack ---

def pgd_attack(model, image_tensor, target_class, epsilon, alpha, num_iter):
    """
    PGD Attack: Projected Gradient Descent
    ... (docstring from cell) ...
    """
    original_image = image_tensor.clone().detach()
    adversarial_image = image_tensor.clone().detach()
    device = image_tensor.device

    for i in range(num_iter):
        adversarial_image.requires_grad = True
        output = model(adversarial_image)
        loss = F.cross_entropy(output, torch.tensor([target_class]).to(device))
        model.zero_grad()
        loss.backward()
        data_grad = adversarial_image.grad.data

        with torch.no_grad():
            adversarial_image = adversarial_image + alpha * data_grad.sign()
            perturbation = torch.clamp(adversarial_image - original_image,
                                       -epsilon, epsilon)
            adversarial_image = original_image + perturbation
            adversarial_image = torch.clamp(adversarial_image,
                                           original_image.min().item(),
                                           original_image.max().item())

    perturbation = adversarial_image - original_image
    return adversarial_image.detach(), perturbation.detach()

# --- From Cell 5: Attack Visualization Utilities ---

def visualize_attack_comparison(original_img, adversarial_tensor, perturbation,
                                original_preds, adv_preds, attack_name, epsilon, labels):
    """
    Comprehensive visualization of adversarial attack
    ... (docstring from cell) ...
    """
    fig = plt.figure(figsize=(18, 5))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
    
    # ... (rest of the function code from Cell 5) ...
    # Note: This function uses plt.show(), which will open a window locally.
    # For Gradio, you'll want to modify it to return the 'fig' object.
    
    # Convert tensors for visualization
    adv_img = inv_normalize(adversarial_tensor.cpu()).squeeze().permute(1, 2, 0).numpy()
    adv_img = np.clip(adv_img, 0, 1)

    # Enhanced perturbation visualization
    pert_vis = perturbation.cpu().squeeze().permute(1, 2, 0).numpy()
    # Amplify for visibility
    pert_vis = (pert_vis - pert_vis.min()) / (pert_vis.max() - pert_vis.min() + 1e-8)

    # 1. Original Image
    ax1 = fig.add_subplot(gs[:, 0])
    if isinstance(original_img, torch.Tensor):
        original_img = inv_normalize(original_img.cpu()).squeeze().permute(1, 2, 0).numpy()
        original_img = np.clip(original_img, 0, 1)
    ax1.imshow(original_img)
    ax1.set_title(f'Original Image\n\"{original_preds[0]["class"]}\"',
                  fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. Perturbation (amplified)
    ax2 = fig.add_subplot(gs[:, 1])
    im = ax2.imshow(pert_vis, cmap='seismic')
    ax2.set_title(f'Perturbation (Amplified)\nε={epsilon}',
                  fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # 3. Adversarial Image
    ax3 = fig.add_subplot(gs[:, 2])
    ax3.imshow(adv_img)
    ax3.set_title(f'Adversarial Image\n\"{adv_preds[0]["class"]}\"',
                  fontsize=12, fontweight='bold', color='red')
    ax3.axis('off')

    # 4. Prediction Comparison (Top 3)
    ax4 = fig.add_subplot(gs[0, 3])
    classes_orig = [p['class'][:20] for p in original_preds[:3]]
    probs_orig = [p['probability'] * 100 for p in original_preds[:3]]
    ax4.barh(classes_orig, probs_orig, color='green', alpha=0.7)
    ax4.set_xlabel('Confidence (%)', fontsize=10)
    ax4.set_title('Original Predictions', fontsize=11, fontweight='bold')
    ax4.set_xlim(0, 100)
    for i, prob in enumerate(probs_orig):
        ax4.text(prob + 2, i, f'{prob:.1f}%', va='center', fontsize=9)

    ax5 = fig.add_subplot(gs[1, 3])
    classes_adv = [p['class'][:20] for p in adv_preds[:3]]
    probs_adv = [p['probability'] * 100 for p in adv_preds[:3]]
    ax5.barh(classes_adv, probs_adv, color='red', alpha=0.7)
    ax5.set_xlabel('Confidence (%)', fontsize=10)
    ax5.set_title('Adversarial Predictions', fontsize=11, fontweight='bold')
    ax5.set_xlim(0, 100)
    for i, prob in enumerate(probs_adv):
        ax5.text(prob + 2, i, f'{prob:.1f}%', va='center', fontsize=9)

    plt.suptitle(f'{attack_name} Attack Results', fontsize=16, fontweight='bold', y=1.02)
    
    # IMPORTANT: Return the figure for Gradio
    return fig


# --- From Cell 6: GradCAM ---

class GradCAM:
    """
    GradCAM: Visualizes which regions of image the model focuses on
    ... (docstring from cell) ...
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate GradCAM heatmap
        ... (docstring from cell) ...
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], device=activations.device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), target_class

def apply_colormap_on_image(org_img, activation_map, colormap='jet'):
    """
    Apply heatmap overlay on original image
    ... (docstring from cell) ...
    """
    if isinstance(org_img, torch.Tensor):
        org_img = inv_normalize(org_img.cpu()).squeeze().permute(1, 2, 0).numpy()
        org_img = np.clip(org_img, 0, 1)
    elif isinstance(org_img, Image.Image):
        org_img = np.array(org_img) / 255.0

    height, width = org_img.shape[:2]
    activation_map_resized = cv2.resize(activation_map, (width, height))
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(activation_map_resized)[:, :, :3]
    overlayed_img = 0.5 * org_img + 0.5 * heatmap
    overlayed_img = np.clip(overlayed_img, 0, 1)
    return overlayed_img, heatmap

def visualize_gradcam(original_img, input_tensor, model, labels, title="GradCAM Visualization"):
    """
    Visualize GradCAM for an image
    """
    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)
    cam, pred_class = gradcam.generate_cam(input_tensor)
    predictions = predict(input_tensor, model, labels, top_k=3)
    overlayed, heatmap = apply_colormap_on_image(original_img, cam, 'jet')
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    if isinstance(original_img, torch.Tensor):
        img_display = inv_normalize(original_img.cpu()).squeeze().permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1)
    else:
        img_display = original_img if isinstance(original_img, np.ndarray) else np.array(original_img)/255.0

    axes[0].imshow(img_display)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('GradCAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(overlayed)
    axes[2].set_title('Overlay (What Model Sees)', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    classes = [p['class'][:25] for p in predictions]
    probs = [p['probability'] * 100 for p in predictions]
    colors = sns.color_palette("RdYlGn_r", len(classes))
    axes[3].barh(classes, probs, color=colors)
    axes[3].set_xlabel('Confidence (%)', fontsize=11)
    axes[3].set_title('Predictions', fontsize=12, fontweight='bold')
    axes[3].set_xlim(0, 100)
    for i, prob in enumerate(probs):
        axes[3].text(prob + 2, i, f'{prob:.1f}%', va='center', fontsize=10)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig, cam, predictions


# --- From Cell 7: Integrated Gradients ---

def integrated_gradients(model, input_tensor, target_class, baseline=None, steps=50):
    """
    Integrated Gradients: Attributes prediction to input features
    ... (docstring from cell) ...
    """
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    
    scaled_inputs = []
    for i in range(steps + 1):
        scaled_input = baseline + (i / steps) * (input_tensor - baseline)
        scaled_inputs.append(scaled_input)

    scaled_inputs = torch.cat(scaled_inputs, dim=0).to(input_tensor.device)
    scaled_inputs.requires_grad = True
    outputs = model(scaled_inputs)
    target_outputs = outputs[:, target_class]
    gradients = torch.autograd.grad(outputs=target_outputs,
                                     inputs=scaled_inputs,
                                     grad_outputs=torch.ones_like(target_outputs),
                                     create_graph=False)[0]
    avg_gradients = gradients.mean(dim=0, keepdim=True)
    integrated_grads = (input_tensor - baseline) * avg_gradients
    return integrated_grads.detach()

def visualize_integrated_gradients(original_img, input_tensor, model, labels, title="Integrated Gradients"):
    """
    Visualize Integrated Gradients attribution
    """
    predictions = predict(input_tensor, model, labels, top_k=1)
    target_class = predictions[0]['index']
    attributions = integrated_gradients(model, input_tensor, target_class)
    attr_np = attributions.cpu().squeeze().permute(1, 2, 0).numpy()
    attr_sum = np.sum(np.abs(attr_np), axis=2)
    attr_sum = (attr_sum - attr_sum.min()) / (attr_sum.max() - attr_sum.min() + 1e-8)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    if isinstance(original_img, torch.Tensor):
        img_display = inv_normalize(original_img.cpu()).squeeze().permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1)
    else:
        img_display = original_img if isinstance(original_img, np.ndarray) else np.array(original_img)/255.0

    axes[0].imshow(img_display)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(attr_sum, cmap='hot')
    axes[1].set_title('Attribution Map', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    resized_attr = cv2.resize(attr_sum, (img_display.shape[1], img_display.shape[0]))
    cmap = plt.get_cmap('hot')
    colored_attr = cmap(resized_attr)[:, :, :3]
    overlay = 0.6 * img_display + 0.4 * colored_attr
    overlay = np.clip(overlay, 0, 1)
    axes[2].imshow(overlay)
    axes[2].set_title('Attribution Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    attr_pos = np.maximum(attr_sum, 0)
    axes[3].imshow(img_display, alpha=0.5)
    axes[3].imshow(attr_pos, cmap='Reds', alpha=0.5)
    axes[3].set_title('Positive Attributions\n(Supporting Prediction)',
                     fontsize=12, fontweight='bold', color='red')
    axes[3].axis('off')

    plt.suptitle(f'{title}\nPrediction: {predictions[0]["class"]} ({predictions[0]["probability"]*100:.1f}%)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig, attributions, predictions

# --- From Cell 9: Combined Attack & Explainability ---

def complete_adversarial_analysis(original_img, image_tensor, model, labels,
                                   attack_type='fgsm', epsilon=0.03,
                                   alpha=0.01, num_iter=10):
    """
    Complete pipeline: Attack + Explainability Analysis
    ... (docstring from cell) ...
    """
    device = image_tensor.device
    orig_preds = predict(image_tensor, model, labels, top_k=5)
    target_class = orig_preds[0]['index']

    if attack_type.lower() == 'fgsm':
        adv_tensor, perturbation = generate_fgsm_adversarial(
            model, image_tensor.clone(), target_class, epsilon
        )
    elif attack_type.lower() == 'pgd':
        adv_tensor, perturbation = pgd_attack(
            model, image_tensor.clone(), target_class, epsilon, alpha, num_iter
        )
    else:
        raise ValueError("attack_type must be 'fgsm' or 'pgd'")

    adv_preds = predict(adv_tensor, model, labels, top_k=5)

    # GradCAM for original
    target_layer = model.layer4[-1]
    gradcam_orig = GradCAM(model, target_layer)
    cam_orig, _ = gradcam_orig.generate_cam(image_tensor)

    # GradCAM for adversarial
    gradcam_adv = GradCAM(model, target_layer)
    cam_adv, _ = gradcam_adv.generate_cam(adv_tensor)

    img_display = inv_normalize(image_tensor.cpu()).squeeze().permute(1, 2, 0).numpy()
    img_display = np.clip(img_display, 0, 1)
    adv_display = inv_normalize(adv_tensor.cpu()).squeeze().permute(1, 2, 0).numpy()
    adv_display = np.clip(adv_display, 0, 1)

    overlay_orig, _ = apply_colormap_on_image(img_display, cam_orig, 'jet')
    overlay_adv, _ = apply_colormap_on_image(adv_display, cam_adv, 'jet')
    pert_vis = perturbation.cpu().squeeze().permute(1, 2, 0).numpy()
    pert_vis = (pert_vis - pert_vis.min()) / (pert_vis.max() - pert_vis.min() + 1e-8)

    # --- Create the comprehensive plot ---
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    # ... (All the plt code from Cell 9) ...
    # This function is complex, but the definition is in Cell 9
    # For Gradio, we must return 'fig' instead of plt.show()
    
    # ===== ROW 1: ORIGINAL IMAGE ANALYSIS =====
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_display)
    ax1.set_title('Original Image', fontsize=13, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(cam_orig, cmap='jet')
    ax2.set_title('GradCAM (Original)', fontsize=13, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(overlay_orig)
    ax3.set_title('What Model Sees (Original)', fontsize=13, fontweight='bold')
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[0, 3])
    classes = [p['class'][:20] for p in orig_preds[:3]]
    probs = [p['probability'] * 100 for p in orig_preds[:3]]
    ax4.barh(classes, probs, color='green', alpha=0.7)
    ax4.set_xlabel('Confidence (%)', fontsize=11)
    ax4.set_title('Original Predictions', fontsize=13, fontweight='bold')
    ax4.set_xlim(0, 100)
    for i, prob in enumerate(probs):
        ax4.text(prob + 2, i, f'{prob:.1f}%', va='center', fontsize=10)

    # ===== ROW 2: ATTACK VISUALIZATION =====
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.axis('off')
    attack_info = f"{attack_type.upper()} Attack\nε = {epsilon}"
    if attack_type.lower() == 'pgd':
        attack_info += f"\nα = {alpha}\niterations = {num_iter}"
    ax5.text(0.5, 0.5, attack_info, ha='center', va='center',
             fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
             transform=ax5.transAxes)

    ax6 = fig.add_subplot(gs[1, 1])
    im = ax6.imshow(pert_vis, cmap='seismic')
    ax6.set_title('Perturbation (Amplified)', fontsize=13, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)

    ax7 = fig.add_subplot(gs[1, 2])
    diff = np.abs(adv_display - img_display)
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    ax7.imshow(diff, cmap='hot')
    ax7.set_title('Image Difference', fontsize=13, fontweight='bold')
    ax7.axis('off')

    ax8 = fig.add_subplot(gs[1, 3])
    pert_l2 = torch.norm(perturbation).item()
    pert_linf = perturbation.abs().max().item()
    stats_text = f"L2 Norm: {pert_l2:.4f}\nL∞ Norm: {pert_linf:.4f}\n\nMax pixel change:\n{pert_linf*255:.2f}/255"
    ax8.text(0.5, 0.5, stats_text, ha='center', va='center',
             fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
             transform=ax8.transAxes)
    ax8.axis('off')

    # ===== ROW 3: ADVERSARIAL IMAGE ANALYSIS =====
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.imshow(adv_display)
    ax9.set_title('Adversarial Image', fontsize=13, fontweight='bold', color='red')
    ax9.axis('off')

    ax10 = fig.add_subplot(gs[2, 1])
    ax10.imshow(cam_adv, cmap='jet')
    ax10.set_title('GradCAM (Adversarial)', fontsize=13, fontweight='bold', color='red')
    ax10.axis('off')

    ax11 = fig.add_subplot(gs[2, 2])
    ax11.imshow(overlay_adv)
    ax11.set_title('What Model Sees (Adversarial)', fontsize=13, fontweight='bold', color='red')
    ax11.axis('off')

    ax12 = fig.add_subplot(gs[2, 3])
    classes_adv = [p['class'][:20] for p in adv_preds[:3]]
    probs_adv = [p['probability'] * 100 for p in adv_preds[:3]]
    ax12.barh(classes_adv, probs_adv, color='red', alpha=0.7)
    ax12.set_xlabel('Confidence (%)', fontsize=11)
    ax12.set_title('Adversarial Predictions', fontsize=13, fontweight='bold', color='red')
    ax12.set_xlim(0, 100)
    for i, prob in enumerate(probs_adv):
        ax12.text(prob + 2, i, f'{prob:.1f}%', va='center', fontsize=10)

    success = "✅ ATTACK SUCCESSFUL" if orig_preds[0]['class'] != adv_preds[0]['class'] else "⚠️ ATTACK FAILED"
    plt.suptitle(f'Complete Adversarial Attack Analysis - {success}',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Return fig for Gradio, and also the report string
    report_text = f"""
    DETAILED ANALYSIS REPORT
    ================================================
    ATTACK CONFIGURATION:
    • Attack Type: {attack_type.upper()}
    • Epsilon (ε): {epsilon}
    {f"• Alpha (α): {alpha}" if attack_type.lower() == 'pgd' else ''}
    {f"• Iterations: {num_iter}" if attack_type.lower() == 'pgd' else ''}

    PREDICTION CHANGES:
    Original:
        1. {orig_preds[0]['class']}: {orig_preds[0]['probability']*100:.2f}%
        2. {orig_preds[1]['class']}: {orig_preds[1]['probability']*100:.2f}%
        3. {orig_preds[2]['class']}: {orig_preds[2]['probability']*100:.2f}%

    Adversarial:
        1. {adv_preds[0]['class']}: {adv_preds[0]['probability']*100:.2f}% {'❌' if fooled else ''}
        2. {adv_preds[1]['class']}: {adv_preds[1]['probability']*100:.2f}%
        3. {adv_preds[2]['class']}: {adv_preds[2]['probability']*100:.2f}%

    PERTURBATION METRICS:
    • L2 Norm: {pert_l2:.6f}
    • L∞ Norm: {pert_linf:.6f}

    EXPLAINABILITY INSIGHTS:
    • GradCAM shows the model's attention shifted to irrelevant regions.
    """
    
    fooled = orig_preds[0]['class'] != adv_preds[0]['class']
    
    return fig, report_text, adv_tensor, fooled

# --- From Cell 20: C&W Attack ---

def cw_l2_attack(model, image_tensor, target_class_idx, c=1e-2, kappa=0,
                 max_iter=100, learning_rate=0.01):
    """
    C&W L2 Attack (Carlini-Wagner)
    ... (docstring from cell) ...
    """
    device = image_tensor.device
    w = torch.atanh(image_tensor * 1.9999).detach().clone().to(device) # Inverse of tanh(w)*0.5 + 0.5
    w.requires_grad = True

    optimizer = torch.optim.Adam([w], lr=learning_rate)
    
    # We use a fixed target class (second most likely)
    orig_output = model(image_tensor)
    orig_class = orig_output.argmax(dim=1).item()
    
    # Get second most likely class as target
    second_best = orig_output.argsort(dim=1, descending=True)[0, 1]
    target_class_idx = second_best.item()
    target_label = torch.tensor([target_class_idx]).to(device)

    for step in range(max_iter):
        optimizer.zero_grad()
        adv_image = 0.5 * (torch.tanh(w) + 1)
        adv_image = (adv_image - torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)) / \
                    torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)

        output = model(adv_image)
        
        # C&W loss function
        output_orig = output[0, orig_class]
        output_target = output[0, target_class_idx]
        
        loss_adv = torch.clamp(output_orig - output_target, min=-kappa)
        
        l2_dist = torch.sum((adv_image - image_tensor)**2)
        total_loss = l2_dist + c * loss_adv
        
        total_loss.backward()
        optimizer.step()

        if (step + 1) % 20 == 0:
            pred = output.argmax(dim=1).item()
            if pred == target_class_idx:
                break
    
    adv_image = 0.5 * (torch.tanh(w) + 1)
    adv_image_norm = (adv_image - torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)) / \
                     torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)
                     
    perturbation = adv_image_norm - image_tensor
    
    return adv_image_norm.detach(), perturbation.detach()


# --- From Cell 22: DeepFool Attack ---

def deepfool_attack(image_tensor, model, max_iter=50, overshoot=0.02):
    """
    DeepFool Attack
    ... (docstring from cell) ...
    """
    device = image_tensor.device
    image_tensor = image_tensor.clone().detach().to(device)
    image_tensor.requires_grad = True

    output = model(image_tensor)
    orig_class = output.argmax(dim=1).item()
    
    perturbed_image = image_tensor.clone()
    
    total_perturbation = torch.zeros_like(image_tensor).to(device)
    
    for i in range(max_iter):
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        pred_class = output.argmax(dim=1).item()
        
        if pred_class != orig_class:
            break # Success
            
        # Get gradients for original class
        model.zero_grad()
        output[0, orig_class].backward(retain_graph=True)
        grad_orig = perturbed_image.grad.data.clone()
        
        min_perturbation = None
        
        # Get gradients for other top classes
        top_k = output.argsort(dim=1, descending=True)[0, 1:10]
        
        for k in top_k:
            model.zero_grad()
            output[0, k].backward(retain_graph=True)
            grad_k = perturbed_image.grad.data.clone()
            
            # Difference in outputs and gradients
            w_k = grad_k - grad_orig
            f_k = output[0, k] - output[0, orig_class]
            
            # Calculate perturbation
            perturb_k = (torch.abs(f_k) / (torch.norm(w_k, p=2)**2 + 1e-8)) * w_k
            
            if min_perturbation is None or torch.norm(perturb_k, p=2) < torch.norm(min_perturbation, p=2):
                min_perturbation = perturb_k
        
        # Apply minimal perturbation
        r_i = min_perturbation
        total_perturbation += r_i
        
        with torch.no_grad():
            perturbed_image += r_i
            
    # Apply overshoot
    with torch.no_grad():
        total_perturbation = (1 + overshoot) * total_perturbation
        adv_image = image_tensor + total_perturbation
        # Clamp to valid normalized range
        adv_image = torch.clamp(adv_image, image_tensor.min().item(), image_tensor.max().item())
        
    perturbation = adv_image - image_tensor
    
    return adv_image.detach(), perturbation.detach(), i+1


# --- From Cell 24: Defense Mechanisms ---

def apply_jpeg_defense(adv_tensor, quality=75):
    """Apply JPEG compression defense"""
    adv_img = inv_normalize(adv_tensor.cpu()).squeeze().permute(1, 2, 0).numpy()
    adv_img = np.clip(adv_img * 255, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(adv_img)
    
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    defended_img = Image.open(buffer).convert('RGB')
    
    return preprocess(defended_img).unsqueeze(0).to(adv_tensor.device)

def apply_bit_reduction_defense(adv_tensor, bit_depth=4):
    """Apply bit depth reduction defense"""
    adv_img = inv_normalize(adv_tensor.cpu())
    quant_levels = 2**bit_depth
    quant_img = (adv_img * (quant_levels - 1)).round() / (quant_levels - 1)
    defended_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(quant_img)
    # --- THIS IS THE FIX for the RuntimeError ---
    # The tensor already has the batch dimension, so we remove .unsqueeze(0)
    return defended_tensor.to(adv_tensor.device)

def apply_gaussian_defense(adv_tensor, sigma=1.0):
    """Apply Gaussian blur defense"""
    blurred_tensor = transforms.functional.gaussian_blur(adv_tensor, kernel_size=3, sigma=sigma)
    return blurred_tensor

# --- From Cell 25: Adversarial Detection ---

def detect_adversarial_statistical(image_tensor, model, num_samples=10, threshold=0.05):
    """Statistical Detection Method"""
    predictions = []
    device = image_tensor.device

    with torch.no_grad():
        orig_output = model(image_tensor)
        orig_class = orig_output.argmax(dim=1).item()

        for _ in range(num_samples):
            noise = torch.randn_like(image_tensor) * 0.01
            noisy_input = image_tensor + noise
            output = model(noisy_input)
            pred_class = output.argmax(dim=1).item()
            predictions.append(pred_class)

    unique_predictions = len(set(predictions))
    variance_score = unique_predictions / num_samples
    is_adversarial = variance_score > threshold
    return is_adversarial, variance_score

# --- From Cell 26: LIME ---
from skimage.segmentation import mark_boundaries

def explain_with_lime(image, image_tensor, model, labels, num_samples=100, num_features=5):
    """
    Explain prediction using LIME
    ... (docstring from cell) ...
    """
    if isinstance(image, torch.Tensor):
        img_np = inv_normalize(image.cpu()).squeeze().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
    elif isinstance(image, Image.Image):
        img_np = np.array(image) / 255.0
    else:
        img_np = image
        
    preds = predict(image_tensor, model, labels, top_k=1)
    pred_class_idx = preds[0]['index']
    
    def predict_fn(images):
        batch_preds = []
        for img in images:
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            img_tensor_lime = preprocess(img_pil).unsqueeze(0).to(image_tensor.device)
            with torch.no_grad():
                output = model(img_tensor_lime)
                probs = F.softmax(output, dim=1)
                batch_preds.append(probs.cpu().numpy()[0])
        return np.array(batch_preds)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_np, predict_fn, top_labels=3, hide_color=0, num_samples=num_samples
    )
    
    temp, mask = explanation.get_image_and_mask(
        pred_class_idx, positive_only=True, num_features=num_features, hide_rest=False
    )
    
    return mark_boundaries(temp, mask)

# --- From Cell 27: SHAP ---

def explain_with_shap(image, image_tensor, model, labels):
    """
    Explain prediction using SHAP
    ... (docstring from cell) ...
    """
    if isinstance(image, torch.Tensor):
        img_np = inv_normalize(image.cpu()).squeeze().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
    elif isinstance(image, Image.Image):
        img_np = np.array(image) / 255.0
    else:
        img_np = image

    # Create dummy data for background
    batch = torch.randn(10, 3, 224, 224, device=image_tensor.device) * 0.1 + image_tensor
    
    e = shap.DeepExplainer(model, batch)
    shap_values = e.shap_values(image_tensor)
    
    # Process SHAP values for plotting
    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    img_numpy = np.swapaxes(np.swapaxes(image_tensor.cpu().numpy(), 1, -1), 1, 2)
    
    # Get top prediction
    preds = predict(image_tensor, model, labels, top_k=1)
    pred_class_idx = preds[0]['index']
    
    # Generate plot
    fig = plt.figure(figsize=(12, 5))
    plt.suptitle(f"SHAP Explanation for: {preds[0]['class']}", fontsize=14)
    shap.image_plot(shap_numpy, img_numpy, show=False)
    
    return fig

# --- From Cell 34 & 36: Gradio Interface ---

def gradio_interface(model_zoo, device, labels):
    """
    Creates the complete Gradio web interface
    """
    
    # Helper function to get model info
    def get_model_info(model_name):
        model_arch = model_zoo[model_name]
        total_params = sum(p.numel() for p in model_arch.parameters())
        size_mb = total_params * 4 / (1024**2)
        return f"**{model_name}**\n- Parameters: {total_params:,}\n- Size: {size_mb:.2f} MB"

    # Core analysis function for Gradio
    def run_analysis(model_name, image_upload, attack_type, epsilon, defense_type):
        
        if image_upload is None:
            return None, None, "Please upload an image.", None, None

        selected_model = model_zoo[model_name].to(device)
        selected_model.eval()
        
        img = image_upload.convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        # 1. Original Prediction
        orig_preds = predict(img_tensor, selected_model, labels, top_k=5)
        target_class = orig_preds[0]['index']
        orig_preds_df = {labels[p['index']]: p['probability'] for p in orig_preds}

        # 2. Generate Attack
        if attack_type == 'FGSM':
            adv_tensor, pert = generate_fgsm_adversarial(selected_model, img_tensor.clone(), target_class, epsilon)
        elif attack_type == 'PGD':
            adv_tensor, pert = pgd_attack(selected_model, img_tensor.clone(), target_class, epsilon, alpha=epsilon/4, num_iter=10)
        else: # C&W
            adv_tensor, pert = cw_l2_attack(selected_model, img_tensor.clone(), target_class, max_iter=50) # Faster iter for demo
        
        adv_preds_raw = predict(adv_tensor, selected_model, labels, top_k=5)
        
        # 3. Apply Defense
        defended_tensor = adv_tensor.clone()
        if defense_type == "JPEG (Q=75)":
            defended_tensor = apply_jpeg_defense(adv_tensor, 75)
        elif defense_type == "Bit Reduction (4-bit)":
            defended_tensor = apply_bit_reduction_defense(adv_tensor)
        elif defense_type == "Gaussian Blur (σ=1.0)":
            defended_tensor = apply_gaussian_defense(adv_tensor, 1.0)
        elif defense_type == "Ensemble (JPEG + Blur)":
            defended_tensor = apply_gaussian_defense(apply_jpeg_defense(adv_tensor, 75))
        
        defended_preds = predict(defended_tensor, selected_model, labels, top_k=5)
        defended_preds_df = {labels[p['index']]: p['probability'] for p in defended_preds}

        # 4. Generate Images for output
        adv_img_out = np.clip(inv_normalize(adv_tensor.cpu()).squeeze().permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
        defended_img_out = np.clip(inv_normalize(defended_tensor.cpu()).squeeze().permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
        
        pert_vis = pert.cpu().squeeze().permute(1, 2, 0).numpy()
        pert_vis = (pert_vis - pert_vis.min()) / (pert_vis.max() - pert_vis.min() + 1e-8)
        pert_vis_out = (pert_vis * 255).astype(np.uint8) # Grayscale, but show as heatmap later

        # 5. Generate GradCAMs
        fig_orig, _, _ = visualize_gradcam(img, img_tensor, selected_model, labels, "Original")
        fig_adv, _, _ = visualize_gradcam(adv_img_out, adv_tensor, selected_model, labels, "Adversarial")
        
        # Create summary
        summary = f"""
        **Attack Analysis**
        - Original: **{orig_preds[0]['class']}** ({orig_preds[0]['probability']*100:.1f}%)
        - Attack: {attack_type} (ε={epsilon})
        - Adversarial: **{adv_preds_raw[0]['class']}** ({adv_preds_raw[0]['probability']*100:.1f}%)
        - **Status: {'✅ FOOLED' if orig_preds[0]['class'] != adv_preds_raw[0]['class'] else '❌ ROBUST'}**

        **Defense Analysis**
        - Defense: {defense_type}
        - Defended: **{defended_preds[0]['class']}** ({defended_preds[0]['probability']*100:.1f}%)
        
         
        - **Status: {'✅ RECOVERED' if defense_type != "None" else '❌ FAILED'}**
        """
        
        return orig_preds_df, adv_img_out, summary, defended_img_out, defended_preds_df

    # Gradio UI Definition
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🤖 Trustworthy AI: Attack, Explain, Defend
            This dashboard demonstrates the core concepts of adversarial machine learning.
            1.  **Upload** an image and select a **Model**.
            2.  Choose an **Attack** (like FGSM) to generate an adversarial example.
            3.  Select a **Defense** to try and recover the original prediction.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                image_upload = gr.Image(type="pil", label="Upload Image")
                model_name = gr.Dropdown(list(model_zoo.keys()), label="Select Model", value="ResNet50")
                model_info = gr.Markdown(get_model_info("ResNet50"))
                attack_type = gr.Radio(["FGSM", "PGD", "C&W (L2)"], label="Attack Type", value="FGSM")
                epsilon = gr.Slider(0.005, 0.1, value=0.03, step=0.005, label="Attack Strength (Epsilon)")
                defense_type = gr.Dropdown(
                    ["None", "JPEG (Q=75)", "Bit Reduction (4-bit)", "Gaussian Blur (σ=1.0)", "Ensemble (JPEG + Blur)"],
                    label="Defense Mechanism",
                    value="None"
                )
                run_btn = gr.Button("Run Analysis", variant="primary")
            
            with gr.Column(scale=3):
                gr.Markdown("### Analysis Results")
                summary_output = gr.Markdown("Analysis report will appear here.")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Original Prediction")
                        pred_orig = gr.Label(num_top_classes=5)
                    with gr.Column():
                        gr.Markdown("#### Adversarial Image")
                        img_adv = gr.Image(type="pil", label="Adversarial Image", interactive=False)
                    with gr.Column():
                        gr.Markdown("#### Defended Image")
                        img_defended = gr.Image(type="pil", label="Defended Image", interactive=False)
                        pred_defended = gr.Label(num_top_classes=5)

        # Update model info when dropdown changes
        model_name.change(get_model_info, inputs=model_name, outputs=model_info)
        
        # Run analysis on button click
        run_btn.click(
            run_analysis,
            inputs=[model_name, image_upload, attack_type, epsilon, defense_type],
            outputs=[pred_orig, img_adv, summary_output, img_defended, pred_defended]
        )

    return demo
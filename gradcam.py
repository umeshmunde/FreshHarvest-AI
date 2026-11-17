"""
Grad-CAM (Gradient-weighted Class Activation Mapping) module for FreshHarvest AI.
Generates explainable visual heatmaps highlighting which regions (discoloration,
decay, fungal blemishes, shape) influenced the model's prediction.
"""

import os
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

class GradCAM:
    """
    Grad-CAM implementation compatible with EfficientNet, ResNet, and MobileNet.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        if target_layer is None:
            self.target_layer = self.model.get_gradcam_target_layer()
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target convolutional layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hooks.extend([h1, h2])

    def remove_hooks(self):
        """Remove hooks to avoid memory leaks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generate raw Grad-CAM heatmap for a given input image tensor.

        Args:
            input_tensor (torch.Tensor): 4D tensor (1, C, H, W)
            target_class (int, optional): Index of target class. Defaults to top-1 class.

        Returns:
            np.ndarray: 2D float array (H, W) normalized between 0.0 and 1.0
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        # Backward pass on the target class logit
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        # Global average pooling on gradients
        # activations: (1, Channels, H, W), gradients: (1, Channels, H, W)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, Channels, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # ReLU to focus only on features that have positive influence

        # Convert to numpy
        cam = cam.squeeze().cpu().numpy()

        # Normalize between 0 and 1
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_heatmap(self, original_image, heatmap, alpha=0.55, beta=0.45, colormap=cv2.COLORMAP_JET):
        """
        Resize heatmap and blend with original image.

        Args:
            original_image (PIL.Image or np.ndarray): Original image
            heatmap (np.ndarray): 2D float array [0.0, 1.0]
            alpha (float): Original image blending weight
            beta (float): Heatmap blending weight
            colormap: OpenCV colormap constant

        Returns:
            np.ndarray: Blended RGB image uint8
        """
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image.convert('RGB'))

        orig_h, orig_w = original_image.shape[:2]

        # Resize heatmap to original image dimensions
        resized_cam = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        uint8_cam = np.uint8(255 * resized_cam)

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(uint8_cam, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend
        blended = cv2.addWeighted(original_image, alpha, heatmap_rgb, beta, 0)
        return blended, heatmap_rgb

def generate_and_save_gradcam(model, input_tensor, original_image, output_path, target_class=None):
    """
    High-level convenience function to generate and write Grad-CAM overlay image to disk.

    Args:
        model: FruitFreshnessModel instance
        input_tensor: 4D tensor (1, 3, 224, 224)
        original_image: PIL Image or path
        output_path: Path to save the resulting overlay image
        target_class: Target class index

    Returns:
        tuple: (output_path_str, raw_heatmap_2d)
    """
    if isinstance(original_image, (str, Path)):
        original_image = Image.open(str(original_image)).convert('RGB')

    cam_engine = GradCAM(model)
    try:
        heatmap = cam_engine.generate_heatmap(input_tensor, target_class=target_class)
        blended, _ = cam_engine.overlay_heatmap(original_image, heatmap)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as RGB image
        blended_pil = Image.fromarray(blended)
        blended_pil.save(str(output_path), quality=92)
        return str(output_path), heatmap
    finally:
        cam_engine.remove_hooks()


"""
Distortion Attack Methods based on WAVESBench

Implements various image distortions to test the robustness of image protection
algorithms. All methods follow the WAVES distortion framework with exact
parameter ranges and strength calculations.

Distortion types supported:
- rotation: Image rotation (0-45 degrees)  
- resizedcrop: Random resized crop (0.5-1.0 scale)
- erasing: Random erasing (0-0.25 area)
- brightness: Brightness adjustment (1-2x)
- contrast: Contrast adjustment (1-2x)
- blurring: Gaussian blur (0-20 kernel size)
- noise: Gaussian noise (0-0.1 std)
- compression: JPEG compression (10-90 quality)
"""

import torch
import numpy as np
import random
import io
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from typing import Dict, Union, Any, Tuple, Optional
from .base import BaseAttack


# WAVES distortion strength parameters - exact values from original
DISTORTION_STRENGTH_PARAMS = {
    "rotation": (0, 45),
    "resizedcrop": (1, 0.5),  # Note: inverted range (max, min)
    "erasing": (0, 0.25),
    "brightness": (1, 2),
    "contrast": (1, 2),
    "blurring": (0, 20),
    "noise": (0, 0.1),
    "saltpepper": (0, 0.1),  # Salt and pepper noise
    "compression": (90, 10),  # Note: inverted range (max quality, min quality)
}


def relative_strength_to_absolute(strength: float, distortion_type: str) -> float:
    """
    Convert relative strength [0,1] to absolute parameter value
    
    Args:
        strength: Relative strength in [0, 1]
        distortion_type: Type of distortion
        
    Returns:
        Absolute parameter value
    """
    assert 0 <= strength <= 1, f"Strength must be in [0,1], got {strength}"
    assert distortion_type in DISTORTION_STRENGTH_PARAMS, f"Unknown distortion type: {distortion_type}"
    
    min_val, max_val = DISTORTION_STRENGTH_PARAMS[distortion_type]
    absolute_strength = strength * (max_val - min_val) + min_val
    
    # Clamp to valid range
    absolute_strength = max(absolute_strength, min(min_val, max_val))
    absolute_strength = min(absolute_strength, max(min_val, max_val))
    
    return absolute_strength


def set_random_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    return transforms.ToPILImage()(tensor.cpu())


def to_tensor(pil_image: Image.Image, device: str = "cpu") -> torch.Tensor:
    """Convert PIL Image to tensor"""
    return transforms.ToTensor()(pil_image).to(device)



# Individual distortion attack classes - directly inheriting from BaseAttack
class RotationAttack(BaseAttack):
    """Rotation attack with WAVES parameters"""
    def __init__(self, strength: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.strength = strength
    
    def attack(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply rotation attack"""
        strength = kwargs.get('strength', self.strength)
        
        # Convert to PIL and apply rotation
        pil_image = to_pil(image)
        angle = strength if strength is not None else random.uniform(*DISTORTION_STRENGTH_PARAMS["rotation"])
        rotated_pil = F.rotate(pil_image, angle)
        return to_tensor(rotated_pil, self.device)


class ResizedCropAttack(BaseAttack):
    """Resized crop attack with WAVES parameters"""
    def __init__(self, strength: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.strength = strength
    
    def attack(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply resized crop attack"""
        strength = kwargs.get('strength', self.strength)
        
        # Convert to PIL and apply resized crop
        pil_image = to_pil(image)
        scale = strength if strength is not None else random.uniform(*DISTORTION_STRENGTH_PARAMS["resizedcrop"])
        
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            pil_image, scale=(scale, scale), ratio=(1, 1)
        )
        cropped_pil = F.resized_crop(pil_image, i, j, h, w, pil_image.size)
        return to_tensor(cropped_pil, self.device)


class ErasingAttack(BaseAttack):
    """Random erasing attack with WAVES parameters"""
    def __init__(self, strength: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.strength = strength
    
    def attack(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply random erasing attack"""
        strength = kwargs.get('strength', self.strength)
        
        # Convert to tensor for erasing
        tensor = image.unsqueeze(0) if len(image.shape) == 3 else image
        
        scale = strength if strength is not None else random.uniform(*DISTORTION_STRENGTH_PARAMS["erasing"])
        i, j, h, w, v = transforms.RandomErasing.get_params(
            tensor, scale=(scale, scale), ratio=(1, 1), value=[0]
        )
        
        erased_tensor = F.erase(tensor, i, j, h, w, v)
        return erased_tensor.squeeze(0) if len(image.shape) == 3 else erased_tensor


class BrightnessAttack(BaseAttack):
    """Brightness adjustment attack with WAVES parameters"""
    def __init__(self, strength: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.strength = strength
    
    def attack(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply brightness attack"""
        strength = kwargs.get('strength', self.strength)
        
        # Convert to PIL and apply brightness
        pil_image = to_pil(image)
        factor = strength if strength is not None else random.uniform(*DISTORTION_STRENGTH_PARAMS["brightness"])
        
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced_pil = enhancer.enhance(factor)
        return to_tensor(enhanced_pil, self.device)


class ContrastAttack(BaseAttack):
    """Contrast adjustment attack with WAVES parameters"""
    def __init__(self, strength: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.strength = strength
    
    def attack(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply contrast attack"""
        strength = kwargs.get('strength', self.strength)
        
        # Convert to PIL and apply contrast
        pil_image = to_pil(image)
        factor = strength if strength is not None else random.uniform(*DISTORTION_STRENGTH_PARAMS["contrast"])
        
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced_pil = enhancer.enhance(factor)
        return to_tensor(enhanced_pil, self.device)


class BlurringAttack(BaseAttack):
    """Gaussian blur attack with WAVES parameters"""
    def __init__(self, strength: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.strength = strength
    
    def attack(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply blurring attack"""
        strength = kwargs.get('strength', self.strength)
        
        # Convert to PIL and apply blur
        pil_image = to_pil(image)
        kernel_size = int(strength) if strength is not None else random.uniform(*DISTORTION_STRENGTH_PARAMS["blurring"])
        
        blurred_pil = pil_image.filter(ImageFilter.GaussianBlur(kernel_size))
        return to_tensor(blurred_pil, self.device)


class NoiseAttack(BaseAttack):
    """Gaussian noise attack with WAVES parameters"""
    def __init__(self, strength: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.strength = strength
    
    def attack(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply noise attack"""
        strength = kwargs.get('strength', self.strength)
        
        # Add Gaussian noise
        std = strength if strength is not None else random.uniform(*DISTORTION_STRENGTH_PARAMS["noise"])
        noise = torch.randn(image.size(), device=self.device) * std
        noisy_tensor = (image + noise).clamp(0, 1)
        return noisy_tensor


class SaltPepperAttack(BaseAttack):
    """Salt and pepper noise attack with WAVES parameters"""
    def __init__(self, strength: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.strength = strength
    
    def attack(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply salt and pepper noise attack"""
        strength = kwargs.get('strength', self.strength)
        
        # Add salt and pepper noise
        prob = strength if strength is not None else random.uniform(*DISTORTION_STRENGTH_PARAMS["saltpepper"])
        
        noise_mask = torch.rand(image.size(), device=self.device) < prob
        salt_mask = torch.rand(image.size(), device=self.device) < 0.5
        
        noisy_tensor = image.clone()
        noisy_tensor[noise_mask & salt_mask] = 1.0  # Salt
        noisy_tensor[noise_mask & ~salt_mask] = 0.0  # Pepper
        
        return noisy_tensor


class CompressionAttack(BaseAttack):
    """JPEG compression attack with WAVES parameters"""
    def __init__(self, strength: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.strength = strength
    
    def attack(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply compression attack"""
        strength = kwargs.get('strength', self.strength)
        
        # Convert to PIL and apply compression
        pil_image = to_pil(image)
        quality = int(strength) if strength is not None else random.uniform(*DISTORTION_STRENGTH_PARAMS["compression"])
        quality = max(1, min(100, int(quality)))
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed_pil = Image.open(buffer)
        return to_tensor(compressed_pil, self.device)

"""
Distortion Attack Methods with Direct Parameter Control

Implements various image distortions to test the robustness of image protection
algorithms. All methods use direct parameter control without strength conversion.

Distortion types supported:
- RotationAttack: Image rotation with direct angle control (0-45 degrees)  
- ResizedCropAttack: Random resized crop with direct scale control (0.5-1.0)
- ErasingAttack: Random erasing with direct area ratio control (0-0.25)
- BrightnessAttack: Brightness adjustment with direct factor control (1-2x)
- ContrastAttack: Contrast adjustment with direct factor control (1-2x)
- BlurringAttack: Gaussian blur with direct kernel size control (0-20)
- NoiseAttack: Gaussian noise with direct std control (0-0.1)
- SaltPepperAttack: Salt & pepper noise with direct probability control (0-0.1)
- CompressionAttack: JPEG compression with direct quality control (1-100)
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

DISTORTION_STRENGTH_PARAMS = {
    "rotation": (0, 45),        # DEPRECATED: Use angle parameter directly
    "resizedcrop": (1, 0.5),    # DEPRECATED: Use scale parameter directly  
    "erasing": (0, 0.25),       # DEPRECATED: Use area_ratio parameter directly
    "brightness": (1, 2),       # DEPRECATED: Use factor parameter directly
    "contrast": (1, 2),         # DEPRECATED: Use factor parameter directly
    "blurring": (0, 20),        # DEPRECATED: Use kernel_size parameter directly
    "noise": (0, 0.1),          # DEPRECATED: Use std parameter directly
    "saltpepper": (0, 0.1),     # DEPRECATED: Use prob parameter directly
    "compression": (90, 10),    # DEPRECATED: Use quality parameter directly
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
    """Rotation attack with direct angle control"""
    def __init__(self, angle: float = 22.5, **kwargs):
        super().__init__(**kwargs)
        self.angle = angle
    
    def attack(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply rotation attack
        
        Args:
            image: Input image tensor [C, H, W]
            angle: Rotation angle in degrees (0-45), higher values = more rotation
        
        Returns:
            Rotated image tensor
        """
        angle = kwargs.get('angle', self.angle)
        
        # Convert to PIL and apply rotation
        pil_image = to_pil(image)
        angle = max(0, min(45, float(angle)))  # Clamp to valid range
        rotated_pil = F.rotate(pil_image, angle)
        return to_tensor(rotated_pil, self.device)


class ResizedCropAttack(BaseAttack):
    """Resized crop attack with direct scale control"""
    def __init__(self, scale: float = 0.75, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
    
    def attack(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply resized crop attack
        
        Args:
            image: Input image tensor [C, H, W]
            scale: Crop scale factor (0.5-1.0), lower values = more cropping
        
        Returns:
            Cropped and resized image tensor
        """
        scale = kwargs.get('scale', self.scale)
        
        # Convert to PIL and apply resized crop
        pil_image = to_pil(image)
        scale = max(0.5, min(1.0, float(scale)))  # Clamp to valid range
        
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            pil_image, scale=(scale, scale), ratio=(1, 1)
        )
        cropped_pil = F.resized_crop(pil_image, i, j, h, w, pil_image.size)
        return to_tensor(cropped_pil, self.device)


class ErasingAttack(BaseAttack):
    """Random erasing attack with direct area control"""
    def __init__(self, area_ratio: float = 0.125, **kwargs):
        super().__init__(**kwargs)
        self.area_ratio = area_ratio
    
    def attack(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply random erasing attack
        
        Args:
            image: Input image tensor [C, H, W]
            area_ratio: Ratio of area to erase (0-0.25), higher values = more erasing
        
        Returns:
            Image tensor with random areas erased
        """
        area_ratio = kwargs.get('area_ratio', self.area_ratio)
        
        # Convert to tensor for erasing
        tensor = image.unsqueeze(0) if len(image.shape) == 3 else image
        
        area_ratio = max(0, min(0.25, float(area_ratio)))  # Clamp to valid range
        i, j, h, w, v = transforms.RandomErasing.get_params(
            tensor, scale=(area_ratio, area_ratio), ratio=(1, 1), value=[0]
        )
        
        erased_tensor = F.erase(tensor, i, j, h, w, v)
        return erased_tensor.squeeze(0) if len(image.shape) == 3 else erased_tensor


class BrightnessAttack(BaseAttack):
    """Brightness adjustment attack with direct factor control"""
    def __init__(self, factor: float = 1.5, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
    
    def attack(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply brightness attack
        
        Args:
            image: Input image tensor [C, H, W]
            factor: Brightness factor (1-2), 1=no change, 2=double brightness
        
        Returns:
            Brightness-adjusted image tensor
        """
        factor = kwargs.get('factor', self.factor)
        
        # Convert to PIL and apply brightness
        pil_image = to_pil(image)
        factor = max(1.0, min(2.0, float(factor)))  # Clamp to valid range
        
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced_pil = enhancer.enhance(factor)
        return to_tensor(enhanced_pil, self.device)


class ContrastAttack(BaseAttack):
    """Contrast adjustment attack with direct factor control"""
    def __init__(self, factor: float = 1.5, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
    
    def attack(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply contrast attack
        
        Args:
            image: Input image tensor [C, H, W]
            factor: Contrast factor (1-2), 1=no change, 2=double contrast
        
        Returns:
            Contrast-adjusted image tensor
        """
        factor = kwargs.get('factor', self.factor)
        
        # Convert to PIL and apply contrast
        pil_image = to_pil(image)
        factor = max(1.0, min(2.0, float(factor)))  # Clamp to valid range
        
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced_pil = enhancer.enhance(factor)
        return to_tensor(enhanced_pil, self.device)


class BlurringAttack(BaseAttack):
    """Gaussian blur attack with direct kernel size control"""
    def __init__(self, kernel_size: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
    
    def attack(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply Gaussian blur attack
        
        Args:
            image: Input image tensor [C, H, W]
            kernel_size: Blur kernel size (0-20), higher values = more blur
        
        Returns:
            Blurred image tensor
        """
        kernel_size = kwargs.get('kernel_size', self.kernel_size)
        
        # Convert to PIL and apply blur
        pil_image = to_pil(image)
        kernel_size = max(0, min(20, float(kernel_size)))  # Clamp to valid range
        
        if kernel_size > 0:
            blurred_pil = pil_image.filter(ImageFilter.GaussianBlur(kernel_size))
        else:
            blurred_pil = pil_image  # No blur if kernel_size is 0
        return to_tensor(blurred_pil, self.device)


class NoiseAttack(BaseAttack):
    """Gaussian noise attack with direct standard deviation control"""
    def __init__(self, std: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.std = std
    
    def attack(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply Gaussian noise attack
        
        Args:
            image: Input image tensor [C, H, W]
            std: Standard deviation of Gaussian noise (0-0.1), higher values = more noise
        
        Returns:
            Noisy image tensor
        """
        std = kwargs.get('std', self.std)
        
        # Add Gaussian noise
        std = max(0, min(0.1, float(std)))  # Clamp to valid range
        noise = torch.randn(image.size(), device=self.device) * std
        noisy_tensor = (image + noise).clamp(0, 1)
        return noisy_tensor


class SaltPepperAttack(BaseAttack):
    """Salt and pepper noise attack with direct probability control"""
    def __init__(self, prob: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.prob = prob
    
    def attack(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply salt and pepper noise attack
        
        Args:
            image: Input image tensor [C, H, W]
            prob: Probability of salt/pepper noise (0-0.1), higher values = more noise
        
        Returns:
            Noisy image tensor with salt and pepper noise
        """
        prob = kwargs.get('prob', self.prob)
        
        # Add salt and pepper noise
        prob = max(0, min(0.1, float(prob)))  # Clamp to valid range
        
        noise_mask = torch.rand(image.size(), device=self.device) < prob
        salt_mask = torch.rand(image.size(), device=self.device) < 0.5
        
        noisy_tensor = image.clone()
        noisy_tensor[noise_mask & salt_mask] = 1.0  # Salt
        noisy_tensor[noise_mask & ~salt_mask] = 0.0  # Pepper
        
        return noisy_tensor


class CompressionAttack(BaseAttack):
    """JPEG compression attack with direct quality factor control"""
    def __init__(self, quality: int = 80, **kwargs):
        super().__init__(**kwargs)
        self.quality = quality
    
    def attack(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply JPEG compression attack
        
        Args:
            image: Input image tensor [C, H, W]
            quality: JPEG quality factor (1-100), lower values = more compression
                    Common values:
                    - 90-100: High quality (light compression)
                    - 70-90: Good quality (moderate compression) 
                    - 50-70: Average quality (standard compression)
                    - 30-50: Low quality (heavy compression)
                    - 10-30: Very low quality (extreme compression)
        
        Returns:
            Compressed image tensor
        """
        quality = kwargs.get('quality', self.quality)
        
        # Convert to PIL and apply compression
        pil_image = to_pil(image)
        quality = max(1, min(100, int(quality)))
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed_pil = Image.open(buffer)
        return to_tensor(compressed_pil, self.device)

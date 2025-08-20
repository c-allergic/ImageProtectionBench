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


class DistortionAttack(BaseAttack):
    """
    Universal Distortion Attack implementing all WAVES distortion types
    
    This class can apply any of the 8 distortion types supported by WAVES
    with exact parameter ranges and strength calculations.
    """
    
    def __init__(self, 
                 distortion_type: str = "rotation",
                 strength: Optional[float] = None,
                 relative_strength: bool = True,
                 distortion_seed: int = 0,
                 **kwargs):
        """
        Initialize distortion attack
        
        Args:
            distortion_type: Type of distortion to apply
            strength: Distortion strength (relative [0,1] or absolute)
            relative_strength: Whether strength is relative [0,1] or absolute
            distortion_seed: Random seed for reproducibility
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        assert distortion_type in DISTORTION_STRENGTH_PARAMS, \
            f"Unsupported distortion type: {distortion_type}"
        
        self.distortion_type = distortion_type
        self.strength = strength
        self.relative_strength = relative_strength
        self.distortion_seed = distortion_seed
    
    def attack(self, 
               image: torch.Tensor,
               distortion_type: str = None,
               strength: float = None,
               distortion_seed: int = None,
               **kwargs) -> torch.Tensor:
        """
        Apply distortion to image
        
        Args:
            image: Input image tensor [C, H, W] in [0, 1] range
            distortion_type: Distortion type (overrides default)
            strength: Distortion strength (overrides default)
            distortion_seed: Random seed (overrides default)
            **kwargs: Additional parameters
            
        Returns:
            Distorted image tensor [C, H, W] in [0, 1] range
        """
        # Use provided parameters or defaults
        dtype = distortion_type if distortion_type is not None else self.distortion_type
        dstrength = strength if strength is not None else self.strength
        seed = distortion_seed if distortion_seed is not None else self.distortion_seed
        
        # Set random seed for reproducibility
        set_random_seed(seed)
        
        # Convert tensor to PIL for processing
        pil_image = to_pil(image)
        
        # Apply the specific distortion
        distorted_pil = self._apply_single_distortion(pil_image, dtype, dstrength)
        
        # Convert back to tensor
        return to_tensor(distorted_pil, self.device)
    
    def _apply_single_distortion(self, 
                                image: Image.Image, 
                                distortion_type: str, 
                                strength: Optional[float]) -> Image.Image:
        """
        Apply single distortion to PIL image following WAVES implementation
        
        Args:
            image: PIL Image
            distortion_type: Type of distortion
            strength: Distortion strength (None for random)
            
        Returns:
            Distorted PIL Image
        """
        # Convert relative strength to absolute if needed
        if strength is not None and self.relative_strength:
            strength = relative_strength_to_absolute(strength, distortion_type)
        
        # Validate strength range
        if strength is not None:
            min_val, max_val = DISTORTION_STRENGTH_PARAMS[distortion_type]
            assert min(min_val, max_val) <= strength <= max(min_val, max_val), \
                f"Strength {strength} out of range for {distortion_type}"
        
        # Apply specific distortion
        if distortion_type == "rotation":
            return self._apply_rotation(image, strength)
        elif distortion_type == "resizedcrop":
            return self._apply_resized_crop(image, strength)
        elif distortion_type == "erasing":
            return self._apply_erasing(image, strength)
        elif distortion_type == "brightness":
            return self._apply_brightness(image, strength)
        elif distortion_type == "contrast":
            return self._apply_contrast(image, strength)
        elif distortion_type == "blurring":
            return self._apply_blurring(image, strength)
        elif distortion_type == "noise":
            return self._apply_noise(image, strength)
        elif distortion_type == "saltpepper":
            return self._apply_salt_pepper_noise(image, strength)
        elif distortion_type == "compression":
            return self._apply_compression(image, strength)
        else:
            raise ValueError(f"Unknown distortion type: {distortion_type}")
    
    def _apply_rotation(self, image: Image.Image, strength: Optional[float]) -> Image.Image:
        """Apply rotation distortion"""
        angle = (strength if strength is not None 
                else random.uniform(*DISTORTION_STRENGTH_PARAMS["rotation"]))
        return F.rotate(image, angle)
    
    def _apply_resized_crop(self, image: Image.Image, strength: Optional[float]) -> Image.Image:
        """Apply resized crop distortion"""
        scale = (strength if strength is not None 
                else random.uniform(*DISTORTION_STRENGTH_PARAMS["resizedcrop"]))
        
        # Get random crop parameters
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=(scale, scale), ratio=(1, 1)
        )
        return F.resized_crop(image, i, j, h, w, image.size)
    
    def _apply_erasing(self, image: Image.Image, strength: Optional[float]) -> Image.Image:
        """Apply random erasing distortion"""
        scale = (strength if strength is not None 
                else random.uniform(*DISTORTION_STRENGTH_PARAMS["erasing"]))
        
        # Convert to tensor for erasing
        tensor = to_tensor(image, self.device).unsqueeze(0)  # Add batch dim
        
        # Get erasing parameters
        i, j, h, w, v = transforms.RandomErasing.get_params(
            tensor, scale=(scale, scale), ratio=(1, 1), value=[0]
        )
        
        # Apply erasing
        erased_tensor = F.erase(tensor, i, j, h, w, v)
        
        # Convert back to PIL
        return to_pil(erased_tensor.squeeze(0))
    
    def _apply_brightness(self, image: Image.Image, strength: Optional[float]) -> Image.Image:
        """Apply brightness adjustment"""
        factor = (strength if strength is not None 
                 else random.uniform(*DISTORTION_STRENGTH_PARAMS["brightness"]))
        
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def _apply_contrast(self, image: Image.Image, strength: Optional[float]) -> Image.Image:
        """Apply contrast adjustment"""
        factor = (strength if strength is not None 
                 else random.uniform(*DISTORTION_STRENGTH_PARAMS["contrast"]))
        
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def _apply_blurring(self, image: Image.Image, strength: Optional[float]) -> Image.Image:
        """Apply Gaussian blur"""
        kernel_size = (int(strength) if strength is not None 
                      else random.uniform(*DISTORTION_STRENGTH_PARAMS["blurring"]))
        
        return image.filter(ImageFilter.GaussianBlur(kernel_size))
    
    def _apply_noise(self, image: Image.Image, strength: Optional[float]) -> Image.Image:
        """Apply Gaussian noise"""
        std = (strength if strength is not None 
               else random.uniform(*DISTORTION_STRENGTH_PARAMS["noise"]))
        
        # Convert to tensor
        tensor = to_tensor(image, self.device)
        
        # Add noise
        noise = torch.randn(tensor.size(), device=self.device) * std
        noisy_tensor = (tensor + noise).clamp(0, 1)
        
        # Convert back to PIL
        return to_pil(noisy_tensor)
    
    def _apply_salt_pepper_noise(self, image: Image.Image, strength: Optional[float]) -> Image.Image:
        """Apply salt and pepper noise"""
        prob = (strength if strength is not None 
                else random.uniform(*DISTORTION_STRENGTH_PARAMS["saltpepper"]))
        
        # Convert to tensor
        tensor = to_tensor(image, self.device)
        
        # Generate random mask for noise locations
        noise_mask = torch.rand(tensor.size(), device=self.device) < prob
        
        # Generate salt/pepper decisions (0.5 probability each)
        salt_mask = torch.rand(tensor.size(), device=self.device) < 0.5
        
        # Apply noise
        noisy_tensor = tensor.clone()
        # Salt (white) noise
        noisy_tensor[noise_mask & salt_mask] = 1.0
        # Pepper (black) noise  
        noisy_tensor[noise_mask & ~salt_mask] = 0.0
        
        # Convert back to PIL
        return to_pil(noisy_tensor)
    
    def _apply_compression(self, image: Image.Image, strength: Optional[float]) -> Image.Image:
        """Apply JPEG compression"""
        quality = (int(strength) if strength is not None 
                  else random.uniform(*DISTORTION_STRENGTH_PARAMS["compression"]))
        
        # Ensure quality is integer and in valid range
        quality = max(1, min(100, int(quality)))
        
        # Apply JPEG compression
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)
    
    def get_parameter_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Get valid parameter ranges for current distortion type"""
        min_val, max_val = DISTORTION_STRENGTH_PARAMS[self.distortion_type]
        
        return {
            "strength": {
                "min": 0.0 if self.relative_strength else min(min_val, max_val),
                "max": 1.0 if self.relative_strength else max(min_val, max_val),
                "default": 0.5 if self.relative_strength else (min_val + max_val) / 2,
                "type": "float"
            },
            "distortion_type": {
                "type": "str",
                "choices": list(DISTORTION_STRENGTH_PARAMS.keys()),
                "default": "rotation"
            },
            "distortion_seed": {
                "min": 0,
                "max": 9999,
                "default": 0,
                "type": "int"
            }
        }


# Individual distortion attack classes for convenience
class RotationAttack(DistortionAttack):
    """Rotation attack with WAVES parameters"""
    def __init__(self, strength: float = 0.5, **kwargs):
        super().__init__(distortion_type="rotation", strength=strength, **kwargs)


class ResizedCropAttack(DistortionAttack):
    """Resized crop attack with WAVES parameters"""
    def __init__(self, strength: float = 0.5, **kwargs):
        super().__init__(distortion_type="resizedcrop", strength=strength, **kwargs)


class ErasingAttack(DistortionAttack):
    """Random erasing attack with WAVES parameters"""
    def __init__(self, strength: float = 0.5, **kwargs):
        super().__init__(distortion_type="erasing", strength=strength, **kwargs)


class BrightnessAttack(DistortionAttack):
    """Brightness adjustment attack with WAVES parameters"""
    def __init__(self, strength: float = 0.5, **kwargs):
        super().__init__(distortion_type="brightness", strength=strength, **kwargs)


class ContrastAttack(DistortionAttack):
    """Contrast adjustment attack with WAVES parameters"""
    def __init__(self, strength: float = 0.5, **kwargs):
        super().__init__(distortion_type="contrast", strength=strength, **kwargs)


class BlurringAttack(DistortionAttack):
    """Gaussian blur attack with WAVES parameters"""
    def __init__(self, strength: float = 0.5, **kwargs):
        super().__init__(distortion_type="blurring", strength=strength, **kwargs)


class NoiseAttack(DistortionAttack):
    """Gaussian noise attack with WAVES parameters"""
    def __init__(self, strength: float = 0.5, **kwargs):
        super().__init__(distortion_type="noise", strength=strength, **kwargs)


class SaltPepperAttack(DistortionAttack):
    """Salt and pepper noise attack with WAVES parameters"""
    def __init__(self, strength: float = 0.5, **kwargs):
        super().__init__(distortion_type="saltpepper", strength=strength, **kwargs)

class CompressionAttack(DistortionAttack):
    """JPEG compression attack with WAVES parameters"""
    def __init__(self, strength: float = 0.5, **kwargs):
        super().__init__(distortion_type="compression", strength=strength, **kwargs)

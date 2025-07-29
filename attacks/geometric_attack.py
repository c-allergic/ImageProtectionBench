"""
Geometric Attack Methods

Implements various geometric transformations including rotation, 
cropping, scaling, and other spatial transformations (wave distortions).
"""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from typing import Dict, Union, Any, Tuple
from .base import BaseAttack
from data import pt_to_pil
from torchvision import transforms

class RotationAttack(BaseAttack):
    """
    Rotation Attack
    
    Rotates the image by a specified angle to test robustness against
    rotational transformations.
    """
    
    def __init__(self, 
                 angle: float = 15.0,
                 fill_color: Tuple[int, int, int] = (0, 0, 0),
                 **kwargs):
        """
        Initialize rotation attack
        
        Args:
            angle: Rotation angle in degrees
            fill_color: Color to fill empty areas after rotation
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.angle = angle
        self.fill_color = fill_color
    
    def attack(self, 
               image: torch.Tensor,
               angle: float = None,
               fill_color: Tuple[int, int, int] = None,
               **kwargs) -> torch.Tensor:
        """
        Apply rotation to image
        
        Args:
            image: Input image tensor [C, H, W] in [0, 1] range
            angle: Rotation angle (overrides default)
            fill_color: Fill color (overrides default)
            **kwargs: Additional parameters
            
        Returns:
            Rotated image tensor [C, H, W] in [0, 1] range
        """
        # Use provided parameters or defaults
        rot_angle = angle if angle is not None else self.angle
        fill = fill_color if fill_color is not None else self.fill_color
        
        # Convert tensor to PIL Image for rotation
        pil_image = pt_to_pil(image.cpu())
        
        # Apply rotation
        rotated_image = pil_image.rotate(rot_angle, fillcolor=fill)
        
        # Convert back to tensor
        return transforms.ToTensor()(rotated_image).to(self.device)
    
    def get_parameter_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Get valid parameter ranges"""
        return {
            "angle": {"min": 0.0, "max": 360.0, "default": 15.0},
            "fill_color": {"type": "tuple", "min": 0, "max": 255, "default": (0, 0, 0)}
        }
    
    # def get_attack_info(self) -> Dict[str, Any]:
    #     """Get attack information"""
    #     return {
    #         "name": "Rotation",
    #         "type": "Geometric",
    #         "description": "Rotates image by specified angle",
    #         "parameters": {
    #             "angle": self.angle,
    #             "fill_color": self.fill_color
    #         }
    #     }


class CropAttack(BaseAttack):
    """
    Crop Attack
    
    Crops a portion of the image to test robustness against
    partial occlusion.
    """
    
    def __init__(self, 
                 crop_ratio: float = 0.8,
                 position: str = "center",
                 **kwargs):
        """
        Initialize crop attack
        
        Args:
            crop_ratio: Ratio of original size to keep (0.8 = keep 80%)
            position: Crop position ('center', 'random', 'top-left', etc.)
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.crop_ratio = crop_ratio
        self.position = position
    
    def attack(self, 
               image: torch.Tensor,
               crop_ratio: float = None,
               position: str = None,
               **kwargs) -> torch.Tensor:
        """
        Apply cropping to image
        
        Args:
            image: Input image tensor [C, H, W] in [0, 1] range
            crop_ratio: Crop ratio (overrides default)
            position: Crop position (overrides default)
            **kwargs: Additional parameters
            
        Returns:
            Cropped and resized image tensor [C, H, W] in [0, 1] range
        """
        # Use provided parameters or defaults
        ratio = crop_ratio if crop_ratio is not None else self.crop_ratio
        pos = position if position is not None else self.position
        
        # Convert tensor to PIL Image for cropping
        pil_image = pt_to_pil(image.cpu())
        
        # Calculate crop size
        original_size = pil_image.size
        crop_width = int(original_size[0] * ratio)
        crop_height = int(original_size[1] * ratio)
        
        # Calculate crop position
        if pos == "center":
            left = (original_size[0] - crop_width) // 2
            top = (original_size[1] - crop_height) // 2
        elif pos == "top-left":
            left, top = 0, 0
        elif pos == "top-right":
            left = original_size[0] - crop_width
            top = 0
        elif pos == "bottom-left":
            left = 0
            top = original_size[1] - crop_height
        elif pos == "bottom-right":
            left = original_size[0] - crop_width
            top = original_size[1] - crop_height
        elif pos == "random":
            left = np.random.randint(0, original_size[0] - crop_width + 1)
            top = np.random.randint(0, original_size[1] - crop_height + 1)
        else:
            # Default to center
            left = (original_size[0] - crop_width) // 2
            top = (original_size[1] - crop_height) // 2
        
        # Apply crop
        right = left + crop_width
        bottom = top + crop_height
        cropped_image = pil_image.crop((left, top, right, bottom))
        
        # Resize back to original size
        resized_image = cropped_image.resize(original_size, Image.BILINEAR)
        
        # Convert back to tensor
        return transforms.ToTensor()(resized_image).to(self.device)
    
    # def get_attack_info(self) -> Dict[str, Any]:
    #     """Get attack information"""
    #     return {
    #         "name": "Crop",
    #         "type": "Geometric", 
    #         "description": "Crops and resizes image",
    #         "parameters": {
    #             "crop_ratio": self.crop_ratio,
    #             "position": self.position
    #         }
    #     }


class ScalingAttack(BaseAttack):
    """
    Scaling Attack
    
    Scales the image by a specified factor to test robustness against
    size changes.
    """
    
    def __init__(self, 
                 scale_factor: float = 0.8,
                 mode: str = "bilinear",
                 **kwargs):
        """
        Initialize scaling attack
        
        Args:
            scale_factor: Scaling factor (1.0 = no change)
            mode: Interpolation mode ('bilinear', 'nearest', etc.)
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.scale_factor = scale_factor
        self.mode = mode
    
    def attack(self, 
               image: torch.Tensor,
               scale_factor: float = None,
               mode: str = None,
               **kwargs) -> torch.Tensor:
        """
        Apply scaling to image
        
        Args:
            image: Input image tensor [C, H, W] in [0, 1] range
            scale_factor: Scaling factor (overrides default)
            mode: Interpolation mode (overrides default)
            **kwargs: Additional parameters
            
        Returns:
            Scaled image tensor [C, H, W] in [0, 1] range
        """
        # Use provided parameters or defaults
        scale = scale_factor if scale_factor is not None else self.scale_factor
        interp_mode = mode if mode is not None else self.mode
        
        # Convert tensor to PIL Image for scaling
        pil_image = pt_to_pil(image.cpu())
        
        # Calculate new size
        original_size = pil_image.size
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        
        # Apply scaling
        if interp_mode == "bilinear":
            resample = Image.BILINEAR
        elif interp_mode == "nearest":
            resample = Image.NEAREST
        else:
            resample = Image.BILINEAR
            
        scaled_image = pil_image.resize(new_size, resample)
        
        # Resize back to original size if needed
        if scale != 1.0:
            scaled_image = scaled_image.resize(original_size, resample)
        
        # Convert back to tensor
        return transforms.ToTensor()(scaled_image).to(self.device)
    
    def get_parameter_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Get valid parameter ranges"""
        return {
            "scale_factor": {"min": 0.0, "max": 2.0, "default": 0.8},
            "mode": {"type": "str", "choices": ["bilinear", "nearest"], "default": "bilinear"}
        }
    
    # def get_attack_info(self) -> Dict[str, Any]:
    #     """Get attack information"""
    #     return {
    #         "name": "Scaling",
    #         "type": "Geometric",
    #         "description": "Scales image by specified factor",
    #         "parameters": {
    #             "scale_factor": self.scale_factor,
    #             "mode": self.mode
    #         }
    #     }


class WaveAttack(BaseAttack):
    """
    Wave Distortion Attack
    
    Applies wave-like distortion to the image to test robustness against
    non-linear geometric transformations.
    """
    
    def __init__(self, 
                 amplitude: float = 10.0,
                 frequency: float = 0.1,
                 direction: str = "horizontal",
                 **kwargs):
        """
        Initialize wave attack
        
        Args:
            amplitude: Wave amplitude in pixels
            frequency: Wave frequency
            direction: Wave direction ('horizontal', 'vertical', 'both')
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.amplitude = amplitude
        self.frequency = frequency
        self.direction = direction
    
    def attack(self, 
               image: torch.Tensor,
               amplitude: float = None,
               frequency: float = None,
               direction: str = None,
               **kwargs) -> torch.Tensor:
        """
        Apply wave distortion to image
        
        Args:
            image: Input image tensor [C, H, W] in [0, 1] range
            amplitude: Wave amplitude (overrides default)
            frequency: Wave frequency (overrides default)
            direction: Wave direction (overrides default)
            **kwargs: Additional parameters
            
        Returns:
            Wave-distorted image tensor [C, H, W] in [0, 1] range
        """
        # Use provided parameters or defaults
        amp = amplitude if amplitude is not None else self.amplitude
        freq = frequency if frequency is not None else self.frequency
        wave_dir = direction if direction is not None else self.direction
        
        # Apply wave distortion directly on tensor
        return self._apply_wave_distortion(image, amp, freq, wave_dir)
    
    def _apply_wave_distortion(self, tensor: torch.Tensor, amp: float, freq: float, direction: str) -> torch.Tensor:
        """Apply wave distortion to tensor"""
        C, H, W = tensor.shape
        
        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=self.device),
            torch.arange(W, dtype=torch.float32, device=self.device),
            indexing='ij'
        )
        
        # Apply wave distortion
        if direction == "horizontal":
            # Horizontal waves
            offset_x = amp * torch.sin(2 * np.pi * freq * y_coords / H)
            new_x = x_coords + offset_x
            new_y = y_coords
        elif direction == "vertical":
            # Vertical waves
            offset_y = amp * torch.sin(2 * np.pi * freq * x_coords / W)
            new_x = x_coords
            new_y = y_coords + offset_y
        else:  # both
            # Both directions
            offset_x = amp * torch.sin(2 * np.pi * freq * y_coords / H)
            offset_y = amp * torch.sin(2 * np.pi * freq * x_coords / W)
            new_x = x_coords + offset_x
            new_y = y_coords + offset_y
        
        # Normalize coordinates to [-1, 1] for grid_sample
        grid_x = 2.0 * new_x / (W - 1) - 1.0
        grid_y = 2.0 * new_y / (H - 1) - 1.0
        
        # Create sampling grid
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]
        
        # Sample from original image
        tensor_batch = tensor.unsqueeze(0)  # [1, C, H, W]
        distorted = F.grid_sample(tensor_batch, grid, mode='bilinear', 
                                padding_mode='border', align_corners=True)
        
        return distorted.squeeze(0)  # [C, H, W]
    
    def get_parameter_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Get valid parameter ranges"""
        return {
            "amplitude": {"min": 0.0, "max": 20.0, "default": 10.0},
            "frequency": {"min": 0.0, "max": 1.0, "default": 0.1},
            "direction": {"type": "str", "choices": ["horizontal", "vertical", "both"], "default": "horizontal"}
        }
    
    # def get_attack_info(self) -> Dict[str, Any]:
    #     """Get attack information"""
    #     return {
    #         "name": "Wave Distortion",
    #         "type": "Geometric",
    #         "description": "Applies wave-like distortion to image",
    #         "parameters": {
    #             "amplitude": self.amplitude,
    #             "frequency": self.frequency,
    #             "direction": self.direction
    #         }
    #     } 
"""
JPEG Compression Attack

Implements JPEG compression attacks to test robustness against 
lossy compression techniques.
"""

import torch
import numpy as np
from PIL import Image
import io
from typing import Dict, Union, Any
from .base import BaseAttack
from data import pt_to_pil
from torchvision import transforms

class JPEGCompressionAttack(BaseAttack):
    """
    JPEG Compression Attack
    
    Applies JPEG compression with specified quality to test robustness
    against lossy compression artifacts.
    """
    
    def __init__(self, 
                 quality: int = 75,
                 **kwargs):
        """
        Initialize JPEG compression attack
        
        Args:
            quality: JPEG quality (1-100, higher = better quality)
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.quality = quality
    
    def attack(self, 
               image: torch.Tensor,
               quality: int = None,
               **kwargs) -> torch.Tensor:
        """
        Apply JPEG compression to image
        
        Args:
            image: Input image tensor [C, H, W] in [0, 1] range
            quality: JPEG quality (overrides default)
            **kwargs: Additional parameters
            
        Returns:
            Compressed image tensor [C, H, W] in [0, 1] range
        """
        # Use provided quality or default
        jpeg_quality = quality if quality is not None else self.quality
        
        # Convert tensor to PIL Image for compression
        pil_image = pt_to_pil(image.cpu())
        
        # Apply JPEG compression
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=jpeg_quality)
        buffer.seek(0)
        compressed_image = Image.open(buffer)
        
        # Convert back to tensor
        return transforms.ToTensor()(compressed_image).to(self.device)
    
    def get_parameter_ranges(self) -> Dict[str, Dict[str, Any]]:
    """Get valid parameter ranges"""
    return {
        "quality": {
            "type": "int",
            "min": 1,
            "max": 100,
            "default": 75,
            "description": "JPEG compression quality"
        }
    } 
    
    # def get_attack_info(self) -> Dict[str, Any]:
    #     """Get attack information"""
    #     return {
    #         "name": "JPEG Compression",
    #         "type": "Compression",
    #         "description": "Applies JPEG compression with specified quality",
    #         "parameters": {
    #             "quality": self.quality
    #         },
    #         "parameter_ranges": {
    #             "quality": {"min": 1, "max": 100, "default": 75}
    #         }
    #     }
    

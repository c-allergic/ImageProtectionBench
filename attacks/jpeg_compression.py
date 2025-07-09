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
               image: Union[Image.Image, torch.Tensor, np.ndarray],
               quality: int = None,
               **kwargs) -> Union[Image.Image, torch.Tensor, np.ndarray]:
        """
        Apply JPEG compression to image
        
        Args:
            image: Input image
            quality: JPEG quality (overrides default)
            **kwargs: Additional parameters
            
        Returns:
            Compressed image in same format as input
        """
        original_format = type(image)
        
        # Convert to PIL Image for compression
        if isinstance(image, Image.Image):
            pil_image = image
        else:
            tensor = self._process_input(image)
            pil_image = self.to_pil(tensor.cpu())
        
        # Use provided quality or default
        jpeg_quality = quality if quality is not None else self.quality
        
        # Apply JPEG compression
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=jpeg_quality)
        buffer.seek(0)
        compressed_image = Image.open(buffer)
        
        # Convert back to original format
        if original_format == Image.Image:
            return compressed_image
        else:
            # Convert back to tensor/array
            tensor = self.to_tensor(compressed_image).to(self.device)
            return self._process_output(tensor, original_format)
    
    def get_attack_info(self) -> Dict[str, Any]:
        """Get attack information"""
        return {
            "name": "JPEG Compression",
            "type": "Compression",
            "description": "Applies JPEG compression with specified quality",
            "parameters": {
                "quality": self.quality
            },
            "parameter_ranges": {
                "quality": {"min": 1, "max": 100, "default": 75}
            }
        }
    
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
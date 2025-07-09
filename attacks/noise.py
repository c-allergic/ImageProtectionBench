"""
Noise-based Attack Methods

Implements various noise attacks to test robustness of image protection.
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Union, Any
from .base import BaseAttack


class GaussianNoiseAttack(BaseAttack):
    """
    Gaussian Noise Attack
    
    Adds Gaussian noise to the input image to test robustness against
    random perturbations.
    """
    
    def __init__(self, 
                 std: float = 0.1,
                 mean: float = 0.0,
                 **kwargs):
        """
        Initialize Gaussian noise attack
        
        Args:
            std: Standard deviation of Gaussian noise
            mean: Mean of Gaussian noise
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.std = std
        self.mean = mean
    
    def attack(self, 
               image: Union[Image.Image, torch.Tensor, np.ndarray],
               std: float = None,
               mean: float = None,
               **kwargs) -> Union[Image.Image, torch.Tensor, np.ndarray]:
        """
        Apply Gaussian noise to image
        
        Args:
            image: Input image
            std: Noise standard deviation (overrides default)
            mean: Noise mean (overrides default)
            **kwargs: Additional parameters
            
        Returns:
            Noisy image in same format as input
        """
        original_format = type(image)
        
        # Convert to tensor
        tensor = self._process_input(image)
        
        # Use provided parameters or defaults
        noise_std = std if std is not None else self.std
        noise_mean = mean if mean is not None else self.mean
        
        # Generate Gaussian noise
        noise = torch.normal(
            mean=noise_mean,
            std=noise_std,
            size=tensor.shape,
            device=self.device
        )
        
        # Add noise to image
        noisy_tensor = tensor + noise
        
        # Convert back to original format
        return self._process_output(noisy_tensor, original_format)
    
    def get_attack_info(self) -> Dict[str, Any]:
        """Get attack information"""
        return {
            "name": "Gaussian Noise",
            "type": "Noise",
            "description": "Adds Gaussian noise to test robustness",
            "parameters": {
                "std": self.std,
                "mean": self.mean
            },
            "parameter_ranges": {
                "std": {"min": 0.0, "max": 1.0, "default": 0.1},
                "mean": {"min": -1.0, "max": 1.0, "default": 0.0}
            }
        }


class SaltPepperAttack(BaseAttack):
    """
    Salt and Pepper Noise Attack
    
    Adds impulse noise (salt and pepper) to the input image.
    """
    
    def __init__(self, 
                 prob: float = 0.05,
                 salt_prob: float = 0.5,
                 **kwargs):
        """
        Initialize salt and pepper attack
        
        Args:
            prob: Probability of noise occurrence
            salt_prob: Probability of salt vs pepper (0.5 = equal)
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.prob = prob
        self.salt_prob = salt_prob
    
    def attack(self, 
               image: Union[Image.Image, torch.Tensor, np.ndarray],
               prob: float = None,
               salt_prob: float = None,
               **kwargs) -> Union[Image.Image, torch.Tensor, np.ndarray]:
        """
        Apply salt and pepper noise to image
        
        Args:
            image: Input image
            prob: Noise probability (overrides default)
            salt_prob: Salt vs pepper probability (overrides default)
            **kwargs: Additional parameters
            
        Returns:
            Noisy image in same format as input
        """
        original_format = type(image)
        
        # Convert to tensor
        tensor = self._process_input(image)
        
        # Use provided parameters or defaults
        noise_prob = prob if prob is not None else self.prob
        s_prob = salt_prob if salt_prob is not None else self.salt_prob
        
        # Generate random mask for noise locations
        noise_mask = torch.rand(tensor.shape, device=self.device) < noise_prob
        
        # Generate salt/pepper decisions
        salt_mask = torch.rand(tensor.shape, device=self.device) < s_prob
        
        # Apply noise
        noisy_tensor = tensor.clone()
        # Salt (white) noise
        noisy_tensor[noise_mask & salt_mask] = 1.0
        # Pepper (black) noise  
        noisy_tensor[noise_mask & ~salt_mask] = 0.0
        
        # Convert back to original format
        return self._process_output(noisy_tensor, original_format)
    
    def get_attack_info(self) -> Dict[str, Any]:
        """Get attack information"""
        return {
            "name": "Salt and Pepper Noise",
            "type": "Noise",
            "description": "Adds impulse noise (salt and pepper)",
            "parameters": {
                "prob": self.prob,
                "salt_prob": self.salt_prob
            },
            "parameter_ranges": {
                "prob": {"min": 0.0, "max": 1.0, "default": 0.05},
                "salt_prob": {"min": 0.0, "max": 1.0, "default": 0.5}
            }
        } 
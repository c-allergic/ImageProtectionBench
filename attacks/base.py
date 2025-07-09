"""
Base Attack Class for ImageProtectionBench

Defines the common interface and utilities for all attack methods.
"""

import torch
import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Any
import torchvision.transforms as transforms


class BaseAttack(ABC):
    """
    Base class for all attack methods
    
    This class defines the common interface that all attack methods should implement.
    It provides utilities for image processing and parameter validation.
    """
    
    def __init__(self, 
                 device: str = "auto",
                 **kwargs):
        """
        Initialize base attack
        
        Args:
            device: Device to run attack on ('cpu', 'cuda', or 'auto')
            **kwargs: Additional attack-specific parameters
        """
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Store attack parameters
        self.params = kwargs
        
        # Initialize image transforms
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
        
    @abstractmethod
    def attack(self, 
               image: Union[Image.Image, torch.Tensor, np.ndarray],
               **kwargs) -> Union[Image.Image, torch.Tensor, np.ndarray]:
        """
        Apply attack to input image
        
        Args:
            image: Input image in various formats
            **kwargs: Attack-specific parameters
            
        Returns:
            Attacked image in same format as input
        """
        pass
    
    @abstractmethod
    def get_attack_info(self) -> Dict[str, Any]:
        """
        Get information about the attack method
        
        Returns:
            Dictionary containing attack information
        """
        pass
    
    def _process_input(self, 
                      image: Union[Image.Image, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Convert input image to tensor format
        
        Args:
            image: Input image
            
        Returns:
            Image as tensor [C, H, W] with values in [0, 1]
        """
        if isinstance(image, Image.Image):
            # PIL Image to tensor
            tensor = self.to_tensor(image)
        elif isinstance(image, np.ndarray):
            # Numpy array to tensor
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            if len(image.shape) == 3 and image.shape[2] == 3:
                # HWC to CHW
                tensor = torch.from_numpy(image.transpose(2, 0, 1))
            else:
                tensor = torch.from_numpy(image)
        elif isinstance(image, torch.Tensor):
            tensor = image.clone()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Ensure tensor is in [0, 1] range
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
            
        return tensor.to(self.device)
    
    def _process_output(self,
                       tensor: torch.Tensor,
                       original_format: type) -> Union[Image.Image, torch.Tensor, np.ndarray]:
        """
        Convert tensor back to original format
        
        Args:
            tensor: Output tensor [C, H, W] with values in [0, 1]
            original_format: Type of original input
            
        Returns:
            Image in original format
        """
        # Clamp values to [0, 1]
        tensor = torch.clamp(tensor, 0, 1)
        
        if original_format == Image.Image:
            # Convert to PIL Image
            return self.to_pil(tensor.cpu())
        elif original_format == np.ndarray:
            # Convert to numpy array
            if len(tensor.shape) == 3:
                # CHW to HWC
                array = tensor.cpu().numpy().transpose(1, 2, 0)
            else:
                array = tensor.cpu().numpy()
            return (array * 255).astype(np.uint8)
        else:
            # Return as tensor
            return tensor
    
    def batch_attack(self,
                    images: List[Union[Image.Image, torch.Tensor, np.ndarray]],
                    **kwargs) -> List[Union[Image.Image, torch.Tensor, np.ndarray]]:
        """
        Apply attack to a batch of images
        
        Args:
            images: List of input images
            **kwargs: Attack-specific parameters
            
        Returns:
            List of attacked images
        """
        attacked_images = []
        for image in images:
            attacked_image = self.attack(image, **kwargs)
            attacked_images.append(attacked_image)
        return attacked_images
    
    def validate_parameters(self, **kwargs) -> Dict[str, Any]:
        """
        Validate and process attack parameters
        
        Args:
            **kwargs: Attack parameters to validate
            
        Returns:
            Validated parameters dictionary
        """
        # Default implementation - subclasses should override
        validated_params = self.params.copy()
        validated_params.update(kwargs)
        return validated_params
    
    def get_parameter_ranges(self) -> Dict[str, Dict[str, Any]]:
        """
        Get valid parameter ranges for this attack
        
        Returns:
            Dictionary mapping parameter names to their valid ranges
        """
        # Default implementation - subclasses should override
        return {}
    
    def __str__(self) -> str:
        """String representation of the attack"""
        info = self.get_attack_info()
        return f"{info['name']} (Type: {info['type']})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"{self.__class__.__name__}(device='{self.device}', params={self.params})" 
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
               image: torch.Tensor,
               **kwargs) -> torch.Tensor:
        """
        Apply attack to input image
        
        Args:
            image: Input image in various formats
            **kwargs: Attack-specific parameters
            
        Returns:
            Attacked image in same format as input
        """
        pass
    
    def attack_multiple(
        self, 
        images: Union[torch.Tensor, List[torch.Tensor]], 
        **kwargs
    ) -> torch.Tensor:
        """
        对多张图片进行攻击
        
        Args:
            images: 图片张量 [B, C, H, W] 或图片张量列表
            **kwargs: 传递给attack方法的其他参数    
        Returns:
            被攻击的图片张量 [B, C, H, W]
        """
        # 处理输入格式
        if isinstance(images, list):
            images = torch.stack(images)
        
        if len(images.shape) == 3:
            # 单张图片，添加批次维度
            images = images.unsqueeze(0)
        
        images = images.to(self.device)
        attacked_images = []
        total_images = images.size(0)
        
        for i in range(total_images):
            single_image = images[i]
            attacked_single = self.attack(single_image, **kwargs)
            attacked_images.append(attacked_single)
        
        return torch.stack(attacked_images)
    
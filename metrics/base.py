"""
Base classes for metrics in ImageProtectionBench

Provides unified interfaces for all metric types.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List
import torch
import numpy as np
from PIL import Image


class BaseMetric(ABC):
    """
    Base class for all metrics in ImageProtectionBench
    
    Provides unified interface for image quality, video quality, 
    and attack effectiveness metrics.
    """
    
    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
    
    @abstractmethod
    def compute(self, *args, **kwargs) -> Union[float, Dict[str, float]]:
        """
        Main computation method for the metric
        
        Returns:
            Either a single float value or dict of metric values
        """
        pass

class ImageQualityMetric(BaseMetric):
    """
    Base class for image quality metrics
    
    These metrics compare two images and return a quality score.
    """
    
    @abstractmethod
    def compute(self, 
                image1: torch.Tensor,
                image2: torch.Tensor,
                **kwargs) -> float:
        """
        Compute image quality metric between two images
        
        Args:
            image1: Reference image
            image2: Comparison image
            **kwargs: Additional parameters
            
        Returns:
            Quality score as float
        """
        pass


class VideoQualityMetric(BaseMetric):
    """
    Base class for video quality metrics
    
    These metrics evaluate the quality of video sequences.
    """
    
    @abstractmethod  
    def compute(self,
                video: torch.Tensor,
                **kwargs) -> Union[float, Dict[str, float]]:
        """
        Compute video quality metric
        
        Args:
            video: Input video frames
            reference: Reference video (optional)
            **kwargs: Additional parameters
            
        Returns:
            Quality score(s) as float or dict
        """
        pass


class EffectivenessMetric(BaseMetric):
    """
    Base class for attack/protection effectiveness metrics
    
    These metrics evaluate how effective attacks or protections are.
    """
    
    @abstractmethod
    def compute(self, *args, **kwargs) -> Union[float, Dict[str, float]]:
        """
        Compute effectiveness metric
        
        Returns:
            Effectiveness score(s) as float or dict
        """
        pass 
    



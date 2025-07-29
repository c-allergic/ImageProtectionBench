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
    


class TimeMetric(BaseMetric):
    """
    时间测量Metric
    
    从保护方法的timeit装饰器中收集时间信息，
    计算平均保护时长和总时长。
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def compute(self, protection_method=None, **kwargs) -> Dict[str, float]:
        """
        计算时间统计信息
        
        Args:
            protection_method: 保护方法对象，从中获取时间记录
            **kwargs: 其他参数
            
        Returns:
            包含平均保护时长和总时长的字典
        """
        # 检查是否能获取有效的时间记录
        if (protection_method is None or 
            not hasattr(protection_method, '_timing_records') or
            not protection_method._timing_records):
            return {
                "total_protection_time": 0.0,
                "average_protection_time": 0.0,
                "total_images_processed": 0,
                "images_per_second": 0.0
            }
            
        # 获取protect_multiple方法的记录
        protect_records = [r for r in protection_method._timing_records 
                         if r['method'] == 'protect_multiple']
        
        # 如果没有protect_multiple记录则返回0
        if not protect_records:
            return {
                "total_protection_time": 0.0,
                "average_protection_time": 0.0,
                "total_images_processed": 0,
                "images_per_second": 0.0
            }
        
        # 计算统计信息
        total_time = sum(r['elapsed_time'] for r in protect_records)
        total_images = sum(r['num_images'] for r in protect_records)
        average_time = total_time / len(protect_records) if protect_records else 0.0
        images_per_second = total_images / total_time if total_time > 0 else 0.0
        
        return {
            "total_protection_time": float(total_time),
            "total_images_processed": int(total_images),
            "images_per_second": float(images_per_second)
        }
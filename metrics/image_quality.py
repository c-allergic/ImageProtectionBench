"""
Image Quality Metrics

Implements various image quality assessment metrics including PSNR, SSIM, and LPIPS
for evaluating the impact of protection methods on image quality.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Dict, Union, List, Any
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .base import ImageQualityMetric


class PSNRMetric(ImageQualityMetric):
    """
    Peak Signal-to-Noise Ratio (PSNR) Metric
    
    Measures the ratio between the maximum possible power of a signal
    and the power of corrupting noise.
    """
    
    def __init__(self, max_val: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.max_val = max_val

    def compute(self, 
                image1: torch.Tensor,
                image2: torch.Tensor,
                **kwargs) -> float:
        """
        Compute PSNR between two images
        
        Args:
            image1: Reference image
            image2: Comparison image
            **kwargs: Additional parameters
            
        Returns:
            PSNR value in dB
        """
        # Convert to numpy arrays for skimage
        array1 = image1.cpu().numpy().transpose(1, 2, 0)
            
        array2 = image2.cpu().numpy().transpose(1, 2, 0)
        
        # Compute PSNR
        psnr_value = peak_signal_noise_ratio(
            array1, array2, 
            data_range=self.max_val
        )
        
        return float(psnr_value)
    
    def compute_multiple(self, 
                        original_images: torch.Tensor,
                        protected_images: torch.Tensor,
                        **kwargs) -> Dict[str, Any]:
        """
        Compute PSNR for multiple image pairs
        
        Args:
            original_images: Original images tensor [B, C, H, W] in [0, 1] range
            protected_images: Protected images tensor [B, C, H, W] in [0, 1] range
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with PSNR statistics and individual scores
        """
        batch_size = original_images.size(0)
        psnr_scores = []
        
        for i in range(batch_size):
            # Extract single image from batch [C, H, W]
            orig_image = original_images[i]
            prot_image = protected_images[i]
            
            # Compute PSNR for this image pair
            psnr_score = self.compute(orig_image, prot_image, **kwargs)
            psnr_scores.append(psnr_score)
        
        # Aggregate results - 只保留average指标
        psnr_scores = np.array(psnr_scores)
        return {
            "average_psnr": float(np.mean(psnr_scores))
            # "max_psnr": float(np.max(psnr_scores)),
            # "min_psnr": float(np.min(psnr_scores)),
            # "std_psnr": float(np.std(psnr_scores))
        }
    
class SSIMMetric(ImageQualityMetric):
    """
    Structural Similarity Index (SSIM) Metric
    
    Measures the structural similarity between two images based on 
    luminance, contrast, and structure comparisons.
    """
    
    def __init__(self, 
                 window_size: int = 11,
                 data_range: float = 1.0,
                 multichannel: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.data_range = data_range
        self.multichannel = multichannel
    
    def compute(self, 
                image1: torch.Tensor,
                image2: torch.Tensor,
                **kwargs) -> float:
        """
        Compute SSIM between two images
        
        Args:
            image1: Reference image
            image2: Comparison image
            **kwargs: Additional parameters
            
        Returns:
            SSIM value between -1 and 1
        """
        # Convert to numpy arrays for skimage
        array1 = image1.cpu().numpy().transpose(1, 2, 0)
            
        array2 = image2.cpu().numpy().transpose(1, 2, 0)
        
        # Handle grayscale images
        if array1.shape[-1] == 1:
            array1 = array1.squeeze(-1)
            array2 = array2.squeeze(-1)
            multichannel = False
        else:
            multichannel = self.multichannel
        
        # Compute SSIM
        ssim_value = structural_similarity(
            array1, array2,
            win_size=self.window_size,
            data_range=self.data_range,
            channel_axis=-1 if multichannel else None
        )
        
        return float(ssim_value)
    
    def compute_multiple(self, 
                        original_images: torch.Tensor,
                        protected_images: torch.Tensor,
                        **kwargs) -> Dict[str, Any]:
        """
        Compute SSIM for multiple image pairs
        
        Args:
            original_images: Original images tensor [B, C, H, W] in [0, 1] range
            protected_images: Protected images tensor [B, C, H, W] in [0, 1] range
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with SSIM statistics and individual scores
        """
        batch_size = original_images.size(0)
        ssim_scores = []
        
        for i in range(batch_size):
            # Extract single image from batch [C, H, W]
            orig_image = original_images[i]
            prot_image = protected_images[i]
            
            # Compute SSIM for this image pair
            ssim_score = self.compute(orig_image, prot_image, **kwargs)
            ssim_scores.append(ssim_score)
        
        # Aggregate results - 只保留average指标
        ssim_scores = np.array(ssim_scores)
        return {
            "average_ssim": float(np.mean(ssim_scores))
            # "max_ssim": float(np.max(ssim_scores)),
            # "min_ssim": float(np.min(ssim_scores)),
            # "std_ssim": float(np.std(ssim_scores))
        }

class LPIPSMetric(ImageQualityMetric):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) Metric
    
    Uses deep neural networks to measure perceptual similarity between images.
    """
    
    def __init__(self, 
                 network: str = "alex",
                 **kwargs):
        super().__init__(**kwargs)
        self.network = network
        self._load_model()
    
    def _load_model(self):
        """Load LPIPS model"""
        try:
            import lpips
            self.model = lpips.LPIPS(net=self.network).to(self.device)
            self.model.eval()
            print(f"LPIPS model ({self.network}) loaded successfully")
        except ImportError:
            print("LPIPS not available. Install with: pip install lpips")
            self.model = None
        except Exception as e:
            print(f"Error loading LPIPS model: {e}")
            self.model = None
    
    def compute(self, 
                image1: torch.Tensor,
                image2: torch.Tensor,
                **kwargs) -> float:
        """
        Compute LPIPS between two images
        
        Args:
            image1: Reference image
            image2: Comparison image
            **kwargs: Additional parameters
            
        Returns:
            LPIPS distance value
        """
        if self.model is None:
            print("LPIPS model not available, returning placeholder value")
            return 0.0
                
        # Add batch dimension and normalize to [-1, 1]
        tensor1 = image1.unsqueeze(0) * 2.0 - 1.0
        tensor2 = image2.unsqueeze(0) * 2.0 - 1.0
        
        # Compute LPIPS
        with torch.no_grad():
            lpips_value = self.model(tensor1, tensor2)
        
        return float(lpips_value.item())
    
    def compute_multiple(self, 
                        original_images: torch.Tensor,
                        protected_images: torch.Tensor,
                        **kwargs) -> Dict[str, Any]:
        """
        Compute LPIPS for multiple image pairs
        
        Args:
            original_images: Original images tensor [B, C, H, W] in [0, 1] range
            protected_images: Protected images tensor [B, C, H, W] in [0, 1] range
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with LPIPS statistics and individual scores
        """
        if self.model is None:
            batch_size = original_images.size(0)
            return {
                "average_lpips": 0.0
                # "max_lpips": 0.0,
                # "min_lpips": 0.0,
                # "std_lpips": 0.0
            }
        
        batch_size = original_images.size(0)
        lpips_scores = []
        
        for i in range(batch_size):
            # Extract single image from batch [C, H, W]
            orig_image = original_images[i]
            prot_image = protected_images[i]
            
            # Compute LPIPS for this image pair
            lpips_score = self.compute(orig_image, prot_image, **kwargs)
            lpips_scores.append(lpips_score)
        
        # Aggregate results
        lpips_scores = np.array(lpips_scores)
        return {
            "average_lpips": float(np.mean(lpips_scores))
            # "max_lpips": float(np.max(lpips_scores)),
            # "min_lpips": float(np.min(lpips_scores)),
            # "std_lpips": float(np.std(lpips_scores))
        }
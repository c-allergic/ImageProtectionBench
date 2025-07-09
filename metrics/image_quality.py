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


class BaseImageMetric:
    """Base class for image quality metrics"""
    
    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.to_tensor = transforms.ToTensor()
    
    def _process_image(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Convert image to tensor format"""
        if isinstance(image, Image.Image):
            tensor = self.to_tensor(image)
        elif isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            if len(image.shape) == 3 and image.shape[2] == 3:
                tensor = torch.from_numpy(image.transpose(2, 0, 1))
            else:
                tensor = torch.from_numpy(image)
        elif isinstance(image, torch.Tensor):
            tensor = image.clone()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
            
        return tensor.to(self.device)
    
    def compute(self, image1, image2, **kwargs):
        """Compute metric between two images"""
        raise NotImplementedError


class PSNRMetric(BaseImageMetric):
    """
    Peak Signal-to-Noise Ratio (PSNR) Metric
    
    Measures the ratio between the maximum possible power of a signal
    and the power of corrupting noise.
    """
    
    def __init__(self, max_val: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.max_val = max_val
    
    def compute(self, 
                image1: Union[Image.Image, torch.Tensor, np.ndarray],
                image2: Union[Image.Image, torch.Tensor, np.ndarray],
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
        if isinstance(image1, (torch.Tensor, Image.Image)):
            tensor1 = self._process_image(image1)
            array1 = tensor1.cpu().numpy().transpose(1, 2, 0)
        else:
            array1 = image1
            
        if isinstance(image2, (torch.Tensor, Image.Image)):
            tensor2 = self._process_image(image2)
            array2 = tensor2.cpu().numpy().transpose(1, 2, 0)
        else:
            array2 = image2
        
        # Compute PSNR
        psnr_value = peak_signal_noise_ratio(
            array1, array2, 
            data_range=self.max_val
        )
        
        return float(psnr_value)
    
    def batch_compute(self, 
                     images1: List[Union[Image.Image, torch.Tensor, np.ndarray]],
                     images2: List[Union[Image.Image, torch.Tensor, np.ndarray]]) -> List[float]:
        """Compute PSNR for batch of image pairs"""
        psnr_values = []
        for img1, img2 in zip(images1, images2):
            psnr_values.append(self.compute(img1, img2))
        return psnr_values
    
    def get_metric_info(self) -> Dict[str, Any]:
        """Get metric information"""
        return {
            "name": "PSNR",
            "description": "Peak Signal-to-Noise Ratio",
            "range": [0, float('inf')],
            "higher_is_better": True,
            "unit": "dB"
        }


class SSIMMetric(BaseImageMetric):
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
                image1: Union[Image.Image, torch.Tensor, np.ndarray],
                image2: Union[Image.Image, torch.Tensor, np.ndarray],
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
        if isinstance(image1, (torch.Tensor, Image.Image)):
            tensor1 = self._process_image(image1)
            array1 = tensor1.cpu().numpy().transpose(1, 2, 0)
        else:
            array1 = image1
            
        if isinstance(image2, (torch.Tensor, Image.Image)):
            tensor2 = self._process_image(image2)
            array2 = tensor2.cpu().numpy().transpose(1, 2, 0)
        else:
            array2 = image2
        
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
    
    def batch_compute(self, 
                     images1: List[Union[Image.Image, torch.Tensor, np.ndarray]],
                     images2: List[Union[Image.Image, torch.Tensor, np.ndarray]]) -> List[float]:
        """Compute SSIM for batch of image pairs"""
        ssim_values = []
        for img1, img2 in zip(images1, images2):
            ssim_values.append(self.compute(img1, img2))
        return ssim_values
    
    def get_metric_info(self) -> Dict[str, Any]:
        """Get metric information"""
        return {
            "name": "SSIM",
            "description": "Structural Similarity Index",
            "range": [-1, 1],
            "higher_is_better": True,
            "unit": "similarity"
        }


class LPIPSMetric(BaseImageMetric):
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
                image1: Union[Image.Image, torch.Tensor, np.ndarray],
                image2: Union[Image.Image, torch.Tensor, np.ndarray],
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
        
        # Convert to tensors
        tensor1 = self._process_image(image1)
        tensor2 = self._process_image(image2)
        
        # Add batch dimension and normalize to [-1, 1]
        tensor1 = tensor1.unsqueeze(0) * 2.0 - 1.0
        tensor2 = tensor2.unsqueeze(0) * 2.0 - 1.0
        
        # Compute LPIPS
        with torch.no_grad():
            lpips_value = self.model(tensor1, tensor2)
        
        return float(lpips_value.item())
    
    def batch_compute(self, 
                     images1: List[Union[Image.Image, torch.Tensor, np.ndarray]],
                     images2: List[Union[Image.Image, torch.Tensor, np.ndarray]]) -> List[float]:
        """Compute LPIPS for batch of image pairs"""
        if self.model is None:
            return [0.0] * len(images1)
        
        lpips_values = []
        for img1, img2 in zip(images1, images2):
            lpips_values.append(self.compute(img1, img2))
        return lpips_values
    
    def get_metric_info(self) -> Dict[str, Any]:
        """Get metric information"""
        return {
            "name": "LPIPS",
            "description": "Learned Perceptual Image Patch Similarity",
            "range": [0, float('inf')],
            "higher_is_better": False,
            "unit": "distance"
        } 
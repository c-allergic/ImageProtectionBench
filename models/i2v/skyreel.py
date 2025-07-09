"""
Skyreel Image-to-Video Model Implementation
Based on Skyreel diffusion model for video generation
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Union, Tuple
from .base import BaseI2VModel


class SkyreelModel(BaseI2VModel):
    """
    Skyreel Image-to-Video Generation Model
    
    A high-quality image-to-video generation model based on diffusion architecture.
    Supports efficient video generation with good temporal consistency.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = "auto",
                 dtype: torch.dtype = torch.float16,
                 cache_dir: Optional[str] = None,
                 **kwargs):
        """
        Initialize Skyreel model
        
        Args:
            model_path: Path to model weights or HuggingFace model ID
            device: Device to run model on
            dtype: Model data type
            cache_dir: Directory to cache model files
            **kwargs: Additional model configuration
        """
        super().__init__(model_path, device, dtype, cache_dir, **kwargs)
        
        self.model_name = "skyreel"
        self.max_frames = kwargs.get('max_frames', 16)
        self.default_height = kwargs.get('height', 512)
        self.default_width = kwargs.get('width', 512)
        self.default_fps = kwargs.get('fps', 8)
        
        # Model configuration
        self.guidance_scale = kwargs.get('guidance_scale', 7.5)
        self.num_inference_steps = kwargs.get('num_inference_steps', 50)
        
    def load_model(self) -> None:
        """Load the Skyreel model"""
        try:
            # Import Skyreel-specific dependencies
            from diffusers import DiffusionPipeline
            
            print(f"Loading Skyreel model from {self.model_path}")
            
            # Load pipeline
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.model_path or "skyreel/skyreel-v1",
                torch_dtype=self.dtype,
                cache_dir=self.cache_dir,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable memory efficient attention if available
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing()
            
            if hasattr(self.pipeline, 'enable_vae_slicing'):
                self.pipeline.enable_vae_slicing()
                
            self.is_loaded = True
            print(f"Skyreel model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading Skyreel model: {e}")
            print("Note: Skyreel model may require specific installation or access")
            self.is_loaded = False
    
    def generate_video(self,
                      image: Union[Image.Image, str, torch.Tensor],
                      prompt: str = "",
                      negative_prompt: str = "",
                      num_frames: int = 16,
                      height: Optional[int] = None,
                      width: Optional[int] = None,
                      fps: int = 8,
                      guidance_scale: Optional[float] = None,
                      num_inference_steps: Optional[int] = None,
                      seed: Optional[int] = None,
                      **kwargs) -> List[Image.Image]:
        """
        Generate video from input image
        
        Args:
            image: Input image (PIL Image, path, or tensor)
            prompt: Text prompt to guide generation
            negative_prompt: Negative text prompt
            num_frames: Number of frames to generate
            height: Output video height
            width: Output video width
            fps: Frames per second
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of denoising steps
            seed: Random seed for reproducibility
            **kwargs: Additional generation parameters
            
        Returns:
            List of PIL Images representing video frames
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Set generation parameters
        guidance_scale = guidance_scale or self.guidance_scale
        num_inference_steps = num_inference_steps or self.num_inference_steps
        height = height or self.default_height
        width = width or self.default_width
        num_frames = min(num_frames, self.max_frames)
        
        # Process input image
        input_image = self._process_input_image(image, height, width)
        
        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        try:
            # Generate video
            with torch.inference_mode():
                result = self.pipeline(
                    image=input_image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=torch.Generator(device=self.device).manual_seed(seed) if seed else None,
                    **kwargs
                )
            
            # Extract frames
            if hasattr(result, 'frames'):
                frames = result.frames[0]  # First (and only) video
            elif isinstance(result, torch.Tensor):
                # Convert tensor to PIL images
                frames = self._tensor_to_pil_list(result)
            else:
                frames = result
            
            return frames
            
        except Exception as e:
            print(f"Error during video generation: {e}")
            return []
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "name": "Skyreel",
            "type": "Image-to-Video",
            "architecture": "Diffusion-based",
            "max_frames": self.max_frames,
            "default_resolution": f"{self.default_width}x{self.default_height}",
            "default_fps": self.default_fps,
            "features": [
                "High-quality video generation",
                "Good temporal consistency", 
                "Efficient processing",
                "Text-guided generation"
            ],
            "paper": "Skyreel: High-Quality Image-to-Video Generation",
            "repository": "https://github.com/skyreel/skyreel"
        }
    
    def estimate_memory_usage(self, 
                            num_frames: int = 16,
                            height: int = 512, 
                            width: int = 512) -> Dict[str, float]:
        """
        Estimate memory usage for video generation
        
        Args:
            num_frames: Number of frames to generate
            height: Video height
            width: Video width
            
        Returns:
            Dictionary with memory estimates in GB
        """
        # Base model memory
        base_memory = 4.0  # GB
        
        # Video memory (frames × resolution × channels × precision)
        frame_memory = (num_frames * height * width * 3 * 4) / (1024**3)  # 4 bytes for float32
        
        # Intermediate activations (estimated)
        activation_memory = frame_memory * 2
        
        return {
            "model": base_memory,
            "frames": frame_memory,
            "activations": activation_memory,
            "total": base_memory + frame_memory + activation_memory
        }
    
    def cleanup(self) -> None:
        """Clean up model resources"""
        if hasattr(self, 'pipeline'):
            del self.pipeline
        self.is_loaded = False
        torch.cuda.empty_cache() 
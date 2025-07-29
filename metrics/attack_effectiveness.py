"""
Attack Effectiveness Metrics

Implements metrics for evaluating the effectiveness of attacks against
image protection methods, including CLIP scores and success rates.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Dict, Union, List, Any, Optional
import torchvision.transforms as transforms

from .base import EffectivenessMetric
from data import pt_to_pil


class CLIPScoreMetric(EffectivenessMetric):
    """
    CLIP Score Metric
    
    Measures semantic similarity between original and protected video frames using CLIP model.
    Used to evaluate whether protected videos maintain semantic alignment with original videos.
    """
    
    def __init__(self, 
                 model_name: str = "ViT-B/32",
                 **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self._load_model()
    
    def _load_model(self):
        """Load CLIP model"""
        import clip
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.model.eval()
        print(f"CLIP model ({self.model_name}) loaded successfully")
    
    def compute(self, 
                original_video: torch.Tensor,
                protected_video: torch.Tensor,
                **kwargs) -> Dict[str, float]:
        """
        Compute CLIP scores between one original and one protected video
        
        Args:
            original_video: Original video tensor [T, C, H, W] in [0, 1] range
            protected_video: Protected video tensor [T, C, H, W] in [0, 1] range
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with CLIP score statistics for this video pair
        """
        if self.model is None:
            print("CLIP model not available, returning placeholder values")
            return None
        
        # Ensure videos have same number of frames
        num_frames = min(original_video.size(0), protected_video.size(0))
        original_video = original_video[:num_frames]
        protected_video = protected_video[:num_frames]
        
        frame_scores = []
        
        with torch.no_grad():
            # Process each frame pair
            for i in range(num_frames):
                orig_frame = original_video[i]  # [C, H, W]
                prot_frame = protected_video[i]  # [C, H, W]
                
                # Convert frames to PIL for CLIP preprocessing using pt_to_pil
                orig_pil = pt_to_pil(orig_frame)
                prot_pil = pt_to_pil(prot_frame)
                
                # Preprocess for CLIP
                orig_input = self.preprocess(orig_pil).unsqueeze(0).to(self.device)
                prot_input = self.preprocess(prot_pil).unsqueeze(0).to(self.device)
                
                # Get features
                orig_features = self.model.encode_image(orig_input)
                prot_features = self.model.encode_image(prot_input)
                
                # Normalize features
                orig_features = orig_features / orig_features.norm(dim=-1, keepdim=True)
                prot_features = prot_features / prot_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity
                similarity = torch.cosine_similarity(orig_features, prot_features, dim=-1)
                frame_scores.append(similarity.item())
        
        # Compute statistics
        frame_scores = np.array(frame_scores)
        return {
            "average_clip_score": float(np.mean(frame_scores)),
            # "max_clip_score": float(np.max(frame_scores)),
            # "min_clip_score": float(np.min(frame_scores)),
            # "std_clip_score": float(np.std(frame_scores)),
            "total_frames": len(frame_scores)
        }
    
    def compute_multiple(self, 
                        original_videos: torch.Tensor,
                        protected_videos: torch.Tensor,
                        **kwargs) -> Dict[str, float]:
        """
        Compute CLIP scores between multiple video pairs
        
        Args:
            original_videos: Original videos tensor [B, T, C, H, W] in [0, 1] range
            protected_videos: Protected videos tensor [B, T, C, H, W] in [0, 1] range
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with aggregated CLIP score statistics across all video pairs
        """
        batch_size = original_videos.size(0)
        all_scores = []
        
        for i in range(batch_size):
            # Extract single video from batch [T, C, H, W]
            orig_video = original_videos[i]
            prot_video = protected_videos[i]
            
            # Compute CLIP score for this video pair
            video_result = self.compute(orig_video, prot_video, **kwargs)
            all_scores.append(video_result["average_clip_score"])
        
        # Aggregate results across all videos - 只保留average指标
        all_scores = np.array(all_scores)
        return {
            "average_clip_score": float(np.mean(all_scores)),
            "total_videos": len(all_scores)
            # "max_clip_score": float(np.max(all_scores)),
            # "min_clip_score": float(np.min(all_scores)),
            # "std_clip_score": float(np.std(all_scores))
        }

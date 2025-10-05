"""
Effectiveness Metrics

Implements metrics for evaluating the effectiveness of image protection methods, including CLIP scores and success rates.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Dict, Union, List, Any, Optional
import torchvision.transforms as transforms
import clip
from .base import EffectivenessMetric
from data import pt_to_pil


class CLIPVideoScoreMetric(EffectivenessMetric):
    """
    CLIP Video Score Metric
    
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
            "clip_score": float(np.mean(frame_scores)),
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
            all_scores.append(video_result["clip_score"])
        
        # Aggregate results across all videos - 只保留average指标
        all_scores = np.array(all_scores)
        return {
            "clip_score": float(np.mean(all_scores)),
            # "max_clip_score": float(np.max(all_scores)),
            # "min_clip_score": float(np.min(all_scores)),
            # "std_clip_score": float(np.std(all_scores))
        }
    
    def compute_upper_bound(self, videos: torch.Tensor, sample_size: int = 10) -> float:
        """
        计算CLIP分数的理论上限：视频与自身对比
        
        Args:
            videos: 视频张量 [B, T, C, H, W]
            sample_size: 采样大小，避免计算过多
            
        Returns:
            上限分数的平均值
        """
        batch_size = videos.size(0)
        sample_size = min(sample_size, batch_size)
        
        # 随机采样一部分视频
        indices = np.random.choice(batch_size, sample_size, replace=False)
        sampled_videos = videos[indices]
        
        upper_scores = []
        with torch.no_grad():
            for i in range(sample_size):
                video = sampled_videos[i]
                # 视频与自身对比
                result = self.compute(video, video)
                if result:
                    upper_scores.append(result["clip_score"])
        
        return float(np.mean(upper_scores)) if upper_scores else 1.0
    
    def compute_lower_bound(self, videos: torch.Tensor, sample_size: int = 10) -> float:
        """
        计算CLIP分数的理论下限：随机无关视频对比
        
        Args:
            videos: 视频张量 [B, T, C, H, W]  
            sample_size: 采样大小
            
        Returns:
            下限分数的平均值
        """
        batch_size = videos.size(0)
        if batch_size < 2:
            return 0.0
            
        # 确保有足够的视频进行配对
        sample_size = min(sample_size, batch_size * (batch_size - 1) // 2)
        
        lower_scores = []
        with torch.no_grad():
            for _ in range(sample_size):
                # 随机选择两个不同的视频
                i, j = np.random.choice(batch_size, 2, replace=False)
                video1 = videos[i]
                video2 = videos[j]
                
                # 计算无关视频间的CLIP分数
                result = self.compute(video1, video2)
                if result:
                    lower_scores.append(result["clip_score"])
        
        return float(np.mean(lower_scores)) if lower_scores else 0.0


class CLIPVideoTextScoreMetric(EffectivenessMetric):
    """
    CLIP Video-Text Score Metric
    
    Measures semantic similarity between video frames and text prompts using CLIP model.
    Used to evaluate whether generated videos match the semantic meaning of the text prompt.
    """
    
    def __init__(self, 
                 model_name: str = "ViT-B/32",
                 **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self._load_model()
    
    def _load_model(self):
        """Load CLIP model"""
        
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.model.eval()
        print(f"CLIP model ({self.model_name}) loaded successfully for video-text scoring")
    
    def compute(self, 
                video: torch.Tensor,
                prompt: str,
                **kwargs) -> Dict[str, float]:
        """
        Compute CLIP scores between video frames and text prompt
        
        Args:
            video: Video tensor [T, C, H, W] in [0, 1] range
            prompt: Text prompt string
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with CLIP video-text score statistics
        """
        if self.model is None:
            print("CLIP model not available, returning placeholder values")
            return None
        
        num_frames = video.size(0)
        
        with torch.no_grad():
            # Encode text prompt once
            text_input = clip.tokenize([prompt]).to(self.device)
            text_features = self.model.encode_text(text_input)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Collect all frame embeddings
            frame_embeddings = []
            for i in range(num_frames):
                frame = video[i]  # [C, H, W]
                
                # Convert frame to PIL for CLIP preprocessing
                frame_pil = pt_to_pil(frame)
                
                # Preprocess for CLIP
                frame_input = self.preprocess(frame_pil).unsqueeze(0).to(self.device)
                
                # Get image features
                image_features = self.model.encode_image(frame_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                frame_embeddings.append(image_features)
            
            # Compute average embedding of all frames
            avg_video_embedding = torch.stack(frame_embeddings).mean(dim=0)
            avg_video_embedding = avg_video_embedding / avg_video_embedding.norm(dim=-1, keepdim=True)
            
            # Compute similarity between average video embedding and text
            similarity = torch.cosine_similarity(avg_video_embedding, text_features, dim=-1)
        
        return {
            "clip_video_text_score": float(similarity.item()),
            "total_frames": num_frames
        }
    
    def compute_multiple(self, 
                        videos: torch.Tensor,
                        prompts: List[str],
                        **kwargs) -> Dict[str, float]:
        """
        Compute CLIP video-text scores for multiple video-prompt pairs
        
        Args:
            videos: Videos tensor [B, T, C, H, W] in [0, 1] range
            prompts: List of text prompt strings, one for each video
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with aggregated CLIP video-text score statistics
        """
        batch_size = videos.size(0)
        
        if len(prompts) != batch_size:
            print(f"Warning: Number of prompts ({len(prompts)}) does not match batch size ({batch_size})")
            return None
        
        all_scores = []
        
        for i in range(batch_size):
            # Extract single video from batch [T, C, H, W]
            video = videos[i]
            prompt = prompts[i]
            
            # Compute CLIP video-text score for this pair
            video_result = self.compute(video, prompt, **kwargs)
            if video_result:
                all_scores.append(video_result["clip_video_text_score"])
        
        # Aggregate results across all videos
        all_scores = np.array(all_scores)
        return {
            "clip_video_text_score": float(np.mean(all_scores)),
        }
    
    def compute_upper_bound(self, 
                           videos: torch.Tensor, 
                           prompts: List[str],
                           sample_size: int = 10) -> float:
        """
        计算CLIP video-text分数的理论上限：视频与其对应的原始prompt对比
        
        这个上限表示"良好匹配"的参考水平，假设输入的视频-prompt对已经是高质量的配对。
        
        Args:
            videos: 视频张量 [B, T, C, H, W]
            prompts: 对应的文本prompt列表
            sample_size: 采样大小，避免计算过多
            
        Returns:
            上限分数的平均值
        """
        batch_size = videos.size(0)
        
        if len(prompts) != batch_size:
            print(f"Warning: Number of prompts ({len(prompts)}) does not match batch size ({batch_size})")
            return 1.0
        
        sample_size = min(sample_size, batch_size)
        
        # 随机采样一部分视频-prompt对
        indices = np.random.choice(batch_size, sample_size, replace=False)
        
        upper_scores = []
        with torch.no_grad():
            for idx in indices:
                video = videos[idx]
                prompt = prompts[idx]
                
                # 视频与其原始prompt对比（正常配对）
                result = self.compute(video, prompt)
                if result:
                    upper_scores.append(result["clip_video_text_score"])
        
        return float(np.mean(upper_scores)) if upper_scores else 1.0
    
    def compute_lower_bound(self, 
                           videos: torch.Tensor,
                           prompts: List[str], 
                           sample_size: int = 10) -> float:
        """
        计算CLIP video-text分数的理论下限：视频与随机不相关的prompt对比
        
        通过随机打乱prompt，让视频与不属于它的prompt配对，反映语义不相关的基线。
        
        Args:
            videos: 视频张量 [B, T, C, H, W]  
            prompts: 文本prompt列表
            sample_size: 采样大小
            
        Returns:
            下限分数的平均值
        """
        batch_size = videos.size(0)
        
        if len(prompts) != batch_size:
            print(f"Warning: Number of prompts ({len(prompts)}) does not match batch size ({batch_size})")
            return 0.0
        
        if batch_size < 2:
            return 0.0
        
        sample_size = min(sample_size, batch_size)
        
        lower_scores = []
        with torch.no_grad():
            for _ in range(sample_size):
                # 随机选择一个视频
                video_idx = np.random.randint(0, batch_size)
                
                # 随机选择一个不同的prompt（错误配对）
                prompt_idx = np.random.randint(0, batch_size)
                while prompt_idx == video_idx and batch_size > 1:
                    prompt_idx = np.random.randint(0, batch_size)
                
                video = videos[video_idx]
                mismatched_prompt = prompts[prompt_idx]
                
                # 计算不匹配的视频-prompt对的分数
                result = self.compute(video, mismatched_prompt)
                if result:
                    lower_scores.append(result["clip_video_text_score"])
        
        return float(np.mean(lower_scores)) if lower_scores else 0.0
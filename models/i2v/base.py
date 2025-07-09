from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
from PIL import Image
import cv2


class I2VModelBase(ABC):
    """图像到视频模型基类"""
    
    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            device: 计算设备
            model_path: 模型路径
            **kwargs: 其他参数
        """
        self.device = device
        self.model_path = model_path
        self.config = kwargs
        self.model = None
        self._setup_model()
    
    @abstractmethod
    def _setup_model(self):
        """设置模型"""
        pass
    
    @abstractmethod
    def generate_video(
        self,
        images: torch.Tensor,
        num_frames: int = 16,
        **kwargs
    ) -> torch.Tensor:
        """
        从图像生成视频
        
        Args:
            images: 输入图像张量 [B, C, H, W]，范围[0,1]
            num_frames: 生成视频的帧数
            **kwargs: 其他参数
            
        Returns:
            生成的视频张量 [B, T, C, H, W]，范围[0,1]
        """
        pass
    
    def generate_video_from_pil(
        self,
        images: List[Image.Image],
        num_frames: int = 16,
        **kwargs
    ) -> List[List[Image.Image]]:
        """
        从PIL图像生成视频
        
        Args:
            images: PIL图像列表
            num_frames: 生成视频的帧数
            **kwargs: 其他参数
            
        Returns:
            生成的视频列表，每个元素是帧的PIL图像列表
        """
        # 转换为张量
        image_tensors = []
        for img in images:
            img_array = np.array(img)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
            image_tensors.append(img_tensor)
        
        batch_tensor = torch.stack(image_tensors).to(self.device)
        
        # 生成视频
        video_tensor = self.generate_video(batch_tensor, num_frames, **kwargs)
        
        # 转换回PIL图像
        videos = []
        for b in range(video_tensor.size(0)):
            frames = []
            for t in range(video_tensor.size(1)):
                frame_tensor = video_tensor[b, t].cpu()
                frame_array = (frame_tensor.permute(1, 2, 0) * 255).clamp(0, 255).numpy().astype(np.uint8)
                frame_pil = Image.fromarray(frame_array)
                frames.append(frame_pil)
            videos.append(frames)
        
        return videos
    
    def save_video(
        self,
        video_tensor: torch.Tensor,
        output_path: str,
        fps: int = 8
    ):
        """
        保存视频张量为视频文件
        
        Args:
            video_tensor: 视频张量 [T, C, H, W] 或 [B, T, C, H, W]
            output_path: 输出路径
            fps: 帧率
        """
        if video_tensor.dim() == 5:
            # 批量视频，只保存第一个
            video_tensor = video_tensor[0]
        
        # 转换为numpy数组
        video_array = video_tensor.cpu().numpy()
        video_array = (video_array * 255).astype(np.uint8)
        
        # 调整维度顺序为 [T, H, W, C]
        video_array = np.transpose(video_array, (0, 2, 3, 1))
        
        # 创建视频写入器
        height, width = video_array.shape[1:3]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 写入帧
        for frame in video_array:
            # OpenCV使用BGR格式
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
    
    def compute_video_metrics(
        self,
        generated_video: torch.Tensor,
        reference_video: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        计算视频质量指标
        
        Args:
            generated_video: 生成的视频张量
            reference_video: 参考视频张量（可选）
            
        Returns:
            指标字典
        """
        metrics = {}
        
        # 基础质量指标
        metrics['temporal_consistency'] = self._compute_temporal_consistency(generated_video)
        metrics['motion_smoothness'] = self._compute_motion_smoothness(generated_video)
        metrics['frame_quality'] = self._compute_frame_quality(generated_video)
        
        # 如果有参考视频，计算对比指标
        if reference_video is not None:
            metrics['video_mse'] = self._compute_video_mse(generated_video, reference_video)
            metrics['video_ssim'] = self._compute_video_ssim(generated_video, reference_video)
        
        return metrics
    
    def _compute_temporal_consistency(self, video: torch.Tensor) -> float:
        """计算时序一致性"""
        if video.dim() == 5:
            video = video[0]  # 取第一个样本
        
        num_frames = video.size(0)
        if num_frames < 2:
            return 1.0
        
        # 计算相邻帧的差异
        frame_diffs = []
        for t in range(num_frames - 1):
            diff = torch.mean(torch.abs(video[t+1] - video[t]))
            frame_diffs.append(diff.item())
        
        # 一致性分数（差异越小越好）
        avg_diff = np.mean(frame_diffs)
        consistency = np.exp(-avg_diff * 10)  # 缩放因子
        
        return consistency
    
    def _compute_motion_smoothness(self, video: torch.Tensor) -> float:
        """计算运动平滑度"""
        if video.dim() == 5:
            video = video[0]  # 取第一个样本
        
        num_frames = video.size(0)
        if num_frames < 3:
            return 1.0
        
        # 计算运动向量的变化
        motion_changes = []
        for t in range(num_frames - 2):
            # 简化的运动估计
            motion1 = video[t+1] - video[t]
            motion2 = video[t+2] - video[t+1]
            motion_change = torch.mean(torch.abs(motion2 - motion1))
            motion_changes.append(motion_change.item())
        
        # 平滑度分数
        avg_change = np.mean(motion_changes)
        smoothness = np.exp(-avg_change * 5)
        
        return smoothness
    
    def _compute_frame_quality(self, video: torch.Tensor) -> float:
        """计算帧质量"""
        if video.dim() == 5:
            video = video[0]  # 取第一个样本
        
        # 计算每帧的清晰度（基于梯度强度）
        frame_qualities = []
        for t in range(video.size(0)):
            frame = video[t]
            # 计算梯度
            grad_x = torch.abs(frame[:, 1:, :] - frame[:, :-1, :])
            grad_y = torch.abs(frame[:, :, 1:] - frame[:, :, :-1])
            
            # 平均梯度强度作为清晰度指标
            clarity = (torch.mean(grad_x) + torch.mean(grad_y)) / 2
            frame_qualities.append(clarity.item())
        
        return np.mean(frame_qualities)
    
    def _compute_video_mse(
        self,
        video1: torch.Tensor,
        video2: torch.Tensor
    ) -> float:
        """计算视频MSE"""
        mse = torch.mean((video1 - video2) ** 2)
        return mse.item()
    
    def _compute_video_ssim(
        self,
        video1: torch.Tensor,
        video2: torch.Tensor
    ) -> float:
        """计算视频SSIM"""
        # 简化实现，实际中应该使用更完整的SSIM
        if video1.dim() == 5:
            video1 = video1[0]
            video2 = video2[0]
        
        ssim_scores = []
        for t in range(min(video1.size(0), video2.size(0))):
            frame1 = video1[t:t+1]
            frame2 = video2[t:t+1]
            
            # 简化的SSIM计算
            mu1 = torch.mean(frame1)
            mu2 = torch.mean(frame2)
            sigma1 = torch.std(frame1)
            sigma2 = torch.std(frame2)
            sigma12 = torch.mean((frame1 - mu1) * (frame2 - mu2))
            
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2))
            ssim_scores.append(ssim.item())
        
        return np.mean(ssim_scores)
    
    def to(self, device: str):
        """移动模型到指定设备"""
        self.device = device
        if self.model is not None:
            self.model.to(device)
        return self
    
    def eval(self):
        """设置为评估模式"""
        if self.model is not None:
            self.model.eval()
        return self
    
    def train(self):
        """设置为训练模式"""
        if self.model is not None:
            self.model.train()
        return self


class DiffusionI2VBase(I2VModelBase):
    """基于扩散模型的I2V基类"""
    
    def __init__(
        self,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        **kwargs
    ):
        """
        Args:
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
        """
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        super().__init__(**kwargs)
    
    def _setup_scheduler(self):
        """设置调度器"""
        # 子类实现具体的调度器设置
        pass
    
    def _encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """编码图像到潜在空间"""
        # 子类实现具体的编码逻辑
        return images
    
    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """解码潜在表示到图像"""
        # 子类实现具体的解码逻辑
        return latents


class AttentionI2VBase(I2VModelBase):
    """基于注意力机制的I2V基类"""
    
    def __init__(
        self,
        attention_layers: int = 8,
        attention_heads: int = 8,
        **kwargs
    ):
        """
        Args:
            attention_layers: 注意力层数
            attention_heads: 注意力头数
        """
        self.attention_layers = attention_layers
        self.attention_heads = attention_heads
        super().__init__(**kwargs)
    
    def _apply_temporal_attention(
        self,
        features: torch.Tensor,
        num_frames: int
    ) -> torch.Tensor:
        """应用时序注意力"""
        # 子类实现具体的时序注意力逻辑
        return features


class GANBasedI2VBase(I2VModelBase):
    """基于GAN的I2V基类"""
    
    def __init__(
        self,
        generator_layers: int = 6,
        discriminator_layers: int = 4,
        **kwargs
    ):
        """
        Args:
            generator_layers: 生成器层数
            discriminator_layers: 判别器层数
        """
        self.generator_layers = generator_layers
        self.discriminator_layers = discriminator_layers
        super().__init__(**kwargs) 
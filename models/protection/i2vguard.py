import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from .base import ProtectionBase


class I2VGuard(ProtectionBase):
    """ 
    I2VGuard: 专门针对图像到视频模型的保护算法
    结合时序一致性和运动模式干扰
    """
    
    def __init__(
        self,
        i2v_model: Optional[nn.Module] = None,
        motion_predictor: Optional[nn.Module] = None,
        epsilon: float = 0.03,
        num_steps: int = 50,
        temporal_weight: float = 0.5,
        motion_weight: float = 0.3,
        consistency_weight: float = 0.2,
        step_size: float = 0.001,
        **kwargs
    ):
        """
        Args:
            i2v_model: 图像到视频模型（可选）
            motion_predictor: 运动预测器
            epsilon: 扰动预算
            num_steps: 优化步数
            temporal_weight: 时序权重
            motion_weight: 运动权重
            consistency_weight: 一致性权重
            step_size: 优化步长
        """
        self.i2v_model = i2v_model
        self.motion_predictor = motion_predictor
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.temporal_weight = temporal_weight
        self.motion_weight = motion_weight
        self.consistency_weight = consistency_weight
        super().__init__(**kwargs)
    
    def _setup_model(self):
        """设置模型"""
        # 创建时序分析网络
        self.temporal_analyzer = TemporalAnalyzer().to(self.device)
        
        # 创建运动干扰生成器
        self.motion_disruptor = MotionDisruptor().to(self.device)
        
        # 如果没有提供运动预测器，创建一个简单的
        if self.motion_predictor is None:
            self.motion_predictor = SimpleMotionPredictor().to(self.device)
    
    def protect(
        self,
        image: torch.Tensor,
        num_frames: int = 16,
        **kwargs
    ) -> torch.Tensor:
        """
        使用I2VGuard保护单张图像
        
        Args:
            image: 输入图像张量 [C, H, W]
            num_frames: 生成视频的帧数
            **kwargs: 其他参数
            
        Returns:
            受保护的图像张量 [C, H, W]
        """
        image = image.to(self.device)
        
        # 添加批次维度
        images = image.unsqueeze(0)  # [1, C, H, W]
        
        # 应用I2V特定保护
        protected_images = self._apply_i2v_protection(images, num_frames)
        
        # 移除批次维度
        return protected_images.squeeze(0)
    
    def _apply_i2v_protection(
        self,
        images: torch.Tensor,
        num_frames: int
    ) -> torch.Tensor:
        """
        应用I2V保护
        
        Args:
            images: 输入图像
            num_frames: 视频帧数
            
        Returns:
            受保护的图像
        """
        batch_size = images.size(0)
        
        # 初始化扰动
        delta = torch.zeros_like(images).uniform_(-self.epsilon, self.epsilon)
        delta = torch.clamp(delta, 0 - images, 1 - images)
        delta.requires_grad_(True)
        
        optimizer = torch.optim.Adam([delta], lr=self.step_size)
        
        for step in range(self.num_steps):
            optimizer.zero_grad()
            
            # 应用扰动
            perturbed_images = images + delta
            perturbed_images = torch.clamp(perturbed_images, 0, 1)
            
            # 计算多个损失项
            temporal_loss = self._compute_temporal_loss(perturbed_images, num_frames)
            motion_loss = self._compute_motion_loss(perturbed_images)
            consistency_loss = self._compute_consistency_loss(perturbed_images, images)
            
            # 组合损失
            total_loss = (
                self.temporal_weight * temporal_loss +
                self.motion_weight * motion_loss +
                self.consistency_weight * consistency_loss
            )
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            # 投影到约束集合
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
                delta.data = torch.clamp(delta.data, 0 - images, 1 - images)
            
            if step % 10 == 0:
                print(f"Step {step}, Loss: {total_loss.item():.4f}")
        
        return torch.clamp(images + delta.detach(), 0, 1)
    
    def _compute_temporal_loss(
        self,
        images: torch.Tensor,
        num_frames: int
    ) -> torch.Tensor:
        """
        计算时序损失
        
        Args:
            images: 图像张量
            num_frames: 帧数
            
        Returns:
            时序损失
        """
        try:
            # 尝试生成视频
            with torch.no_grad():
                # 生成原始视频
                original_video = self._generate_video_sequence(images, num_frames)
            
            # 生成扰动视频（需要梯度）
            perturbed_video = self._generate_video_sequence(images, num_frames)
            
            # 计算时序一致性损失（最大化不一致性）
            temporal_consistency = self._compute_temporal_consistency(perturbed_video)
            loss = -temporal_consistency  # 最大化不一致性
            
        except Exception as e:
            print(f"Warning: Temporal loss computation failed: {e}")
            # 使用简化的时序损失
            loss = -torch.mean(torch.norm(images, p=2, dim=[1, 2, 3]))
        
        return loss
    
    def _compute_motion_loss(self, images: torch.Tensor) -> torch.Tensor:
        """
        计算运动损失
        
        Args:
            images: 图像张量
            
        Returns:
            运动损失
        """
        # 预测运动模式
        predicted_motion = self.motion_predictor(images)
        
        # 生成运动干扰
        motion_disruption = self.motion_disruptor(images, predicted_motion)
        
        # 计算运动损失（最大化运动预测误差）
        motion_loss = torch.mean(torch.norm(motion_disruption, p=2, dim=[1, 2, 3]))
        
        return motion_loss
    
    def _compute_consistency_loss(
        self,
        perturbed_images: torch.Tensor,
        original_images: torch.Tensor
    ) -> torch.Tensor:
        """
        计算一致性损失
        
        Args:
            perturbed_images: 扰动图像
            original_images: 原始图像
            
        Returns:
            一致性损失
        """
        # 计算感知损失
        perceptual_loss = self._compute_perceptual_loss(perturbed_images, original_images)
        
        # 计算结构损失
        structural_loss = self._compute_structural_loss(perturbed_images, original_images)
        
        # 组合一致性损失
        consistency_loss = perceptual_loss + 0.5 * structural_loss
        
        return consistency_loss
    
    def _generate_video_sequence(
        self,
        images: torch.Tensor,
        num_frames: int
    ) -> torch.Tensor:
        """
        生成视频序列
        
        Args:
            images: 输入图像
            num_frames: 帧数
            
        Returns:
            视频序列张量 [B, T, C, H, W]
        """
        # 这里应该调用实际的I2V模型
        # 为演示目的，创建一个简单的序列
        batch_size, channels, height, width = images.shape
        
        # 模拟视频生成过程
        video_sequence = torch.zeros(batch_size, num_frames, channels, height, width, 
                                   device=images.device)
        
        for t in range(num_frames):
            # 简单的线性插值模拟运动
            alpha = t / max(num_frames - 1, 1)
            noise = torch.randn_like(images) * 0.1 * alpha
            video_sequence[:, t] = torch.clamp(images + noise, 0, 1)
        
        return video_sequence
    
    def _compute_temporal_consistency(self, video: torch.Tensor) -> torch.Tensor:
        """
        计算时序一致性
        
        Args:
            video: 视频张量 [B, T, C, H, W]
            
        Returns:
            时序一致性分数
        """
        batch_size, num_frames, channels, height, width = video.shape
        
        # 计算相邻帧之间的差异
        frame_diffs = []
        for t in range(num_frames - 1):
            diff = torch.mean(torch.abs(video[:, t+1] - video[:, t]), dim=[1, 2, 3])
            frame_diffs.append(diff)
        
        # 计算平均帧差异
        avg_frame_diff = torch.stack(frame_diffs, dim=1).mean(dim=1)
        
        # 一致性分数（差异越小，一致性越高）
        consistency = torch.exp(-avg_frame_diff)
        
        return consistency.mean()
    
    def _compute_perceptual_loss(
        self,
        images1: torch.Tensor,
        images2: torch.Tensor
    ) -> torch.Tensor:
        """
        计算感知损失
        
        Args:
            images1: 第一组图像
            images2: 第二组图像
            
        Returns:
            感知损失
        """
        # 使用特征提取器计算感知损失
        with torch.no_grad():
            features1 = self.feature_extractor(images1) if hasattr(self, 'feature_extractor') else images1
            features2 = self.feature_extractor(images2) if hasattr(self, 'feature_extractor') else images2
        
        # 计算特征差异
        perceptual_loss = F.mse_loss(features1, features2)
        
        return perceptual_loss
    
    def _compute_structural_loss(
        self,
        images1: torch.Tensor,
        images2: torch.Tensor
    ) -> torch.Tensor:
        """
        计算结构损失
        
        Args:
            images1: 第一组图像
            images2: 第二组图像
            
        Returns:
            结构损失
        """
        # 计算SSIM损失
        ssim_loss = 1 - self._compute_ssim(images1, images2)
        
        return ssim_loss
    
    def _compute_ssim(
        self,
        images1: torch.Tensor,
        images2: torch.Tensor,
        window_size: int = 11
    ) -> torch.Tensor:
        """
        计算SSIM
        
        Args:
            images1: 第一组图像
            images2: 第二组图像
            window_size: 窗口大小
            
        Returns:
            SSIM值
        """
        # 简化的SSIM计算
        mu1 = F.avg_pool2d(images1, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(images2, window_size, stride=1, padding=window_size//2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(images1 ** 2, window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(images2 ** 2, window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(images1 * images2, window_size, stride=1, padding=window_size//2) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()


class TemporalAnalyzer(nn.Module):
    """时序分析网络"""
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, stride=2, padding=1)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_dim * 4, hidden_dim)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        分析图像的时序特征
        
        Args:
            images: 输入图像 [B, C, H, W]
            
        Returns:
            时序特征 [B, hidden_dim]
        """
        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = self.global_pool(x).flatten(1)
        x = self.fc(x)
        
        return x


class MotionDisruptor(nn.Module):
    """运动干扰生成器"""
    
    def __init__(self, input_dim: int = 3, motion_dim: int = 64):
        super().__init__()
        
        self.image_encoder = nn.Sequential(
            nn.Conv2d(input_dim, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.motion_fusion = nn.Conv2d(64 + motion_dim, 64, 1)
        
        self.disruptor = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, input_dim, 3, padding=1),
            nn.Tanh(),
        )
    
    def forward(
        self,
        images: torch.Tensor,
        motion_features: torch.Tensor
    ) -> torch.Tensor:
        """
        生成运动干扰
        
        Args:
            images: 输入图像 [B, C, H, W]
            motion_features: 运动特征 [B, motion_dim]
            
        Returns:
            运动干扰 [B, C, H, W]
        """
        # 编码图像
        image_features = self.image_encoder(images)
        
        # 扩展运动特征
        motion_map = motion_features.view(-1, motion_features.size(1), 1, 1)
        motion_map = motion_map.expand(-1, -1, image_features.size(2), image_features.size(3))
        
        # 融合特征
        fused_features = torch.cat([image_features, motion_map], dim=1)
        fused_features = self.motion_fusion(fused_features)
        
        # 生成干扰
        disruption = self.disruptor(fused_features)
        
        return disruption * 0.02  # 限制干扰强度


class SimpleMotionPredictor(nn.Module):
    """简单运动预测器"""
    
    def __init__(self, input_dim: int = 3, output_dim: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.predictor = nn.Linear(128, output_dim)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        预测运动特征
        
        Args:
            images: 输入图像 [B, C, H, W]
            
        Returns:
            运动特征 [B, output_dim]
        """
        features = self.encoder(images).flatten(1)
        motion_features = self.predictor(features)
        
        return motion_features 
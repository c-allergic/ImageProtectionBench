from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image


class ProtectionBase(ABC):
    """图像保护算法基类"""
    
    def __init__(
        self,
        device: str = "cuda",
        **kwargs
    ):
        """
        Args:
            device: 计算设备
            **kwargs: 其他参数
        """
        self.device = device
        self.config = kwargs
        self.model = None
        self._setup_model()
    
    @abstractmethod
    def _setup_model(self):
        """设置模型"""
        pass
    
    @abstractmethod
    def protect(
        self, 
        images: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        对图像应用保护
        
        Args:
            images: 输入图像张量 [B, C, H, W]，范围[0,1]
            **kwargs: 其他参数
            
        Returns:
            受保护的图像张量，范围[0,1]
        """
        pass
    
    def protect_batch(
        self,
        images: torch.Tensor,
        batch_size: int = 4,
        **kwargs
    ) -> torch.Tensor:
        """
        批量保护图像
        
        Args:
            images: 输入图像张量 [B, C, H, W]
            batch_size: 批大小
            **kwargs: 其他参数
            
        Returns:
            受保护的图像张量
        """
        protected_images = []
        num_batches = (images.size(0) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, images.size(0))
            batch = images[start_idx:end_idx]
            
            protected_batch = self.protect(batch, **kwargs)
            protected_images.append(protected_batch)
        
        return torch.cat(protected_images, dim=0)
    
    def protect_image(
        self,
        image: Image.Image,
        **kwargs
    ) -> Image.Image:
        """
        保护单张PIL图像
        
        Args:
            image: PIL图像
            **kwargs: 其他参数
            
        Returns:
            受保护的PIL图像
        """
        # 转换为张量
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # 应用保护
        protected_tensor = self.protect(image_tensor, **kwargs)
        
        # 转换回PIL图像
        protected_tensor = protected_tensor.squeeze(0).cpu()
        protected_array = (protected_tensor.permute(1, 2, 0) * 255).clamp(0, 255).numpy().astype(np.uint8)
        
        return Image.fromarray(protected_array)
    
    def compute_protection_strength(
        self,
        original: torch.Tensor,
        protected: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算保护强度指标
        
        Args:
            original: 原始图像
            protected: 受保护图像
            
        Returns:
            保护强度指标字典
        """
        # 计算PSNR
        mse = torch.mean((original - protected) ** 2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
        
        # 计算扰动强度
        perturbation = torch.mean(torch.abs(original - protected))
        
        # 计算L2距离
        l2_distance = torch.mean(torch.norm(original - protected, p=2, dim=[1, 2, 3]))
        
        return {
            'psnr': psnr.item(),
            'perturbation_strength': perturbation.item(),
            'l2_distance': l2_distance.item()
        }
    
    def save_model(self, path: str):
        """保存模型"""
        if self.model is not None:
            torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        """加载模型"""
        if self.model is not None:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
    
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


class AdversarialProtection(ProtectionBase):
    """对抗样本保护基类"""
    
    def __init__(
        self,
        target_model: nn.Module,
        epsilon: float = 0.03,
        num_steps: int = 10,
        step_size: float = 0.01,
        **kwargs
    ):
        """
        Args:
            target_model: 目标模型（用于生成对抗样本）
            epsilon: 扰动预算
            num_steps: 攻击步数
            step_size: 攻击步长
        """
        self.target_model = target_model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        super().__init__(**kwargs)
    
    def _pgd_attack(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        targeted: bool = False
    ) -> torch.Tensor:
        """
        PGD攻击生成对抗样本
        
        Args:
            images: 输入图像
            labels: 目标标签
            targeted: 是否为目标攻击
            
        Returns:
            对抗样本
        """
        images = images.clone().detach().to(self.device)
        
        # 随机初始化扰动
        delta = torch.zeros_like(images).uniform_(-self.epsilon, self.epsilon)
        delta = torch.clamp(delta, 0 - images, 1 - images)
        delta.requires_grad_(True)
        
        for _ in range(self.num_steps):
            # 前向传播
            adv_images = images + delta
            outputs = self.target_model(adv_images)
            
            # 计算损失
            if labels is not None:
                if isinstance(outputs, dict):
                    # 处理复杂输出（如diffusion模型的latent）
                    loss = torch.mean(torch.norm(outputs['latent'], p=2, dim=1))
                else:
                    loss = nn.functional.cross_entropy(outputs, labels)
            else:
                # 无监督情况下，最大化输出的变化
                loss = torch.mean(torch.norm(outputs, p=2))
            
            if not targeted:
                loss = -loss
            
            # 反向传播
            loss.backward()
            
            # 更新扰动
            grad = delta.grad.detach()
            delta = delta + self.step_size * grad.sign()
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            delta = torch.clamp(delta, 0 - images, 1 - images)
            delta.requires_grad_(True)
        
        return images + delta.detach()


class FeatureDistortionProtection(ProtectionBase):
    """特征扭曲保护基类"""
    
    def __init__(
        self,
        feature_extractor: nn.Module,
        distortion_strength: float = 0.1,
        **kwargs
    ):
        """
        Args:
            feature_extractor: 特征提取器
            distortion_strength: 扭曲强度
        """
        self.feature_extractor = feature_extractor
        self.distortion_strength = distortion_strength
        super().__init__(**kwargs)
    
    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """提取图像特征"""
        with torch.no_grad():
            features = self.feature_extractor(images)
        return features
    
    def _apply_feature_distortion(
        self,
        images: torch.Tensor,
        features: torch.Tensor
    ) -> torch.Tensor:
        """应用特征扭曲"""
        # 这里应该实现具体的特征扭曲逻辑
        # 基类中提供一个简单的实现
        noise = torch.randn_like(images) * self.distortion_strength
        return torch.clamp(images + noise, 0, 1) 
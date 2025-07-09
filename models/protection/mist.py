import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
import numpy as np
from .base import FeatureDistortionProtection


class Mist(FeatureDistortionProtection):
    """
    Mist: 轻量级图像保护方法
    基于论文: "Mist: Towards Improved Adversarial Examples for Diffusion Models"
    """
    
    def __init__(
        self,
        feature_extractor: nn.Module,
        epsilon: float = 0.02,
        num_steps: int = 50,
        mist_strength: float = 0.1,
        blur_kernel_size: int = 3,
        **kwargs
    ):
        """
        Args:
            feature_extractor: 特征提取器
            epsilon: 扰动预算
            num_steps: 迭代步数
            mist_strength: Mist强度
            blur_kernel_size: 模糊核大小
        """
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.mist_strength = mist_strength
        self.blur_kernel_size = blur_kernel_size
        super().__init__(
            feature_extractor=feature_extractor,
            distortion_strength=epsilon,
            **kwargs
        )
    
    def _setup_model(self):
        """设置模型"""
        # 创建低通滤波器
        self.low_pass_filter = self._create_low_pass_filter()
        
        # 创建Mist生成网络
        self.mist_generator = MistGenerator().to(self.device)
    
    def protect(
        self,
        images: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        使用Mist保护图像
        
        Args:
            images: 输入图像张量 [B, C, H, W]
            **kwargs: 其他参数
            
        Returns:
            受保护的图像张量
        """
        images = images.to(self.device)
        
        # 应用Mist保护
        protected_images = self._apply_mist_protection(images)
        
        return protected_images
    
    def _apply_mist_protection(self, images: torch.Tensor) -> torch.Tensor:
        """
        应用Mist保护
        
        Args:
            images: 输入图像
            
        Returns:
            受保护的图像
        """
        batch_size = images.size(0)
        
        # 1. 生成低频扰动
        low_freq_perturbation = self._generate_low_frequency_perturbation(images)
        
        # 2. 生成高频细节保护
        high_freq_protection = self._generate_high_frequency_protection(images)
        
        # 3. 结合两种保护
        combined_perturbation = (
            0.7 * low_freq_perturbation + 
            0.3 * high_freq_protection
        )
        
        # 4. 应用扰动强度控制
        controlled_perturbation = self._apply_strength_control(
            combined_perturbation, 
            images
        )
        
        # 5. 生成最终保护图像
        protected_images = images + controlled_perturbation
        protected_images = torch.clamp(protected_images, 0, 1)
        
        return protected_images
    
    def _generate_low_frequency_perturbation(
        self, 
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        生成低频扰动
        
        Args:
            images: 输入图像
            
        Returns:
            低频扰动
        """
        # 使用Mist生成器创建结构化扰动
        mist_pattern = self.mist_generator(images)
        
        # 应用低通滤波
        filtered_mist = self._apply_low_pass_filter(mist_pattern)
        
        # 缩放到合适的强度
        scaled_mist = filtered_mist * self.mist_strength
        
        return scaled_mist
    
    def _generate_high_frequency_protection(
        self, 
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        生成高频保护
        
        Args:
            images: 输入图像
            
        Returns:
            高频保护扰动
        """
        # 计算图像边缘
        edges = self._compute_edges(images)
        
        # 在边缘区域添加细微扰动
        edge_perturbation = torch.randn_like(images) * 0.01
        edge_mask = (edges > edges.mean()).float()
        
        # 只在边缘区域应用扰动
        masked_perturbation = edge_perturbation * edge_mask.unsqueeze(1)
        
        return masked_perturbation
    
    def _compute_edges(self, images: torch.Tensor) -> torch.Tensor:
        """
        计算图像边缘
        
        Args:
            images: 输入图像
            
        Returns:
            边缘强度图
        """
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=images.dtype, device=images.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=images.dtype, device=images.device)
        
        # 扩展为3D卷积核
        sobel_x = sobel_x.view(1, 1, 3, 3).expand(3, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3).expand(3, 1, 3, 3)
        
        # 计算梯度
        grad_x = F.conv2d(images, sobel_x, padding=1, groups=3)
        grad_y = F.conv2d(images, sobel_y, padding=1, groups=3)
        
        # 计算梯度幅值
        edges = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        edges = torch.mean(edges, dim=1)  # 平均所有通道
        
        return edges
    
    def _apply_strength_control(
        self,
        perturbation: torch.Tensor,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        应用扰动强度控制
        
        Args:
            perturbation: 原始扰动
            images: 输入图像
            
        Returns:
            控制强度后的扰动
        """
        # 计算当前扰动强度
        current_strength = torch.norm(perturbation, p=2, dim=[1, 2, 3])
        
        # 计算目标强度
        target_strength = self.epsilon * torch.ones_like(current_strength)
        
        # 如果扰动过强，进行缩放
        scale_factor = torch.min(
            target_strength / (current_strength + 1e-8),
            torch.ones_like(current_strength)
        )
        
        # 应用缩放
        controlled_perturbation = perturbation * scale_factor.view(-1, 1, 1, 1)
        
        return controlled_perturbation
    
    def _create_low_pass_filter(self) -> torch.Tensor:
        """创建低通滤波器"""
        kernel_size = self.blur_kernel_size
        sigma = kernel_size / 3.0
        
        # 创建高斯核
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        # 扩展为3D卷积核
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.expand(3, 1, kernel_size, kernel_size)
        
        return kernel.to(self.device)
    
    def _apply_low_pass_filter(self, images: torch.Tensor) -> torch.Tensor:
        """应用低通滤波器"""
        padding = self.blur_kernel_size // 2
        filtered = F.conv2d(images, self.low_pass_filter, 
                           padding=padding, groups=3)
        return filtered


class MistGenerator(nn.Module):
    """Mist生成网络"""
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 32):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 3, 
                             stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 3, 
                             stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, input_dim, 3, padding=1),
            nn.Tanh(),  # 输出范围[-1, 1]
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        生成Mist模式
        
        Args:
            images: 输入图像 [B, C, H, W]
            
        Returns:
            Mist模式 [B, C, H, W]
        """
        # 编码
        encoded = self.encoder(images)
        
        # 解码生成Mist
        mist = self.decoder(encoded)
        
        # 缩放到合适的范围
        mist = mist * 0.05  # 限制Mist强度
        
        return mist


class AdaptiveMist(Mist):
    """自适应Mist保护"""
    
    def __init__(
        self,
        feature_extractor: nn.Module,
        content_analyzer: Optional[nn.Module] = None,
        **kwargs
    ):
        """
        Args:
            feature_extractor: 特征提取器
            content_analyzer: 内容分析器
        """
        self.content_analyzer = content_analyzer
        super().__init__(feature_extractor=feature_extractor, **kwargs)
    
    def _apply_mist_protection(self, images: torch.Tensor) -> torch.Tensor:
        """
        应用自适应Mist保护
        
        Args:
            images: 输入图像
            
        Returns:
            受保护的图像
        """
        # 分析图像内容
        content_scores = self._analyze_content(images)
        
        # 根据内容调整保护强度
        adaptive_strength = self._compute_adaptive_strength(content_scores)
        
        # 生成基础Mist保护
        base_protection = super()._apply_mist_protection(images)
        
        # 应用自适应强度
        perturbation = base_protection - images
        adaptive_perturbation = perturbation * adaptive_strength.view(-1, 1, 1, 1)
        
        return images + adaptive_perturbation
    
    def _analyze_content(self, images: torch.Tensor) -> torch.Tensor:
        """
        分析图像内容
        
        Args:
            images: 输入图像
            
        Returns:
            内容分数
        """
        if self.content_analyzer is not None:
            with torch.no_grad():
                content_scores = self.content_analyzer(images)
        else:
            # 简单的内容分析：基于图像复杂度
            # 计算图像的标准差作为复杂度指标
            complexity = torch.std(images, dim=[1, 2, 3])
            content_scores = complexity / complexity.max()
        
        return content_scores
    
    def _compute_adaptive_strength(self, content_scores: torch.Tensor) -> torch.Tensor:
        """
        计算自适应强度
        
        Args:
            content_scores: 内容分数
            
        Returns:
            自适应强度因子
        """
        # 复杂度高的图像需要更强的保护
        min_strength = 0.5
        max_strength = 1.5
        
        adaptive_strength = min_strength + (max_strength - min_strength) * content_scores
        
        return adaptive_strength 
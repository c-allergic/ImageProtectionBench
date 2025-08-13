import torch
import torch.nn.functional as F
from .base import ProtectionBase, timeit
from typing import Union, List


class RandomNoise(ProtectionBase):
    """
    RandomNoise: I2VGuard论文中的随机噪声baseline方法
    
    参考论文: I2VGuard (CVPR 2025)
    
    实现细节:
    - 噪声类型: 均匀随机噪声 (Uniform Random Noise)
    - 扰动上限: L∞范数 ε = 4/255 (8-bit图像上±4灰度级)
    - 应用范围: 整张图像，无区域限制
    - 每个像素独立采样
    """
    
    def __init__(self, 
                 eps: float = 4/255,
                 seed: int = None,
                 **kwargs):
        """
        初始化RandomNoise保护算法
        
        Args:
            eps: L∞范数扰动上限，默认4/255
            seed: 随机种子，用于可重复实验，None表示不固定种子
            **kwargs: 其他参数
        """
        self.eps = eps
        self.seed = seed
        
        # 设置随机种子（如果指定）
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
        
        super().__init__(**kwargs)
        
    def _setup_model(self):
        """
        设置模型（RandomNoise不需要预训练模型）
        """
        pass
    
    def protect(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        为单张图像添加均匀随机噪声保护
        
        Args:
            image: 输入图像张量 [C, H, W]，范围[0,1]
            **kwargs: 其他参数（当前未使用）
            
        Returns:
            受保护的图像张量 [C, H, W]，范围[0,1]
        """
        # 确保图像在正确设备上
        image = image.to(self.device)
        
        # 生成均匀随机噪声
        # 从 [-eps, eps] 均匀分布中采样
        noise = torch.empty_like(image).uniform_(-self.eps, self.eps)
        
        # 添加噪声
        noisy_image = image + noise
        
        # 确保输出在有效范围[0,1]内
        protected_image = torch.clamp(noisy_image, 0.0, 1.0)
        
        return protected_image
    
    @timeit
    def protect_multiple(
        self, 
        images: Union[torch.Tensor, List[torch.Tensor]], 
        **kwargs
    ) -> torch.Tensor:
        """
        批量保护多张图像
        
        Args:
            images: 图像张量 [B, C, H, W] 或图像张量列表
            **kwargs: 传递给protect方法的其他参数
            
        Returns:
            受保护的图像张量 [B, C, H, W]
        """
        # 处理输入格式
        if isinstance(images, list):
            images = torch.stack(images)
        
        if len(images.shape) == 3:
            # 单张图片，添加批次维度
            images = images.unsqueeze(0)
        
        images = images.to(self.device)
        
        # 批量生成噪声（更高效）
        noise = torch.empty_like(images).uniform_(-self.eps, self.eps)
        
        # 批量添加噪声
        noisy_images = images + noise
        
        # 确保输出在有效范围[0,1]内
        protected_images = torch.clamp(noisy_images, 0.0, 1.0)
        
        return protected_images
    
    def __repr__(self):
        return (f"RandomNoise(eps={self.eps:.6f}, "
                f"eps_8bit={self.eps*255:.1f}, "
                f"seed={self.seed}, "
                f"device='{self.device}')")

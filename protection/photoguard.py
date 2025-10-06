import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import numpy as np
from tqdm import tqdm
from .base import ProtectionBase


class PhotoGuard(ProtectionBase):
    """
    PhotoGuard: 针对文本到图像扩散模型的保护方法
    基于论文: "PhotoGuard: Disrupting Diffusion-based Generative Models for Copyright Protection"
    使用原始论文中的simple encoder attack实现
    """
    
    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        epsilon: float = 16/255,
        step_size: float = 2/255,
        num_steps: int = 200,
        clamp_min: float = -1,
        clamp_max: float = 1,
        **kwargs
    ):
        """
        PhotoGuard 保护方法初始化
        
        Args:
            model_name: Stable Diffusion模型名称或路径
            epsilon: L∞ 扰动预算，论文值 16/255 ≈ 0.0627
            step_size: PGD 攻击步长，论文值 2/255 ≈ 0.0078
            num_steps: 优化步数（默认 200）
            clamp_min: 像素值下限（VAE 需要 -1）
            clamp_max: 像素值上限（VAE 需要 1）
        
        Note:
            - 所有内部操作在 [-1, 1] 范围内进行（VAE 要求）
            - 接口输入输出为 [0, 1] 范围
        """
        self.model_name = model_name
        self.diffusion_model = None
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        super().__init__(**kwargs)
    
    def _setup_model(self):
        """设置模型"""
        # 模型将在第一次使用时自动加载
        pass
    
    def protect(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        使用PhotoGuard保护单张图像
        
        Args:
            image: 单张图像张量 [C, H, W]，范围[0,1]
            **kwargs: 其他参数
            
        Returns:
            受保护的图像张量 [C, H, W]，范围[0,1]
        """
        # 将图像移到正确的设备
        image = image.to(self.device)
        
        # 转换到 [-1, 1] 范围（Stable Diffusion VAE 要求）
        image_normalized = image * 2.0 - 1.0  # [0,1] → [-1,1]
        
        # 添加批次维度
        image_batch = image_normalized.unsqueeze(0)  # [1, C, H, W]
        
        # 使用PGD攻击（在 [-1,1] 范围内操作）
        protected_batch = self._pgd_attack(image_batch)
        
        # 移除批次维度
        protected = protected_batch.squeeze(0)
        
        # 转换回 [0, 1] 范围（接口要求）
        protected = (protected + 1.0) / 2.0  # [-1,1] → [0,1]
        protected = torch.clamp(protected, 0, 1)  # 确保在有效范围内
        
        return protected
    
    def _pgd_attack(
        self,
        images: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        原始论文中的PGD攻击实现
        
        Args:
            images: 输入图像 [B, C, H, W]
            mask: 可选的掩码
            
        Returns:
            受保护的图像 [B, C, H, W]
        """
        X = images
        
        # 如果没有加载模型，自动加载
        if self.diffusion_model is None:
            try:
                from diffusers import StableDiffusionImg2ImgPipeline
                print(f"Loading model: {self.model_name}")
                self.diffusion_model = StableDiffusionImg2ImgPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16
                ).to(self.device)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Failed to load model: {e}")
                self.diffusion_model = None
        
        # 初始化对抗样本 - 与原始实现完全一致
        X_adv = X.clone().detach() + (torch.rand(*X.shape) * 2 * self.epsilon - self.epsilon).to(self.device)
        
        # 使用tqdm进度条
        pbar = tqdm(range(self.num_steps), desc="PhotoGuard")
        
        for i in pbar:
            # 计算实际步长 
            actual_step_size = self.step_size - (self.step_size - self.step_size / 100) / self.num_steps * i
            
            X_adv.requires_grad_(True)
            
            # 计算损失 - 使用VAE encoder（原始PhotoGuard实现）
            if self.diffusion_model is not None:
                try:
                    # 转换为float16以匹配模型
                    X_adv_input = X_adv.half()  
                    # 使用VAE encoder计算损失，与原始PhotoGuard实现一致
                    vae_output = self.diffusion_model.vae.encode(X_adv_input)
                    loss = vae_output.latent_dist.mean.norm()
                except Exception as e:
                    print(f"Warning: VAE encoder call failed: {e}")
                    loss = torch.norm(X_adv, p=2)
            else:
                loss = torch.norm(X_adv, p=2)
            
            # 更新进度条描述
            pbar.set_description(f"Loss: {loss.item():.4f}, Step: {actual_step_size:.3f}")
            
            # 计算梯度
            grad, = torch.autograd.grad(loss, [X_adv])
            
            # 更新对抗样本
            X_adv = X_adv - grad.detach().sign() * actual_step_size
            
            # 投影到约束集合 
            X_adv = torch.minimum(torch.maximum(X_adv, X - self.epsilon), X + self.epsilon)
            X_adv.data = torch.clamp(X_adv, min=self.clamp_min, max=self.clamp_max)
            
            # 清理梯度 
            X_adv.grad = None
            
            # 应用掩码（如果提供）
            if mask is not None:
                X_adv.data *= mask
        
        X_adv.detach_()
        return X_adv


 
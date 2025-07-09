import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import numpy as np
from .base import AdversarialProtection


class PhotoGuard(AdversarialProtection):
    """
    PhotoGuard: 针对文本到图像扩散模型的保护方法
    基于论文: "PhotoGuard: Disrupting Diffusion-based Generative Models for Copyright Protection"
    """
    
    def __init__(
        self,
        diffusion_model: nn.Module,
        epsilon: float = 0.03,
        num_steps: int = 100,
        step_size: float = 0.001,
        guidance_scale: float = 7.5,
        immunization_strength: float = 1.0,
        **kwargs
    ):
        """
        Args:
            diffusion_model: 扩散模型（如Stable Diffusion）
            epsilon: 扰动预算
            num_steps: 优化步数
            step_size: 优化步长
            guidance_scale: 引导强度
            immunization_strength: 免疫强度
        """
        self.guidance_scale = guidance_scale
        self.immunization_strength = immunization_strength
        super().__init__(
            target_model=diffusion_model,
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            **kwargs
        )
    
    def _setup_model(self):
        """设置模型"""
        # PhotoGuard不需要额外的模型，直接使用扩散模型
        pass
    
    def protect(
        self,
        images: torch.Tensor,
        prompts: Optional[list] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        使用PhotoGuard保护图像
        
        Args:
            images: 输入图像张量 [B, C, H, W]
            prompts: 文本提示列表
            **kwargs: 其他参数
            
        Returns:
            受保护的图像张量
        """
        if prompts is None:
            prompts = ["a photo"] * images.size(0)
        
        # 将图像移到正确的设备
        images = images.to(self.device)
        
        # 使用改进的对抗攻击
        protected_images = self._photoguard_attack(images, prompts)
        
        return protected_images
    
    def _photoguard_attack(
        self,
        images: torch.Tensor,
        prompts: list
    ) -> torch.Tensor:
        """
        PhotoGuard特定的攻击方法
        
        Args:
            images: 输入图像
            prompts: 文本提示
            
        Returns:
            受保护的图像
        """
        batch_size = images.size(0)
        
        # 编码文本提示
        text_embeddings = self._encode_prompts(prompts)
        
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
            
            # 计算扩散损失
            loss = self._compute_diffusion_loss(perturbed_images, text_embeddings)
            
            # 添加免疫损失
            immunization_loss = self._compute_immunization_loss(delta)
            
            total_loss = loss + self.immunization_strength * immunization_loss
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            # 投影到约束集合
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
                delta.data = torch.clamp(delta.data, 0 - images, 1 - images)
            
            if step % 20 == 0:
                print(f"Step {step}, Loss: {total_loss.item():.4f}")
        
        return torch.clamp(images + delta.detach(), 0, 1)
    
    def _encode_prompts(self, prompts: list) -> torch.Tensor:
        """
        编码文本提示
        
        Args:
            prompts: 文本提示列表
            
        Returns:
            文本嵌入张量
        """
        # 这里应该使用实际的文本编码器（如CLIP）
        # 为了演示，返回随机嵌入
        batch_size = len(prompts)
        embedding_dim = 768  # CLIP嵌入维度
        
        # 模拟文本嵌入
        embeddings = torch.randn(batch_size, 77, embedding_dim, device=self.device)
        
        return embeddings
    
    def _compute_diffusion_loss(
        self,
        images: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        计算扩散模型损失
        
        Args:
            images: 图像张量
            text_embeddings: 文本嵌入
            
        Returns:
            损失值
        """
        batch_size = images.size(0)
        
        # 随机时间步
        timesteps = torch.randint(0, 1000, (batch_size,), device=self.device)
        
        # 添加噪声
        noise = torch.randn_like(images)
        
        # 模拟扩散过程
        # 这里应该使用实际的扩散模型前向过程
        noisy_images = images * 0.5 + noise * 0.5
        
        # 预测噪声
        try:
            # 尝试调用扩散模型
            predicted_noise = self.target_model(
                noisy_images,
                timesteps,
                encoder_hidden_states=text_embeddings
            )
            
            # 计算损失（最大化预测误差）
            loss = -F.mse_loss(predicted_noise, noise)
            
        except Exception as e:
            # 如果模型调用失败，使用简化的损失
            print(f"Warning: Diffusion model call failed: {e}")
            loss = -torch.mean(torch.norm(noisy_images - images, p=2, dim=[1, 2, 3]))
        
        return loss
    
    def _compute_immunization_loss(self, delta: torch.Tensor) -> torch.Tensor:
        """
        计算免疫损失（正则化项）
        
        Args:
            delta: 扰动张量
            
        Returns:
            免疫损失值
        """
        # L2正则化
        l2_loss = torch.mean(torch.norm(delta, p=2, dim=[1, 2, 3]))
        
        # 平滑性损失
        smoothness_loss = self._compute_smoothness_loss(delta)
        
        return l2_loss + 0.1 * smoothness_loss
    
    def _compute_smoothness_loss(self, delta: torch.Tensor) -> torch.Tensor:
        """
        计算平滑性损失
        
        Args:
            delta: 扰动张量
            
        Returns:
            平滑性损失值
        """
        # 计算梯度的L2范数
        grad_x = delta[:, :, 1:, :] - delta[:, :, :-1, :]
        grad_y = delta[:, :, :, 1:] - delta[:, :, :, :-1]
        
        smoothness = torch.mean(grad_x ** 2) + torch.mean(grad_y ** 2)
        
        return smoothness


class PhotoGuardEncoder(PhotoGuard):
    """
    PhotoGuard编码器版本，针对编码器-解码器架构
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        **kwargs
    ):
        """
        Args:
            encoder: 编码器（如VAE编码器）
            decoder: 解码器（如VAE解码器）
        """
        self.encoder = encoder
        self.decoder = decoder
        
        # 使用编码器作为目标模型
        super().__init__(target_model=encoder, **kwargs)
    
    def _compute_diffusion_loss(
        self,
        images: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        计算编码器-解码器损失
        
        Args:
            images: 图像张量
            text_embeddings: 文本嵌入（这里可能不使用）
            
        Returns:
            损失值
        """
        # 编码图像
        with torch.no_grad():
            original_latent = self.encoder(images)
        
        # 对扰动图像编码
        perturbed_latent = self.encoder(images)
        
        # 计算潜在空间的差异（最大化差异）
        latent_loss = -F.mse_loss(perturbed_latent, original_latent)
        
        # 可选：添加重构损失
        reconstructed = self.decoder(perturbed_latent)
        reconstruction_loss = -F.mse_loss(reconstructed, images)
        
        return latent_loss + 0.1 * reconstruction_loss 
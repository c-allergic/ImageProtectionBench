import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
import numpy as np
import logging
from .base import ProtectionBase
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import AutoTokenizer, PretrainedConfig
import clip
from scipy import signal
import torchvision.transforms.functional as TF

# 配置日志记录器
logger = logging.getLogger(__name__)


class EditShield(ProtectionBase):
    """
    EditShield: 基于EOT攻击的图像保护方法
    将EOT攻击算法转换为保护算法，生成相同的扰动来保护图像
    支持多种变换：none（无变换）、center（中心裁剪）、gaussian（高斯模糊）、rotation（旋转）
    """
    
    def __init__(
        self,
        protection_strength: float = 0.05,
        max_steps: int = 30,
        beta: float = 0.2,
        perturbation_budget: float = 4/255,  # 扰动预算，默认 4/255
        model_path: str = './protection/instruct-pix2pix-main/diffuser_cache',
        cop_path: str = './protection/instruct-pix2pix-main/cop_file',
        transform_type: str = "none",  # "none", "center", "gaussian", "rotation"
        **kwargs
    ):
        """
        Args:
            protection_strength: 保护强度
            max_steps: 优化步数
            beta: 感知一致性权重
            perturbation_budget: 扰动预算（L∞范数），默认 4/255
            model_path: 模型缓存路径
            cop_path: 模型文件路径
            transform_type: 变换类型 ("none", "center", "gaussian", "rotation")
        """
        self.protection_strength = protection_strength
        self.max_steps = max_steps
        self.beta = beta
        self.perturbation_budget = perturbation_budget
        self.model_path = model_path
        self.cop_path = cop_path
        self.transform_type = transform_type
        self.weight_dtype = torch.bfloat16
        
        super().__init__(**kwargs)
    
    def _setup_model(self):
        """设置模型 - 加载InstructPix2Pix相关模型"""
        model_id = "timbrooks/instruct-pix2pix"
        revision = 'fp16'
        
        # 加载VAE
        self.vae = AutoencoderKL.from_pretrained(
            model_id, 
            subfolder="vae", 
            revision=revision, 
            cache_dir=self.cop_path
        ).to(self.device)
        self.vae.requires_grad_(False)
        self.vae.to(self.device, dtype=self.weight_dtype)
        
        # 加载噪声调度器
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            model_id, 
            subfolder="scheduler", 
            cache_dir=self.cop_path
        )
        
        # 加载CLIP模型用于特征提取
        self.clip_model, _ = clip.load(
            "./protection/instruct-pix2pix-main/clip-vit-large-patch14/ViT-L-14.pt", 
            device="cpu", 
            download_root="./"
        )
        self.clip_model.eval().requires_grad_(False)
        
        logger.info(f"EditShield模型加载完成，设备: {self.device}, 变换类型: {self.transform_type}")
    
    def get_emb(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: 输入图像张量
            
        Returns:
            嵌入张量
        """
        latents_1 = self.vae.encode(img.to(self.device, dtype=self.weight_dtype)).latent_dist.sample()
        latents = latents_1 * self.vae.config.scaling_factor

        # 采样噪声
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # 添加噪声（前向扩散过程）
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # 获取原始图像嵌入用于条件化
        original_image_embeds = self.vae.encode(img.to(self.device, dtype=self.weight_dtype)).latent_dist.sample()

        # 拼接噪声潜在表示和原始图像嵌入
        concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)
        return concatenated_noisy_latents
    
    def perceptual_consistency_loss(self, perturbed_images: torch.Tensor, original_images: torch.Tensor) -> torch.Tensor:
        """
        感知一致性损失
        
        Args:
            perturbed_images: 扰动图像
            original_images: 原始图像
            
        Returns:
            损失值
        """
        l2_norm = F.mse_loss(perturbed_images, original_images)
        loss = self.beta * l2_norm
        return loss
    
    def center_crop(self, images: torch.Tensor, new_height: int, new_width: int) -> torch.Tensor:
        """
        应用中心裁剪变换
        
        Args:
            images: 输入图像张量
            new_height: 新高度
            new_width: 新宽度
            
        Returns:
            裁剪后的图像
        """
        _, _, height, width = images.shape
        startx = width // 2 - (new_width // 2)
        starty = height // 2 - (new_height // 2)
        endx = startx + new_width
        endy = starty + new_height
        return images[:, :, starty:endy, startx:endx]
    
    def gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """
        生成高斯核
        
        Args:
            size: 核大小
            sigma: 标准差
            
        Returns:
            高斯核张量
        """
        gkern1d = torch.from_numpy(np.outer(signal.gaussian(size, sigma), signal.gaussian(size, sigma)))
        gkern2d = gkern1d.float().unsqueeze(0).unsqueeze(0)
        return gkern2d
    
    def apply_gaussian_smoothing(self, input_tensor: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        """
        应用高斯平滑
        
        Args:
            input_tensor: 输入张量
            kernel_size: 核大小
            sigma: 标准差
            
        Returns:
            平滑后的张量
        """
        kernel = self.gaussian_kernel(kernel_size, sigma)
        smoothed_channels = []
        for c in range(input_tensor.shape[1]):
            channel = input_tensor[:, c:c + 1, :, :]
            padding = kernel_size // 2
            smoothed_channel = F.conv2d(channel, kernel, padding=padding)
            smoothed_channels.append(smoothed_channel)
        smoothed = torch.cat(smoothed_channels, dim=1)
        return smoothed
    
    def apply_transform(self, images: torch.Tensor) -> torch.Tensor:
        """
        根据变换类型应用相应的变换 - 与原始代码保持一致
        
        Args:
            images: 输入图像张量
            
        Returns:
            变换后的图像
        """
        if self.transform_type == "none":
            # 无变换，直接返回原图像
            return images
        
        elif self.transform_type == "center":
            resolution = min(images.size(2), images.size(3))
            return self.center_crop(images, resolution, resolution)
        
        elif self.transform_type == "gaussian":
            kernel_size = 5
            sigma = 1.5
            gaussian_data = self.apply_gaussian_smoothing(images, kernel_size, sigma)
            return torch.clamp(gaussian_data, min=0, max=1)
        
        elif self.transform_type == "rotation":
            angle = 5
            if images.dim() == 3:
                images = images.unsqueeze(0)
                was_batch = False
            else:
                was_batch = True

            rotated_images = torch.stack([TF.rotate(img, angle) for img in images])
            if not was_batch:
                rotated_images = rotated_images.squeeze(0)
            
            return rotated_images
        
        else:
            raise ValueError(f"不支持的变换类型: {self.transform_type}")
    
    def _apply_eot_protection(self, images: torch.Tensor) -> torch.Tensor:
        """
        应用EOT保护 - 基于第一个EditShield的攻击算法
        
        Args:
            images: 输入图像张量 [B, C, H, W]
            
        Returns:
            受保护的图像张量 [B, C, H, W]
        """
        batch_size = images.shape[0]
        protected_images = []
        
        for i in range(batch_size):
            single_image = images[i:i+1]  # [1, C, H, W]
            
            # 应用变换（根据transform_type）
            perturbed_data = self.apply_transform(single_image)
            
            # 保存原始数据
            original_data = perturbed_data.clone()
            
            # 初始化对抗样本，添加小的高斯噪声
            perturbed_images_single = perturbed_data.detach().clone()
            perturbed_images_single = perturbed_images_single + torch.randn_like(perturbed_images_single) * 0.01  # 添加标准差为0.01的高斯噪声
            perturbed_images_single = torch.clamp(perturbed_images_single, -1, 1)  # 保持在合理范围内
            
            # 获取目标嵌入（未变换的原始图像）
            tgt_data = single_image.clone()
            tgt_emb = self.get_emb(tgt_data).detach().clone()
            
            # 设置优化器
            optimizer = torch.optim.Adam([perturbed_images_single])
            
            # EOT攻击迭代（转换为保护）
            for step in range(self.max_steps):
                perturbed_images_single.requires_grad = True
                img_emb = self.get_emb(perturbed_images_single)
                optimizer.zero_grad()
                
                # 损失函数计算（与攻击相同，但用于保护）
                loss_perceptual = self.perceptual_consistency_loss(perturbed_images_single, original_data)
                loss_mse = -F.mse_loss(img_emb.float(), tgt_emb.float())  # 最大化嵌入空间差异
                
                # 对于高斯模糊变换，添加额外的平滑损失
                if self.transform_type == "gaussian":
                    kernel_size = 5
                    sigma = 1.5
                    gaussian_data = self.apply_gaussian_smoothing(original_data, kernel_size, sigma)
                    loss_smooth = F.mse_loss(
                        self.apply_gaussian_smoothing(perturbed_images_single, kernel_size, sigma), 
                        gaussian_data
                    )
                    total_loss = loss_mse + 0.1 * loss_smooth + loss_perceptual
                else:
                    total_loss = loss_mse + loss_perceptual
                
                total_loss.backward()
                optimizer.step()
                
                # 应用扰动预算限制（L∞范数）
                with torch.no_grad():
                    perturbation = perturbed_images_single - single_image
                    perturbation = torch.clamp(perturbation, -self.perturbation_budget, self.perturbation_budget)
                    perturbed_images_single.data = single_image + perturbation
                    perturbed_images_single.data = torch.clamp(perturbed_images_single.data, 0, 1)
            
            # 获取最终的受保护图像
            protected_single = perturbed_images_single.detach()
            protected_images.append(protected_single)
        
        return torch.cat(protected_images, dim=0)
    
    def protect(
        self,
        image: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        使用EditShield保护单张图像
        
        Args:
            image: 输入图像张量 [C, H, W]，范围[0,1]
            **kwargs: 其他参数
            
        Returns:
            受保护的图像张量 [C, H, W]，范围[0,1]
        """
        # 确保图像在正确设备上
        image = image.to(self.device)
        
        # 添加批次维度进行处理
        images = image.unsqueeze(0)  # [1, C, H, W]
        
        # 应用EOT保护
        protected_images = self._apply_eot_protection(images)
        
        # 移除批次维度并确保范围在[0,1]
        protected_image = protected_images.squeeze(0)
        protected_image = torch.clamp(protected_image, 0, 1)
        
        return protected_image
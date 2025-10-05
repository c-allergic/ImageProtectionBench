"""
ExpGuard - 防止爆炸语义prompt的图生视频攻击
借鉴Tarpro的目标函数，优化对抗损失和正则化损失
"""
import torch
import torch.nn.functional as F
from typing import Union, List
from .base import ProtectionBase
from data import pt_to_pil


class ExpGuard(ProtectionBase):
    """
    ExpGuard防护算法
    
    目标：防止用户使用带有爆炸语义的prompt对图片进行图生视频操作
    方法：借鉴Tarpro目标函数，优化两个loss：
    - L_adv: 对抗损失，使(加扰图片+爆炸prompt)输出接近(原图+正常prompt)
    - L_reg: 正则化损失，保证(加扰图片+正常prompt)与(原图+正常prompt)一致
    """
    
    def _setup_model(self):
        """
        初始化ExpGuard模型参数
        """
        # 默认超参数
        self.eps = 8 / 255  # 扰动强度上界
        self.num_steps = 50  # 优化迭代次数
        self.lr = 0.01  # 学习率
        self.lambda_reg = 1.0  # 正则化损失权重，类似Tarpro
        
        # 初始化图生视频生成器 g(.)
        self._setup_generator()
        
    def _setup_generator(self):
        """
        设置图生视频生成器
        使用AnimateDiff作为生成器g(.)，支持文本prompt控制
        """
        from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
        from diffusers.utils import export_to_video
        
        # 加载motion adapter
        adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
        
        # 加载pipeline
        self.generator = AnimateDiffPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            motion_adapter=adapter,
            torch_dtype=torch.float16,
        ).to(self.device)
        
        # 使用DDIM调度器
        self.generator.scheduler = DDIMScheduler.from_config(self.generator.scheduler.config)
        
        # 预定义prompt
        self.y_exp = "explosion, bomb blast, fire and destruction, debris flying everywhere"  # 爆炸语义prompt
        self.y_nor = "peaceful scene, calm and serene environment, gentle natural movement"  # 正常prompt
        
    @torch.enable_grad()
    def protect(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        保护单张图片免受爆炸语义prompt攻击
        
        Args:
            image: 输入图片张量 [C, H, W]，范围[0,1]
            
        Returns:
            受保护的图片张量 [C, H, W]，范围[0,1]
        """
        assert image.dim() == 3 and image.size(0) in {3, 4}
        
        # 将图片移到设备并添加batch维度
        I0 = image.unsqueeze(0).to(self.device)  # [1, C, H, W]
        
        # 初始化扰动δ
        delta = torch.zeros_like(I0, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.lr)
        
        # 计算原图+正常prompt的基准输出 g(I0, y_nor)
        with torch.no_grad():
            baseline_output = self._generate_video(I0, self.y_nor)
        
        # 迭代优化扰动
        for step in range(self.num_steps):
            optimizer.zero_grad()
            
            # 计算加扰图片
            I_perturbed = torch.clamp(I0 + delta, 0, 1)
            
            # 计算两个loss项
            loss_adv = self._compute_adversarial_loss(I_perturbed, I0, baseline_output)
            loss_reg = self._compute_regularization_loss(I_perturbed, I0, baseline_output)
            
            # 总损失
            total_loss = loss_adv + self.lambda_reg * loss_reg
            
            total_loss.backward()
            optimizer.step()
            
            # 限制扰动强度
            with torch.no_grad():
                delta.data = torch.clamp(delta, -self.eps, self.eps)
                
        # 返回受保护的图片
        protected_image = torch.clamp(I0 + delta, 0, 1).squeeze(0).detach()
        return protected_image
    
    def _generate_video(self, image: torch.Tensor, prompt: str) -> torch.Tensor:
        """
        使用生成器g(.)生成视频特征
        
        Args:
            image: 输入图片 [1, C, H, W] 
            prompt: 文本prompt
            
        Returns:
            生成的视频特征表示
        """
        # 调整图片尺寸到模型要求 (512x512)
        image_resized = F.interpolate(image, (512, 512), mode="bilinear", align_corners=False)
        
        # 生成视频特征，这里我们获取UNet的中间表示用于loss计算
        with torch.no_grad():
            # 编码文本prompt
            text_embeddings = self.generator._encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )
            
            # 编码图片到潜在空间
            image_latents = self.generator.vae.encode(2 * image_resized - 1).latent_dist.sample()
            image_latents = image_latents * self.generator.vae.config.scaling_factor
            
            # 扩展到视频时间维度 (AnimateDiff通常使用16帧)
            batch_size, channels, height, width = image_latents.shape
            num_frames = 16
            image_latents = image_latents.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)
            
            # 添加噪声用于UNet前向传播  
            timestep = torch.tensor([500], device=self.device, dtype=torch.long)
            noise = torch.randn_like(image_latents)
            
            # 使用调度器添加噪声
            noisy_latents = self.generator.scheduler.add_noise(image_latents, noise, timestep)
            
            # 通过UNet获取特征表示
            features = self.generator.unet(
                noisy_latents,
                timestep,
                encoder_hidden_states=text_embeddings,
                return_dict=False
            )[0]
            
        return features
    
    def _compute_adversarial_loss(self, I_perturbed: torch.Tensor, I0: torch.Tensor, baseline_output: torch.Tensor) -> torch.Tensor:
        """
        计算对抗损失 L_adv = ||g(I0+δ, y_exp) - g(I0, y_nor)||²
        
        Args:
            I_perturbed: 加扰图片 I0+δ
            I0: 原始图片
            baseline_output: 基准输出 g(I0, y_nor)
            
        Returns:
            对抗损失
        """
        # 计算 g(I0+δ, y_exp)
        perturbed_exp_output = self._generate_video(I_perturbed, self.y_exp)
        
        # 计算L2距离
        loss_adv = F.mse_loss(perturbed_exp_output, baseline_output)
        
        return loss_adv
    
    def _compute_regularization_loss(self, I_perturbed: torch.Tensor, I0: torch.Tensor, baseline_output: torch.Tensor) -> torch.Tensor:
        """
        计算正则化损失 L_reg = ||g(I0+δ, y_nor) - g(I0, y_nor)||²
        
        Args:
            I_perturbed: 加扰图片 I0+δ  
            I0: 原始图片
            baseline_output: 基准输出 g(I0, y_nor)
            
        Returns:
            正则化损失
        """
        # 计算 g(I0+δ, y_nor)
        perturbed_nor_output = self._generate_video(I_perturbed, self.y_nor)
        
        # 计算L2距离
        loss_reg = F.mse_loss(perturbed_nor_output, baseline_output)
        
        return loss_reg
    

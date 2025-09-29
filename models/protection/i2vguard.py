"""
I2VGuard 完整实现（含真实 Attention Hook）
继承 ProtectionBase，直接运行即可得到受保护图像
"""
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from typing import Union, List
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image
from safetensors.torch import load_file
from .base import ProtectionBase
import os

class I2VGuard(ProtectionBase):
    def _setup_model(self):
        # 默认超参（论文 Table 1）
        self.eps = 8 / 255
        self.num_step = 50
        self.alpha = 1.0
        self.beta = 1.0
        self.gamma = 1.0
        self.tau1 = 0.5
        self.tau2 = 0.5
        self.lr = self.eps / 0.35 / self.num_step

        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid",
                torch_dtype=torch.float32,  # 改为float32避免类型不匹配
            ).to(self.device)
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.scheduler = self.pipe.scheduler

        # 定位最后一个 temporal-transformer 的 self-attention 层
        target_blocks = self.unet.down_blocks[2].attentions[1].transformer_blocks
        self._attn_layer = target_blocks[-1].attn1
        self._handle = None
        self._attn_buffer = []

        # 垃圾目标潜变量-全黑
        self.Xtg = torch.zeros(1, 14, 4, 64, 64, dtype=torch.float32, device=self.device)

    # ------------ 注册/清理 hook ------------
    def _register_attention_hook(self):
        if self._handle is not None:
            return
        def hook_fn(module, input, output):
            attn_weights = output[1]  # [B, F, F]
            self._attn_buffer.append(attn_weights.detach().clone())
        self._handle = self._attn_layer.register_forward_hook(hook_fn)

    def _clear_attention_hook(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._attn_buffer.clear()

    # ------------ 核心保护接口 ------------
    @torch.enable_grad()
    def protect(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        assert image.dim() == 3 and image.size(0) in {3, 4}
        image = image.unsqueeze(0).to(self.device)  # [1, C, H, W]

        x_src = self._encode_image(image)
        delta = torch.zeros_like(image, requires_grad=True) # 噪声
        opt = torch.optim.Adam([delta], lr=self.lr) # 优化器

        for _ in range(self.num_step):
            opt.zero_grad()
            x_adv = self._encode_image(torch.clamp(image + delta, 0, 1))
            loss = self._compute_loss(x_adv, x_src)
            loss.backward()
            opt.step()
            with torch.no_grad():
                delta.data = torch.clamp(delta, -self.eps, self.eps)

        return torch.clamp(image + delta, 0, 1).squeeze(0).detach()

    # ------------ 内部工具 ------------
    def _encode_image(self, image):
        image = F.interpolate(image, (512, 512), mode="bilinear", align_corners=False)
        image = 2 * image - 1
        latents = self.vae.encode(image).latent_dist.sample()
        return latents  # 移除.half()，保持float32类型

    def _compute_loss(self, x_adv, x_src):
        # Spatial
        L_enc = F.mse_loss(x_adv, torch.zeros_like(x_adv))

        # 构造 X_t
        B, C, H, W = x_adv.shape
        X0 = x_src.unsqueeze(1).repeat(1, 14, 1, 1, 1)
        X_adv = x_adv.unsqueeze(1).repeat(1, 14, 1, 1, 1)
        noise = torch.randn_like(X0)
        t = torch.tensor([50], device=self.device, dtype=torch.long)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        sac = alphas_cumprod[t].view(-1, 1, 1, 1, 1) ** 0.5
        som = (1 - alphas_cumprod[t]).view(-1, 1, 1, 1, 1) ** 0.5
        X_t = sac * X0 + som * noise

        # Diffusion
        self._attn_buffer.clear()
        self._register_attention_hook()
        with torch.amp.autocast('cuda'):
            print(self.unet.config)
            noise_pred = self.unet(
                sample=X_t, 
                timestep=t, 
                encoder_hidden_states=None,
                image_latents=X_adv
            ).sample
        X0_hat_adv = (X_t - som * noise_pred) / sac
        pull = F.mse_loss(X0_hat_adv, self.Xtg)
        push = F.mse_loss(X0_hat_adv, X0)
        L_con = pull + F.relu(self.tau1 - push)

        # Temporal
        self._attn_buffer.clear()
        with torch.no_grad():
            _ = self.unet(
                sample=X_t, 
                timestep=t, 
                encoder_hidden_states=None,
                image_latents=X0
            )
        A_src = self._attn_buffer[-1].float()

        self._attn_buffer.clear()
        with torch.amp.autocast('cuda'):
            print(self.unet.config)
            _ = self.unet(
                sample=X_t, 
                timestep=t, 
                encoder_hidden_states=None,
                image_latents=X_adv
            )
        A_adv = self._attn_buffer[-1].float()
        L_att = self.tau2 - F.mse_loss(A_adv, A_src)

        self._clear_attention_hook()
        return self.alpha * L_enc + self.beta * L_con + self.gamma * L_att



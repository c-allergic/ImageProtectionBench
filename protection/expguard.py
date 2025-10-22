"""
ExpGuard - 防止爆炸语义prompt的图生视频攻击
借鉴Tarpro的目标函数，优化对抗损失和正则化损失
改进：使用 YCbCr 颜色空间的中频频域扰动
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List
from .base import ProtectionBase
from data import pt_to_pil
import kornia.color as K_color
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import json
import logging
logging.getLogger().setLevel(logging.ERROR)

class FrequencyNoiseGenerator(nn.Module):
    """
    在 YCbCr 的 Cb 和 Cr 通道上生成多频段扰动（低频、中频、高频）
    """
    
    def __init__(self, H: int, W: int, device: torch.device, 
                 low_freq_ratio: float = 0.05,    # 低频扰动范围
                 mid_freq_max: float = 0.3,        # 中频上界
                 high_freq_ratio: float = 0.6,     # 高频扰动范围
                 init_scale: float = 0.1,          # 初始化噪声的标准差
                 freq_weights: dict = None):       # 频段权重
        """
        初始化多频段噪声参数
        
        Args:
            H, W: 图像高度和宽度
            device: 设备
            low_freq_ratio: 低频内半径比例
            mid_freq_max: 中频上界
            high_freq_ratio: 高频外半径比例
            init_scale: 初始化噪声的标准差
            freq_weights: 频段权重字典
        """
        super().__init__()
        
        self.H, self.W = H, W
        
        # 设置频段权重
        if freq_weights is None:
            self.freq_weights = {'low': 1.0, 'mid': 1.0, 'high': 0.5}
        else:
            self.freq_weights = freq_weights
        
        # 为不同频段分配不同参数
        self.noise_low = nn.Parameter(torch.randn(2, H, W, device=device) * init_scale)
        self.noise_mid = nn.Parameter(torch.randn(2, H, W, device=device) * init_scale)  
        self.noise_high = nn.Parameter(torch.randn(2, H, W, device=device) * init_scale)
        
        # 创建并注册多频段掩码
        self.register_buffer('low_freq_mask', self._create_freq_mask(H, W, 0, low_freq_ratio))
        self.register_buffer('mid_freq_mask', self._create_freq_mask(H, W, low_freq_ratio, mid_freq_max))
        self.register_buffer('high_freq_mask', self._create_freq_mask(H, W, mid_freq_max, high_freq_ratio))
    
    def _create_freq_mask(self, H: int, W: int, r_min_ratio: float, r_max_ratio: float) -> torch.Tensor:
        """
        创建频段掩码
        
        Args:
            H, W: 图像高度和宽度
            r_min_ratio: 内半径比例
            r_max_ratio: 外半径比例
            
        Returns: [2, H, W] 的掩码，值为 1.0（目标频段）或 0.0（其他频段）
        """
        # 计算频率坐标（中心化）
        freq_y = torch.fft.fftfreq(H).reshape(-1, 1)  # [H, 1]
        freq_x = torch.fft.fftfreq(W).reshape(1, -1)  # [1, W]
        
        # 计算每个频率点到中心的距离（归一化）
        distance = torch.sqrt(freq_y**2 + freq_x**2)  # [H, W]
        
        # 定义频段范围
        max_freq = 0.5  # Nyquist 频率
        r_min = r_min_ratio * max_freq
        r_max = r_max_ratio * max_freq
        
        # 创建环形掩码
        mask = ((distance >= r_min) & (distance <= r_max)).float()  # [H, W]
        
        # 为 Cb 和 Cr 两个通道复制
        mask = mask.unsqueeze(0).repeat(2, 1, 1)  # [2, H, W]
        
        return mask
    
    def forward(self, image_ycbcr: torch.Tensor) -> torch.Tensor:
        """
        对 YCbCr 图像的 Cb/Cr 通道进行多频段扰动
        
        Args:
            image_ycbcr: [B, 3, H, W], YCbCr 图像
            
        Returns:
            perturbed_ycbcr: [B, 3, H, W], 扰动后的 YCbCr 图像
        """
        B, C, H, W = image_ycbcr.shape
        
        # 分离 Y 和 CbCr 通道
        Y = image_ycbcr[:, 0:1, :, :]      # [B, 1, H, W]
        CbCr = image_ycbcr[:, 1:3, :, :]   # [B, 2, H, W]
        
        # 1. DFT 变换到频域
        CbCr_freq = torch.fft.fft2(CbCr)  # [B, 2, H, W] 复数
        
        # 2. 构造多频段噪声
        # 低频噪声
        noise_low_complex = torch.complex(self.noise_low, torch.zeros_like(self.noise_low))
        masked_noise_low = noise_low_complex * self.low_freq_mask
        
        # 中频噪声
        noise_mid_complex = torch.complex(self.noise_mid, torch.zeros_like(self.noise_mid))
        masked_noise_mid = noise_mid_complex * self.mid_freq_mask
        
        # 高频噪声
        noise_high_complex = torch.complex(self.noise_high, torch.zeros_like(self.noise_high))
        masked_noise_high = noise_high_complex * self.high_freq_mask
        
        # 3. 合并所有频段的噪声（应用权重）
        total_noise = (self.freq_weights['low'] * masked_noise_low + 
                      self.freq_weights['mid'] * masked_noise_mid + 
                      self.freq_weights['high'] * masked_noise_high)  # [2, H, W]
        
        # 4. 扩展到 batch 维度并加到频域上
        total_noise_batch = total_noise.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]
        perturbed_CbCr_freq = CbCr_freq + total_noise_batch
        
        # 5. IDFT 转换回空间域
        perturbed_CbCr = torch.fft.ifft2(perturbed_CbCr_freq).real  # [B, 2, H, W]
        
        # 6. 组合 Y 和扰动后的 CbCr
        perturbed_ycbcr = torch.cat([Y, perturbed_CbCr], dim=1)  # [B, 3, H, W]
        
        return perturbed_ycbcr


class ExpGuard(ProtectionBase):
    """
    ExpGuard防护算法
    
    目标：防止用户使用带有爆炸语义的prompt对图片进行图生视频操作
    方法：借鉴Tarpro目标函数，优化两个loss：
    - L_adv: 对抗损失，使(加扰图片+爆炸prompt)输出远离(原图+正常prompt)
    - L_reg: 正则化损失，保证(加扰图片+正常prompt)与(原图+正常prompt)相似
    """
    
    def _setup_model(self):
        """
        初始化ExpGuard模型参数
        """
        # 默认超参数
        self.eps = 8 / 255  # 扰动强度上界
        self.num_steps = 50  # 优化迭代次数
        self.lr = 0.85
        self.lambda_reg = 0.2  # 正则化损失权重
        self.init_scale = 0.1 
        self.lr_scheduler_type = "cosine"  # 学习率调度器类型
        
        # 多频段扰动配置
        self.freq_weights = {'low': 3, 'mid': 6, 'high': 3}
        
        # 显存优化配置
        self.gradient_frame_ratio = 0.5  
        self.gradient_frame_strategy = "uniform"
        
        # 初始化图生视频生成器
        self._setup_generator()
    def _setup_generator(self):
        """
        加载SVD DiffusionEngine为统一特征提取器/视频生成pipeline
        """
        import sys, os
        sys.path.append('/data_sde/lxf/generative-models') 
        from omegaconf import OmegaConf
        from sgm.util import instantiate_from_config
        svd_config = OmegaConf.load('/data_sde/lxf/generative-models/scripts/sampling/configs/svd.yaml')
        self.svd_pipe = instantiate_from_config(svd_config["model"]).to(self.device)
        
        # 设置采样器参数（优化显存使用）
        self.svd_num_steps = 25  # 采样步数（用于计算sigma）
        # 注意：实际不使用完整采样，仅单步denoising以节省显存
        
        # 预定义prompt
        self.y_exp = "explosion, bomb blast, fire and destruction, debris flying everywhere"
        self.y_nor = "peaceful scene, calm and serene environment, gentle natural movement"
    
        
    @torch.enable_grad()
    def protect(self, image: torch.Tensor, **kwargs) -> torch.Tensor: # 使用 YCbCr 频域中频扰动保护图片免受爆炸语义prompt攻击
        assert image.dim() == 3 and image.size(0) in {3, 4}
        
        # 将图片移到设备并添加batch维度
        I0_rgb = image.unsqueeze(0).to(self.device)  # [1, C, H, W], RGB [0,1]
        _, C, H, W = I0_rgb.shape
        
        # 1. 初始化多频段噪声生成器
        noise_generator = FrequencyNoiseGenerator(
            H, W, self.device, 
            low_freq_ratio=0.05,    # 低频扰动范围 [0, 0.05]
            mid_freq_max=0.3,       # 中频上界 [0.05, 0.3]
            high_freq_ratio=0.6,    # 高频扰动范围 [0.3, 0.6]
            init_scale=self.init_scale,
            freq_weights=self.freq_weights  # 传递频段权重
        ).to(self.device)
        optimizer = torch.optim.Adam(
            list(noise_generator.parameters()),
            lr=self.lr
        )
        
        # 2. 创建学习率调度器
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_steps, eta_min=self.lr * 0.01)
        
        # 2. 计算原图的基准输出（预计算，避免重复生成）
        # 显存优化信息输出
        num_gradient_frames = max(1, int(16 * self.gradient_frame_ratio))
        
        with torch.no_grad():
            baseline_output = self._generate_video(I0_rgb, self.y_nor)  # g(I0, y_nor)
            original_exp_output = self._generate_video(I0_rgb, self.y_exp)  # g(I0, y_exp)
        
        # 3. 迭代优化频域扰动参数
        pbar = tqdm(range(self.num_steps), desc="频域扰动优化", ncols=100)
        
        # 记录loss和学习率历史
        loss_history = {
            'total': [],
            'adv': [],
            'reg': [],
            'lr': [],
        }
        
        for step in pbar:
            optimizer.zero_grad()
            
            # 分步计算loss并清理显存
            
            # 步骤1: 生成扰动图片并计算对抗损失
            I0_ycbcr = K_color.rgb_to_ycbcr(I0_rgb)
            I_perturbed_ycbcr = noise_generator(I0_ycbcr)
            I_perturbed_rgb = K_color.ycbcr_to_rgb(I_perturbed_ycbcr)
            
            # 计算对抗损失：使(加扰图片+爆炸prompt)输出远离(原图+爆炸prompt)
            perturbed_exp_output = self._generate_video(I_perturbed_rgb, self.y_exp)
            loss_adv = -F.mse_loss(perturbed_exp_output, original_exp_output)
            loss_adv_item = loss_adv.item()  # 先记录值
            loss_adv.backward(retain_graph=False)  # 不保留计算图
            
            # 清理loss_adv的计算图显存
            del loss_adv
            torch.cuda.empty_cache()
            
            # 步骤2: 重新生成扰动图片并计算正则化损失
            I0_ycbcr = K_color.rgb_to_ycbcr(I0_rgb)
            I_perturbed_ycbcr = noise_generator(I0_ycbcr)
            I_perturbed_rgb = K_color.ycbcr_to_rgb(I_perturbed_ycbcr)
            
            # 计算正则化损失：保证(加扰图片+正常prompt)与(原图+正常prompt)相似
            perturbed_nor_output = self._generate_video(I_perturbed_rgb, self.y_nor)
            loss_reg = F.mse_loss(perturbed_nor_output, baseline_output)
            loss_reg_item = loss_reg.item()  # 记录损失值
            loss_reg.backward(retain_graph=False)
            
            # 清理计算图显存
            del loss_reg
            torch.cuda.empty_cache()
            
            # 梯度裁剪防止爆炸（包括所有优化的参数）
            all_params = list(noise_generator.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            
            optimizer.step()
            
            # 更新学习率调度器
            lr_scheduler.step()
            # 记录loss值（使用之前记录的值）
            total_loss_item = loss_adv_item + self.lambda_reg * loss_reg_item 
            loss_history['total'].append(total_loss_item)
            loss_history['adv'].append(loss_adv_item)
            loss_history['reg'].append(loss_reg_item)
            
            # 获取当前学习率用于显示和记录
            current_lr = optimizer.param_groups[0]['lr']
            loss_history['lr'].append(current_lr)
            
            # 更新进度条显示（使用记录的值）
            pbar.set_postfix({
                'Total': f'{total_loss_item:.4f}',
                'L_adv': f'{loss_adv_item:.4f}',
                'L_reg': f'{loss_reg_item:.4f}',
                'LR': f'{current_lr:.4f}'
            })
        
        # 4. 生成最终受保护的图片
        with torch.no_grad():
            final_I_ycbcr = K_color.rgb_to_ycbcr(I0_rgb)
            final_perturbed_ycbcr = noise_generator(final_I_ycbcr)
            protected_rgb = K_color.ycbcr_to_rgb(final_perturbed_ycbcr)
            
            # 最终硬裁剪到 eps 范围内，确保不可察觉性
            delta_final = protected_rgb - I0_rgb
            delta_clipped = torch.clamp(delta_final, -self.eps, self.eps)
            protected_image = torch.clamp(I0_rgb + delta_clipped, 0, 1)
            
            print(f"最终扰动范围: [{delta_clipped.min():.4f}, {delta_clipped.max():.4f}]")
        
        # 绘制loss曲线
        self._plot_loss_curves(loss_history)
        
        return protected_image.squeeze(0).detach()
    
    def _generate_video(self, image: torch.Tensor, prompt: str) -> torch.Tensor:
        """
        使用SVD单步denoising获取视频特征，支持梯度传播
        优化显存：不使用完整采样，仅进行单步前向传播
        """
        import torch.nn.functional as F
        from einops import rearrange, repeat
        
        # 使用更小分辨率减少显存：256x256
        image_resized = F.interpolate(image, (256, 256), mode="bilinear", align_corners=False)
        image_resized = image_resized.to(dtype=torch.float32)
        
        # 1. 构造batch（减少帧数以节省显存）
        num_frames = 8  # 从14减少到8帧，大幅减少显存
        H, W = image_resized.shape[2:]
        F_downscale = 8
        C = 4
        shape = (num_frames, C, H // F_downscale, W // F_downscale)  # [8, 4, 32, 32]
        
        # 2. 构造batch
        batch = {
            "cond_frames_without_noise": image_resized,
            "cond_frames": image_resized + 0.02 * torch.randn_like(image_resized),
            "motion_bucket_id": torch.tensor([127], device=self.device),
            "fps_id": torch.tensor([6], device=self.device),
            "cond_aug": torch.tensor([0.02], device=self.device),
            "num_video_frames": num_frames,
        }
        
        batch_uc = {k: (torch.clone(v) if isinstance(v, torch.Tensor) else v) 
                    for k, v in batch.items()}
        
        # 3. 获取条件嵌入
        c, uc = self.svd_pipe.conditioner.get_unconditional_conditioning(
            batch, batch_uc=batch_uc,
            force_uc_zero_embeddings=["cond_frames", "cond_frames_without_noise"],
        )
        
        # 4. 扩展到时间维度
        for k in ["crossattn", "concat"]:
            if k in c:
                uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)
        
        if "vector" in c:
            uc["vector"] = repeat(uc["vector"], "b d -> (b t) d", t=num_frames)
            c["vector"] = repeat(c["vector"], "b d -> (b t) d", t=num_frames)
        
        # 5. 使用单步denoising而非完整采样（节省显存）
        # 临时更新guider的num_frames以匹配当前帧数
        original_num_frames = self.svd_pipe.sampler.guider.num_frames
        original_scale = self.svd_pipe.sampler.guider.scale
        self.svd_pipe.sampler.guider.num_frames = num_frames
        self.svd_pipe.sampler.guider.scale = torch.linspace(
            self.svd_pipe.sampler.guider.min_scale,
            self.svd_pipe.sampler.guider.max_scale,
            num_frames
        ).unsqueeze(0)
        
        # 生成噪声latent
        randn = torch.randn(shape, device=self.device)
        
        # 获取中等噪声级别的sigma
        sigmas = self.svd_pipe.sampler.discretization(
            self.svd_num_steps, device=self.device
        )
        sigma = sigmas[len(sigmas) // 2]  # 使用中间时刻
        
        # 添加噪声
        noisy_latent = randn * torch.sqrt(1.0 + sigma ** 2.0)
        
        # 单步denoising获取特征
        additional_model_inputs = {
            "image_only_indicator": torch.zeros(2, num_frames, device=self.device),
            "num_video_frames": num_frames
        }
        
        # 准备sigma - 需要创建与batch匹配的tensor
        s_in = noisy_latent.new_ones([noisy_latent.shape[0]])  # [num_frames]
        
        # 使用guider准备输入（包含CFG）
        x_in, sigma_in, c_in = self.svd_pipe.sampler.guider.prepare_inputs(
            noisy_latent, s_in * sigma, c, uc
        )
        
        # 单步前向传播获取denoised特征
        denoised = self.svd_pipe.denoiser(
            self.svd_pipe.model, x_in, sigma_in, c_in, **additional_model_inputs
        )
        
        # 应用guider得到最终特征
        output = self.svd_pipe.sampler.guider(denoised, sigma_in)
        
        # 恢复原始guider配置
        self.svd_pipe.sampler.guider.num_frames = original_num_frames
        self.svd_pipe.sampler.guider.scale = original_scale
        
        return output
    

    def _plot_loss_curves(self, loss_history: dict):
        """
        Plot and save loss curves
        
        Args:
            loss_history: Dictionary containing the history of each loss
        """
        plt.figure(figsize=(24, 12))
        
        # Subplot 1: Total Loss
        plt.subplot(2, 5, 1)
        plt.plot(loss_history['total'], linewidth=2, color='#2E86AB')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Total Loss', fontsize=12)
        plt.title('Total Loss', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Adversarial Loss
        plt.subplot(2, 5, 2)
        plt.plot(loss_history['adv'], linewidth=2, color='#A23B72')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Adversarial Loss', fontsize=12)
        plt.title('L_adv: Adversarial Loss', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Regularization Loss
        plt.subplot(2, 5, 3)
        plt.plot(loss_history['reg'], linewidth=2, color='#F18F01')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Regularization Loss', fontsize=12)
        plt.title('L_reg: Regularization Loss', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Learning Rate
        plt.subplot(2, 5, 4)
        plt.plot(loss_history['lr'], linewidth=2, color='#4CAF50')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Learning Rate (cosine)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # 使用对数坐标轴更好地显示学习率变化   
        
        plt.tight_layout()
        
        # Save to current working directory
        save_path = os.path.join(os.getcwd(), 'loss_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nLoss curves saved to: {save_path}")
        
        # Print final statistics
        print(f"\n=== Loss & Learning Rate Statistics ===")
        print(f"Total Loss:     Initial={loss_history['total'][0]:.4f}, Final={loss_history['total'][-1]:.4f}")
        print(f"L_adv:          Initial={loss_history['adv'][0]:.4f}, Final={loss_history['adv'][-1]:.4f}")
        print(f"L_reg:          Initial={loss_history['reg'][0]:.4f}, Final={loss_history['reg'][-1]:.4f}")
        print(f"Learning Rate:  Initial={loss_history['lr'][0]:.6f}, Final={loss_history['lr'][-1]:.6f}")
        print(f"LR Scheduler:   cosine annealing")
        

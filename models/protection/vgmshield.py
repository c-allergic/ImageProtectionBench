import math
import os
from pathlib import Path
from typing import Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from omegaconf import OmegaConf
from .base import ProtectionBase

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    print("Warning: lpips not found. Please install it for VGMShield.")
    LPIPS_AVAILABLE = False

try:
    from sgm.util import default, instantiate_from_config
    SGM_AVAILABLE = True
except ImportError:
    print("Warning: sgm module not found. VGMShield may not work properly.")
    SGM_AVAILABLE = False


def get_loss(img_features, tar_features, img_v_features, tar_v_features, img, ori, lpips):
    """
    计算有目标攻击的损失函数
    
    Args:
        img_features: 当前图像特征
        tar_features: 目标图像特征
        img_v_features: 当前图像+噪声特征
        tar_v_features: 目标图像+噪声特征
        img: 当前图像
        ori: 原始图像
        lpips: LPIPS损失函数
        
    Returns:
        total_loss: 总损失
        l1: LPIPS感知损失
        l2: 特征相似度损失
    """
    alpha_1 = 1
    alpha_2 = 1
    alpha_3 = 1
    
    img_features = img_features.view(-1)
    tar_features = tar_features.view(-1)
    tar_v_features = tar_v_features.view(-1)
    img_v_features = img_v_features.view(-1)
    
    # LPIPS感知损失
    l1 = torch.abs(lpips(img, ori))
    
    # 特征相似度损失
    l2 = 1 - F.cosine_similarity(img_features.unsqueeze(0), tar_features.unsqueeze(0), dim=1)
    
    # 噪声特征相似度损失
    l3 = 1 - F.cosine_similarity(img_v_features.unsqueeze(0), tar_v_features.unsqueeze(0), dim=1)
    
    return l1*alpha_1 + l2*alpha_2 + l3*alpha_3, l1, l2


def get_loss_untarget(img_features, org_features, img_v_features, org_v_features, img, ori, lpips):
    """
    计算无目标攻击的损失函数
    
    Args:
        img_features: 当前图像特征
        org_features: 原始图像特征
        img_v_features: 当前图像+噪声特征
        org_v_features: 原始图像+噪声特征
        img: 当前图像
        ori: 原始图像
        lpips: LPIPS损失函数
        
    Returns:
        total_loss: 总损失
        l1: LPIPS感知损失
        l2: 特征相似度损失
    """
    alpha_1 = 1
    alpha_2 = 1
    alpha_3 = 1
    
    img_features = img_features.view(-1)
    org_features = org_features.view(-1)
    org_v_features = org_v_features.view(-1)
    img_v_features = img_v_features.view(-1)
    
    # LPIPS感知损失
    l1 = torch.abs(lpips(img, ori))
    
    # 特征差异损失（远离原始特征）
    l2 = 1 - F.cosine_similarity(img_features.unsqueeze(0), org_features.unsqueeze(0), dim=1)
    
    # 噪声特征差异损失
    l3 = 1 - F.cosine_similarity(img_v_features.unsqueeze(0), org_v_features.unsqueeze(0), dim=1)
    
    return l1*alpha_1 + l2*alpha_2 + l3*alpha_3, l1, l2


class VGMShield(ProtectionBase):
    """
    VGMShield: 基于SVD (Stable Video Diffusion) 的对抗性攻击保护方法
    
    该方法通过PGD (Projected Gradient Descent) 攻击生成对抗样本，
    使图像在SVD模型的特征空间中与目标图像相似（有目标攻击）
    或远离原始图像（无目标攻击），从而防止AI模型滥用。
    
    参考文献: 基于misuse_prevention脚本实现
    """
    
    def __init__(self,
                 eps: float = 4/255,
                 steps: int = 1000,
                 directed: bool = False,
                 alpha: float = 1/255,
                 noise_std: float = 0.02,
                 num_frames: int = 14,
                 num_steps: int = 25,
                 seed: int = 23,
                 model_config: Optional[str] = None,
                 target_image_path: Optional[str] = None,
                 **kwargs):
        """
        初始化VGMShield保护算法
        
        Args:
            eps: 扰动强度，默认4/255
            steps: 迭代步数，默认1000
            directed: 是否为有目标攻击，默认True
            alpha: 步长，默认1/255
            noise_std: 噪声标准差，默认0.02
            num_frames: 视频帧数，默认14
            num_steps: 采样步数，默认25
            seed: 随机种子，默认23
            model_config: SVD模型配置文件路径
            target_image_path: 目标图像路径（有目标攻击时需要）
            **kwargs: 其他参数
        """
        # 检查依赖
        if not LPIPS_AVAILABLE:
            raise ImportError("lpips is required for VGMShield. Please install: pip install lpips")
        if not SGM_AVAILABLE:
            raise ImportError("sgm module is required for VGMShield. Please install: pip install git+https://github.com/Stability-AI/generative-models.git")
        
        # 存储参数
        self.eps = eps
        self.steps = steps
        self.directed = directed
        self.alpha = alpha
        self.noise_std = noise_std
        self.num_frames = num_frames
        self.num_steps = num_steps
        self.seed = seed
        
        # 设置默认路径
        if model_config is None:
            model_config = "models/protection/vgm_config.yaml"
        self.model_config = model_config
        
        if target_image_path is None:
            target_image_path = "MIST.png"  # 使用默认目标图像
        self.target_image_path = target_image_path
        
        super().__init__(**kwargs)
    
    def _setup_model(self):
        """初始化SVD模型和相关组件"""
        # 设置随机种子
        torch.manual_seed(self.seed)
        
        # 加载模型
        self.model, self.filter = self._load_model(
            self.model_config,
            self.device,
            self.num_frames,
            self.num_steps
        )
        
        # 初始化LPIPS损失函数
        self.lpips = lpips.LPIPS(net='vgg').to(self.device)
        self.lpips.eval()
        
        # 加载目标图像（如果有目标攻击）
        if self.directed:
            self.target_image = self._load_target_image()
        
        print(f"VGMShield initialized with {'directed' if self.directed else 'untargeted'} attack")
    
    def _load_model(self, config_path: str, device: str, num_frames: int, num_steps: int):
        """
        加载SVD模型
        
        Args:
            config_path: 配置文件路径
            device: 计算设备
            num_frames: 视频帧数
            num_steps: 采样步数
            
        Returns:
            model: SVD模型
            filter: 数据过滤器
        """
        # 加载配置文件
        config = OmegaConf.load(config_path)
        
        # 更新动态参数
        config.model.params.sampler_config.params.num_steps = num_steps
        config.model.params.sampler_config.params.guider_config.params.num_frames = num_frames
        
        # 设置设备相关参数（按照原始实现）
        if device == "cuda:0":
            config.model.params.conditioner_config.params.emb_models[
                0
            ].params.open_clip_embedding_config.params.init_device = device
        
        # 实例化模型
        if device == "cuda:0":
            with torch.device(device):
                model = instantiate_from_config(config.model).to(device).eval()
        else:
            model = instantiate_from_config(config.model).to(device).eval()
        
        # 创建过滤器
        filter = None  # DeepFloydDataFiltering(verbose=False, device=device)
        
        return model, filter
    
    def _load_target_image(self):
        """加载目标图像"""
        target_img = Image.open(self.target_image_path).convert('RGB')
        target_tensor = transforms.ToTensor()(target_img)
        target_tensor = target_tensor * 2.0 - 1.0  # 归一化到[-1,1]
        return target_tensor.unsqueeze(0).to(self.device)  # 添加批次维度
    
    def protect(self, image: torch.Tensor, 
                target_image: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        保护单张图像
        
        Args:
            image: 输入图像张量 [C, H, W]，范围[0,1]
            target_image: 目标图像张量 [C, H, W]，范围[0,1]（有目标攻击时使用）
            **kwargs: 其他参数
            
        Returns:
            受保护的图像张量 [C, H, W]，范围[0,1]
        """
        # 图像预处理
        if image.shape[-1] % 64 != 0 or image.shape[-2] % 64 != 0:
            # 调整图像尺寸为64的倍数
            h, w = image.shape[-2:]
            new_h = h - h % 64
            new_w = w - w % 64
            image = transforms.Resize((new_h, new_w))(image)
            print(f"Resized image to {new_h}x{new_w} to be divisible by 64")
        
        # 归一化到[-1,1]并添加批次维度（按照原始实现）
        image = image * 2.0 - 1.0
        image = image.unsqueeze(0).to(self.device)
        
        # 准备目标图像
        if self.directed:
            if target_image is not None:
                tar_img = target_image * 2.0 - 1.0
                tar_img = tar_img.unsqueeze(0).to(self.device)
            else:
                tar_img = self.target_image
        else:
            tar_img = None
        
        # 添加噪声
        tar_img_v = tar_img + self.noise_std * torch.randn_like(tar_img) if tar_img is not None else None
        image_v = image + self.noise_std * torch.randn_like(image)
        
        # 设置图像为可训练
        image = image.clone().detach().requires_grad_(True)
        org_img = image.clone().detach().to(self.device)
        
        # 提取原始特征
        org_features = self.model.conditioner.embedders[0](org_img)
        org_img_v = org_img + self.noise_std * torch.randn_like(org_img)
        org_v_features = self.model.conditioner.embedders[3](org_img_v)
        
        # 提取目标特征（如果有目标攻击）
        if self.directed and tar_img is not None:
            tar_features = self.model.conditioner.embedders[0](tar_img)
            tar_v_features = self.model.conditioner.embedders[3](tar_img_v)
        
        # 执行对抗攻击
        if self.directed:
            # 有目标攻击
            for step in range(self.steps):
                image.requires_grad_(True)
                
                # 计算当前特征
                img_features = self.model.conditioner.embedders[0](image)
                image_v = image + self.noise_std * torch.randn_like(image)
                img_v_features = self.model.conditioner.embedders[3](image_v)
                
                # 计算损失
                loss, l1, l2 = get_loss(img_features, tar_features, img_v_features, tar_v_features, 
                                      image, org_img, self.lpips)
                
                # 计算梯度
                grad = torch.autograd.grad(loss, image, retain_graph=False, create_graph=False)[0]
                
                # PGD更新
                image = image - self.alpha * grad.sign()
                eta = torch.clamp(image - org_img, min=-self.eps, max=self.eps)
                image = torch.clamp(org_img + eta, min=-1, max=1).detach_()
                
                # 打印进度
                if step % 50 == 0:
                    print(f"steps\t{step}\tloss:{loss.item()}\tl1:{l1.item()}\tl2:{l2.item()}\t")
        else:
            # 无目标攻击
            for step in range(self.steps):
                image.requires_grad_(True)
                
                # 计算当前特征
                img_features = self.model.conditioner.embedders[0](image)
                image_v = image + self.noise_std * torch.randn_like(image)
                img_v_features = self.model.conditioner.embedders[3](image_v)
                
                # 计算损失
                loss, l1, l2 = get_loss_untarget(img_features, org_features, img_v_features, 
                                               org_v_features, image, org_img, self.lpips)
                
                # 计算梯度
                grad = torch.autograd.grad(loss, image, retain_graph=False, create_graph=False)[0]
                
                # PGD更新
                image = image + self.alpha * grad.sign()
                eta = torch.clamp(image - org_img, min=-self.eps, max=self.eps)
                image = torch.clamp(org_img + eta, min=-1, max=1).detach_()
                
                # 打印进度
                if step % 100 == 0:
                    print(f"Step {step}/{self.steps}, Loss: {loss.item():.4f}, L1: {l1.item():.4f}, L2: {l2.item():.4f}")
        
        # 后处理：反归一化到[0,1]
        image = (image + 1.0) / 2.0
        image = image.clamp(0.0, 1.0)
        
        return image.squeeze(0)  # 移除批次维度，返回[C, H, W]
    


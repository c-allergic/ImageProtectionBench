import os
import numpy as np
from omegaconf import OmegaConf
import PIL
from PIL import Image
from einops import rearrange
import ssl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from typing import Dict, Any, Optional, List, Union
from .base import ProtectionBase, timeit

# 禁用SSL验证
ssl._create_default_https_context = ssl._create_unverified_context

try:
    from ldm.util import instantiate_from_config
    LDM_AVAILABLE = True
except ImportError:
    print("Warning: ldm module not found. MIST protection may not work properly.")
    instantiate_from_config = None
    LDM_AVAILABLE = False


class IdentityLoss(nn.Module):
    """身份损失函数，用于MIST PGD攻击"""
    
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x


class TargetModel(nn.Module):
    """
    MIST目标模型 - 计算语义和纹理损失的虚拟模型
    这是MIST算法的核心组件，基于Stable Diffusion模型计算损失
    """

    def __init__(self, model, condition: str = "a painting", 
                 target_info: torch.Tensor = None, mode: int = 2, 
                 rate: int = 10000, input_size: int = 512):
        """
        Args:
            model: Stable Diffusion模型
            condition: 语义损失的文本条件
            target_info: 纹理损失的目标图像
            mode: 损失计算模式 (0:语义, 1:纹理, 2:融合)
            rate: 融合权重，越大越强调语义损失
            input_size: 输入图像尺寸
        """
        super().__init__()
        self.model = model
        self.condition = condition
        self.fn = nn.MSELoss(reduction="sum")
        self.target_info = target_info
        self.mode = mode
        self.rate = rate
        self.target_size = input_size

    def get_components(self, x, no_loss=False):
        """
        计算语义损失和编码信息 - MIST算法的核心
        """
        if self.model is None:
            # 回退模式
            z = torch.randn_like(x)
            loss = torch.tensor(0.0, device=x.device)
            return z, loss
        
        # 确保输入在正确的设备上
        x = x.to(next(self.model.parameters()).device)
            
        # 使用Stable Diffusion的VAE编码图像
        z = self.model.get_first_stage_encoding(self.model.encode_first_stage(x))
        
        # 获取文本条件的编码
        if isinstance(self.condition, str):
            c = self.model.get_learned_conditioning([self.condition] * x.shape[0])
        else:
            c = self.model.get_learned_conditioning(self.condition)
            
        if no_loss:
            loss = 0
        else:
            # 计算扩散模型的损失
            loss = self.model(z, c)[0]
            
        return z, loss

    def pre_process(self, x, target_size):
        """预处理图像到目标尺寸"""
        processed_x = torch.zeros([x.shape[0], x.shape[1], target_size, target_size], 
                                 device=x.device)
        trans = transforms.RandomCrop(target_size)
        for p in range(x.shape[0]):
            processed_x[p] = trans(x[p])
        return processed_x

    def forward(self, x, components=False):
        """
        MIST前向传播 - 计算组合损失
        """
        # 计算纹理损失
        zx, loss_semantic = self.get_components(x, True)
        
        if self.target_info is not None:
            zy, _ = self.get_components(self.target_info, True)
            textural_loss = self.fn(zx, zy)
        else:
            textural_loss = torch.tensor(0.0, device=x.device)
            
        # 计算语义损失
        if self.mode != 1:
            _, loss_semantic = self.get_components(self.pre_process(x, self.target_size))
            
        if components:
            return textural_loss, loss_semantic
            
        # 返回组合损失
        if self.mode == 0:
            return -loss_semantic
        elif self.mode == 1:
            return textural_loss
        else:
            return textural_loss - loss_semantic * self.rate


class LinfPGDAttack:
    """
    L-infinity PGD攻击 - MIST使用的对抗攻击方法
    """
    
    def __init__(self, model, loss_fn, epsilon, num_steps, eps_iter, 
                 clip_min=-1.0, clip_max=1.0, targeted=True):
        self.model = model
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.eps_iter = eps_iter
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted

    def perturb(self, x, y, mask=None):
        """执行PGD攻击生成对抗扰动"""
        x = x.clone().detach()
        
        # 随机初始化扰动
        delta = torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        delta = torch.clamp(delta, self.clip_min - x, self.clip_max - x)
        delta.requires_grad_(True)
        
        for i in range(self.num_steps):
            # 清理梯度
            if delta.grad is not None:
                delta.grad.zero_()
                
            adv_x = x + delta
            
            # 前向传播计算损失
            try:
                # 确保adv_x需要梯度
                adv_x.requires_grad_(True)
                model_output = self.model(adv_x)
                loss = self.loss_fn(model_output, y)
                
                # 确保loss是标量且需要梯度
                if loss.dim() > 0:
                    loss = loss.mean()
                    
            except Exception as e:
                # 如果模型调用失败，使用简化的损失
                print(f"Warning: Model forward failed: {e}")
                adv_x.requires_grad_(True)
                loss = torch.mean(torch.abs(adv_x - x))
            
            if not self.targeted:
                loss = -loss
                
            # 反向传播
            loss.backward(retain_graph=False)
            
            # 更新扰动
            if delta.grad is not None:
                grad = delta.grad.detach()
                # 分离delta并重新计算
                delta_data = delta.detach() + self.eps_iter * grad.sign()
                delta_data = torch.clamp(delta_data, -self.epsilon, self.epsilon)
                delta_data = torch.clamp(delta_data, self.clip_min - x, self.clip_max - x)
                
                # 应用mask（如果提供）
                if mask is not None:
                    delta_data = delta_data * mask
                
                # 创建新的delta tensor
                delta = delta_data.clone().detach().requires_grad_(True)
        
        return x + delta.detach()


class Mist(ProtectionBase):
    """
    MIST: 基于扩散模型的轻量级图像保护方法
    
    这是基于论文"MIST: Towards Improved Adversarial Examples for Diffusion Models"的标准实现
    使用Stable Diffusion模型计算语义和纹理损失，通过PGD攻击生成对抗扰动
    """
    
    def __init__(self, 
                 epsilon: int = 16,
                 steps: int = 100, 
                 alpha: int = 1,
                 input_size: int = 512,
                 mode: int = 2,
                 rate: int = 10000,
                 config_path: Optional[str] = None,
                 ckpt_path: Optional[str] = None,
                 target_image_path: Optional[str] = None,
                 **kwargs):
        """
        Args:
            epsilon: 扰动强度 (L∞范数，0-255范围)
            steps: 攻击迭代步数
            alpha: 每步的攻击强度
            input_size: 输入图像尺寸
            mode: 损失计算模式 (0:语义, 1:纹理, 2:融合)
            rate: 融合权重，越大越强调语义损失
            config_path: SD模型配置文件路径
            ckpt_path: SD模型权重文件路径
            target_image_path: 目标图像路径
        """
        self.epsilon = epsilon / 255.0 * 2  # 转换到[-1,1]范围
        self.alpha = alpha / 255.0 * 2
        self.steps = steps
        self.input_size = input_size
        self.mode = mode
        self.rate = rate
        
        # 设置默认路径
        if config_path is None:
            self.config_path = "configs/stable-diffusion/v1-inference-attack.yaml"
        else:
            self.config_path = config_path
            
        if ckpt_path is None:
            self.ckpt_path = "models/ldm/stable-diffusion-v1/model.ckpt"
        else:
            self.ckpt_path = ckpt_path
            
        if target_image_path is None:
            self.target_image_path = "MIST.png"
        else:
            self.target_image_path = target_image_path
        
        super().__init__(**kwargs)
    
    def _setup_model(self):
        """设置和加载MIST所需的Stable Diffusion模型"""
        if not LDM_AVAILABLE:
            print("❌ ldm模块不可用，将使用简化模式")
            self.sd_model = None
            self.loss_fn = IdentityLoss()
            self.target_image = self._create_default_target()
            return
            
        try:
            # 设置种子确保可重现性
            seed_everything(23)
            
            # 检查配置文件和模型权重是否存在
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
            if not os.path.exists(self.ckpt_path):
                raise FileNotFoundError(f"模型权重文件不存在: {self.ckpt_path}")
            
            print(f"✅ 加载MIST配置: {self.config_path}")
            print(f"✅ 加载MIST模型: {self.ckpt_path}")
            
            # 加载配置
            config = OmegaConf.load(self.config_path)
            
            # 加载Stable Diffusion模型
            self.sd_model = self._load_model_from_config(config, self.ckpt_path)
            
            # 设置损失函数
            self.loss_fn = IdentityLoss()
            
            # 加载目标图像
            self.target_image = self._load_target_image()
            
            print(f"✅ MIST模型初始化完成")
            
        except Exception as e:
            print(f"❌ MIST标准模型加载失败: {e}")
            print("将使用简化模式")
            self.sd_model = None
            self.loss_fn = IdentityLoss()
            self.target_image = self._create_default_target()
    
    def _load_model_from_config(self, config, ckpt_path):
        """从配置和权重文件加载Stable Diffusion模型"""
        print(f"正在加载Stable Diffusion模型: {ckpt_path}")
        
        # 兼容PyTorch 2.6+的weights_only限制
        try:
            pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            # 向后兼容旧版本PyTorch
            pl_sd = torch.load(ckpt_path, map_location="cpu")
            
        if "global_step" in pl_sd:
            print(f"模型训练步数: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        
        # 支持NovelAI权重格式
        if "state_dict" in sd:
            import copy
            sd_copy = copy.deepcopy(sd)
            for key in sd.keys():
                if key.startswith('cond_stage_model.transformer') and not key.startswith('cond_stage_model.transformer.text_model'):
                    newkey = key.replace('cond_stage_model.transformer', 'cond_stage_model.transformer.text_model', 1)
                    sd_copy[newkey] = sd[key]
                    del sd_copy[key]
            sd = sd_copy
        
        # 实例化模型
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        
        if len(m) > 0:
            print(f"缺失的权重keys: {len(m)}")
        if len(u) > 0:
            print(f"未使用的权重keys: {len(u)}")
        
        # 确保所有模型组件都在正确的设备上
        model.to(self.device)
        model.eval()
        
        # 强制将所有子模块移动到指定设备
        for module in model.modules():
            if hasattr(module, 'device'):
                module.to(self.device)
        
        return model
    
    def _load_target_image(self):
        """加载MIST目标图像"""
        try:
            if os.path.exists(self.target_image_path):
                target_img = Image.open(self.target_image_path).convert('RGB')
                target_img = target_img.resize((self.input_size, self.input_size))
                
                # 转换为tensor
                target_array = np.array(target_img).astype(np.float32) / 127.5 - 1.0
                target_tensor = torch.from_numpy(target_array).permute(2, 0, 1)
                
                print(f"✅ 成功加载目标图像: {self.target_image_path}")
                return target_tensor.to(self.device)
            else:
                print(f"⚠️  目标图像不存在: {self.target_image_path}，使用默认图像")
                return self._create_default_target()
        except Exception as e:
            print(f"⚠️  加载目标图像失败: {e}，使用默认图像")
            return self._create_default_target()
    
    def _create_default_target(self):
        """创建默认的MIST目标图像"""
        # 创建MIST特征图案
        target = torch.randn(3, self.input_size, self.input_size) * 0.1
        # 添加一些结构化的噪声模式
        x = torch.linspace(-1, 1, self.input_size)
        y = torch.linspace(-1, 1, self.input_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        pattern = torch.sin(xx * 10) * torch.cos(yy * 10) * 0.1
        target[0] += pattern
        target[1] += pattern * 0.5
        target[2] += pattern * 0.3
        return target.to(self.device)
    
    def protect(self, image: torch.Tensor, 
                prompt: str = "a painting", 
                target_image: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        使用MIST算法保护单张图片
        
        Args:
            image: 图片张量 [C, H, W]，范围[0,1]
            prompt: 用于语义损失的文本提示
            target_image: 目标图像（用于纹理损失）
            
        Returns:
            受保护的图片张量 [C, H, W]
        """
        # 转换图像到[-1,1]范围
        image_normalized = image * 2.0 - 1.0
        image_normalized = image_normalized.to(self.device)
        
        # 调整图像尺寸到模型输入尺寸
        if image_normalized.shape[-1] != self.input_size or image_normalized.shape[-2] != self.input_size:
            transform = transforms.Resize((self.input_size, self.input_size))
            image_normalized = transform(image_normalized)
        
        # 添加批次维度
        image_batch = image_normalized.unsqueeze(0)  # [1, C, H, W]
        
        # 设置目标图像
        if target_image is not None:
            target_batch = (target_image * 2.0 - 1.0).unsqueeze(0).to(self.device)
        else:
            target_batch = self.target_image.unsqueeze(0).to(self.device)
        
        # 创建MIST目标模型
        target_model = TargetModel(
            model=self.sd_model,
            condition=prompt,
            target_info=target_batch,
            mode=self.mode,
            rate=self.rate,
            input_size=self.input_size
        ).to(self.device)
        
        # 执行MIST PGD攻击
        attack = LinfPGDAttack(
            model=target_model,
            loss_fn=self.loss_fn,
            epsilon=self.epsilon,
            num_steps=self.steps,
            eps_iter=self.alpha,
            clip_min=-1.0,
            clip_max=1.0,
            targeted=True
        )
        
        # 执行攻击生成保护扰动
        label = torch.zeros_like(image_batch)
        protected_batch = attack.perturb(image_batch, label)
        
        # 移除批次维度并转换回[0,1]范围
        protected = protected_batch.squeeze(0)
        protected = torch.clamp((protected + 1.0) / 2.0, 0.0, 1.0)
        
        # 调整回原始尺寸
        if protected.shape[-1] != image.shape[-1] or protected.shape[-2] != image.shape[-2]:
            transform = transforms.Resize((image.shape[-2], image.shape[-1]))
            protected = transform(protected)
        
        return protected
    
    def set_target_image(self, target_image: torch.Tensor):
        """设置目标图像"""
        self.target_image = target_image.to(self.device)
    
    def set_mode(self, mode: int):
        """设置损失计算模式 (0:语义, 1:纹理, 2:融合)"""
        self.mode = mode
    
    def set_rate(self, rate: int):
        """设置融合权重"""
        self.rate = rate
    
    def set_protection_strength(self, epsilon: int):
        """设置保护强度"""
        self.epsilon = epsilon / 255.0 * 2
    
    @timeit
    def protect_multiple(
        self, 
        images: Union[torch.Tensor, List[torch.Tensor]], 
        batch_size: int = 4,
        prompt: str = "a painting",
        target_image: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        批量保护图片 - MIST标准实现
        
        Args:
            images: 图片张量 [B, C, H, W] 或图片张量列表
            batch_size: 批处理大小
            prompt: 用于语义损失的文本提示
            target_image: 目标图像（用于纹理损失）
            
        Returns:
            受保护的图片张量 [B, C, H, W]
        """
        # 处理输入格式
        if isinstance(images, list):
            images = torch.stack(images)
        
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        
        images = images.to(self.device)
        
        # 如果有SD模型，使用批量优化处理
        if self.sd_model is not None:
            return self._protect_batch_with_sd(
                images, batch_size, prompt, target_image, **kwargs
            )
        else:
            # 回退到逐张处理
            return super().protect_multiple(
                images, prompt=prompt, 
                target_image=target_image, **kwargs
            )
    
    def _protect_batch_with_sd(
        self, 
        images: torch.Tensor, 
        batch_size: int,
        prompt: str,
        target_image: Optional[torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        使用Stable Diffusion模型的批量保护实现
        """
        protected_images = []
        total_images = images.size(0)
        
        for i in range(0, total_images, batch_size):
            end_idx = min(i + batch_size, total_images)
            batch = images[i:end_idx]
            
            batch_protected = []
            for j in range(batch.size(0)):
                protected = self.protect(
                    batch[j], prompt=prompt, target_image=target_image, **kwargs
                )
                batch_protected.append(protected)
            
            protected_images.extend(batch_protected)
        
        return torch.stack(protected_images)


 
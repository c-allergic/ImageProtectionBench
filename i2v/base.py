from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
from PIL import Image
import os
import sys
import time
from contextlib import nullcontext
from diffusers.utils import export_to_video
from data.dataloader import pt_to_pil



class I2VModelBase(ABC):
    """图像到视频模型基类 - 简化版本"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        **kwargs
    ):
        """
        Args:
            model_path: 模型路径或HuggingFace模型ID
            device: 计算设备
            **kwargs: 其他配置参数
        """
        self.model_path = model_path
        self.device = device
        
        # 基础配置
        self.frame_rate = kwargs.get('frame_rate', 12)
        self.num_frames = kwargs.get('num_frames', 60)
        self.height = kwargs.get('height', 480)
        self.width = kwargs.get('width', 832)
        
        # 生成参数
        self.prompt = kwargs.get('prompt', '')
        self.negative_prompt = kwargs.get('negative_prompt', '')
        self.guidance_scale = kwargs.get('guidance_scale', 7.0)
        self.num_inference_steps = kwargs.get('num_inference_steps', 40)
        self.seed = kwargs.get('seed', 42)
        
        # 保存其他配置参数
        self.config = kwargs
        self.pipeline = None
        self.is_loaded = False
        
        self._setup_model()
    
    @abstractmethod
    def _setup_model(self):
        """设置和加载模型pipeline"""
        pass
    
    @abstractmethod
    def generate_video(
        self,
        images: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        从图像张量生成视频张量
        
        Args:
            images: 输入图像张量 [B, C, H, W]，范围[0,1]
            **kwargs: 可覆盖初始化时设置的参数
            
        Returns:
            生成的视频张量 [B, T, C, H, W]，范围[0,1]
        """
        pass

class SkyreelModel(I2VModelBase):
    """SkyReels模型实现，直接使用本地SkyReels-V2官方代码"""
    
    def __init__(self, **kwargs):
        # SkyReels模型特定配置
        model_path = kwargs.get('model_path')
        if model_path is None or model_path == "":
            # 使用本地可用的模型
            self.model_id = "Skywork/SkyReels-V2-I2V-1.3B-540P"
        else:
            self.model_id = model_path
            
        # 根据模型ID自动设置分辨率
        if "720P" in self.model_id:
            self.height, self.width = 720, 1280
        else:  # 540P
            self.height, self.width = 544, 960
            
        # 设置SkyReels模型默认参数
        kwargs.setdefault('frame_rate', 14)
        kwargs.setdefault('num_frames', 28)
        kwargs.setdefault('height', self.height)
        kwargs.setdefault('width', self.width)
        kwargs.setdefault('guidance_scale', 6.0)
        kwargs.setdefault('num_inference_steps', 30)
        kwargs.setdefault('shift', 8.0)
        kwargs.setdefault('prompt', '')
        kwargs.setdefault('seed', 42)
        
        # SkyReels特有参数
        self.use_usp = kwargs.get('use_usp', False)
        self.offload = kwargs.get('offload', True)
        self.prompt_enhancer_enabled = kwargs.get('prompt_enhancer', False)
        
        # 默认负面提示词
        self.default_negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        
        super().__init__(**kwargs)
    
    def _setup_model(self):
        """设置SkyReels模型pipeline - 使用本地SkyReels-V2实现"""
        import sys
        import os
        
        # 添加SkyReels-V2路径到Python路径
        skyreels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'SkyReels-V2')
        absolute_skyreels_path = os.path.abspath(skyreels_path)
        if not os.path.exists(absolute_skyreels_path):
            print(f"❌ SkyReels-V2路径不存在: {absolute_skyreels_path}")
            self.is_loaded = False
            return
            
        sys.path.insert(0, absolute_skyreels_path)
        print(f"✅ SkyReels-V2路径已添加: {absolute_skyreels_path}")
        
        print(f"正在加载SkyReels模型: {self.model_id}")
        
        # 导入SkyReels-V2模块
        from skyreels_v2_infer.modules import download_model
        from skyreels_v2_infer.pipelines import Image2VideoPipeline
        
        # 下载并获取模型路径
        model_path = download_model(self.model_id)
        print(f"✅ 模型路径: {model_path}")
        
        # 初始化pipeline
        self.pipeline = Image2VideoPipeline(
            model_path=model_path,
            dit_path=model_path,
            use_usp=self.use_usp,
            offload=self.offload
        )
        
        self.is_loaded = True
        print(f"✅ SkyReels Pipeline初始化成功")
    
    def generate_video(
        self,
        images: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """使用SkyReels pipeline生成视频"""
        
        if not self.is_loaded or self.pipeline is None:
            raise RuntimeError("SkyReels模型未正确加载")
            
        return self._generate_with_pipeline(images, **kwargs)
    
    def _generate_with_pipeline(self, images: torch.Tensor, **kwargs) -> torch.Tensor:
        """使用SkyReels-V2 pipeline生成视频"""
        prompt = kwargs.get('prompt', self.prompt)
        negative_prompt = kwargs.get('negative_prompt', self.negative_prompt)
        guidance_scale = kwargs.get('guidance_scale', self.guidance_scale)
        num_inference_steps = kwargs.get('num_inference_steps', self.num_inference_steps)
        shift = kwargs.get('shift', self.config.get('shift', 8.0))
        seed = kwargs.get('seed', self.seed)
        num_frames = kwargs.get('num_frames', self.num_frames)
        
        batch_size = images.size(0)
        all_videos = []
        
        for b in range(batch_size):
            # 转换为PIL图像
            pil_image = pt_to_pil(images[b]).convert("RGB")
            
            # 使用SkyReels-V2官方参数生成视频
            generation_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": pil_image,
                "num_frames": num_frames,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "shift": shift,
                "generator": torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed + b),
                "height": self.height,
                "width": self.width,
            }
            
            print(f"正在生成视频 {b+1}/{batch_size}...")
            start_time = time.time()
            
            # 确保图像格式正确
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            generation_kwargs["image"] = pil_image
            
            # 生成视频
            with torch.no_grad():
                video_frames = self.pipeline(**generation_kwargs)[0]
                    
            print(f"✅ 视频 {b+1} 生成完成，用时: {time.time() - start_time:.1f}秒")
            
            # 转换回张量
            frames_tensor = []
            for frame in video_frames:
                if isinstance(frame, np.ndarray):
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                else:
                    # PIL Image
                    frame_array = np.array(frame)
                    frame_tensor = torch.from_numpy(frame_array).permute(2, 0, 1).float() / 255.0
                frames_tensor.append(frame_tensor)
            
            video_tensor = torch.stack(frames_tensor)
            all_videos.append(video_tensor)
            
            # Clear GPU cache after each video generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return torch.stack(all_videos).to(self.device)


class LTXModel(I2VModelBase):
    """LTX模型实现，基于videogen.py中的四步上采样流程"""
    
    def __init__(self, **kwargs):
        # LTX模型特定配置 - 基于成功的ltx.py实现
        model_path = kwargs.get('model_path')
        if model_path is None or model_path == "":
            self.model_id = "Lightricks/LTX-Video-0.9.8-13B-distilled"
            self.upsample_id = "Lightricks/ltxv-spatial-upscaler-0.9.7"
        else:
            self.model_id = model_path
            self.upsample_id = model_path.replace("LTX-Video", "ltxv-spatial-upscaler") if "LTX-Video" in model_path else model_path
            
        # 设置LTX模型默认参数 - 基于成功的ltx.py
        kwargs.setdefault('frame_rate', 14)
        kwargs.setdefault('num_frames', 28) 
        kwargs.setdefault('height', 480)
        kwargs.setdefault('width', 832)
        kwargs.setdefault('guidance_scale', 7.0)
        kwargs.setdefault('num_inference_steps', 30)  # 与ltx.py保持一致
        kwargs.setdefault('prompt', '')
        kwargs.setdefault('negative_prompt', 'worst quality, inconsistent motion, blurry, jittery, distorted')
        kwargs.setdefault('seed', 0)  
        # LTX特有的高级参数 - 基于成功的ltx.py
        self.downscale_factor = kwargs.get('downscale_factor', 2/3)  # 与ltx.py保持一致
        self.denoise_steps = kwargs.get('denoise_steps', 10)  # 与ltx.py保持一致
        self.denoise_strength = kwargs.get('denoise_strength', 0.4)  # 与ltx.py保持一致
        self.decode_timestep = kwargs.get('decode_timestep', 0.05)  # 与ltx.py保持一致
        self.image_cond_noise_scale = kwargs.get('image_cond_noise_scale', 0.025)  # 与ltx.py保持一致
        
        # 初始化两个pipeline - 必须在super().__init__之前设置
        self.pipeline = None
        self.pipeline_upsample = None
        
        super().__init__(**kwargs)
    
    def _setup_model(self):
        """加载LTX模型pipeline - 基于成功的ltx.py实现"""
        print(f"加载LTX模型: {self.model_id}")
            
        # 导入LTX相关模块
        from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
        from diffusers.utils import export_to_video, load_image, load_video
        
        print("✅ LTX依赖导入成功")
        
        # 加载主管道 - 与ltx.py保持一致
        print("正在加载LTX主管道...")
        print(f"模型ID: {self.model_id}")
        self.pipeline = LTXConditionPipeline.from_pretrained(
            self.model_id, 
            torch_dtype=torch.bfloat16
        )
        print(f"✅ LTX主管道加载成功，类型: {type(self.pipeline)}")
        
        # 加载上采样管道 - 与ltx.py保持一致
        print("正在加载LTX上采样管道...")
        print(f"上采样器ID: {self.upsample_id}")
        self.pipeline_upsample = LTXLatentUpsamplePipeline.from_pretrained(
            self.upsample_id, 
            vae=self.pipeline.vae, 
            torch_dtype=torch.bfloat16
        )
        print(f"✅ LTX上采样管道加载成功，类型: {type(self.pipeline_upsample)}")
        
        # 移动到设备
        print(f"正在将模型移动到设备: {self.device}")
        self.pipeline.to(self.device)
        self.pipeline_upsample.to(self.device)
        
        # 启用VAE tiling优化内存
        self.pipeline.vae.enable_tiling()
        
        # 保存辅助类和函数的引用
        self.LTXVideoCondition = LTXVideoCondition
        self.load_video = load_video
        self.export_to_video = export_to_video
        
        self.is_loaded = True
        print(f"✅ LTX模型完全加载成功")
        
    
    def _round_to_vae_acceptable(self, height: int, width: int) -> tuple:
        """将尺寸调整为VAE可接受的尺寸"""
        if self.pipeline is None or self.pipeline_upsample is None:
            raise RuntimeError("LTX模型未正确加载")
        height = height - (height % self.pipeline.vae_spatial_compression_ratio)
        width = width - (width % self.pipeline.vae_spatial_compression_ratio)
        return height, width
    
    def generate_video(
        self,
        images: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """使用LTX四步上采样流程生成视频"""
        
        if not self.is_loaded:
            raise RuntimeError("LTX模型未正确加载: is_loaded=False")
        if self.pipeline is None:
            raise RuntimeError("LTX模型未正确加载: pipeline=None")
        if self.pipeline_upsample is None:
            raise RuntimeError("LTX模型未正确加载: pipeline_upsample=None")
        
        prompt = kwargs.get('prompt', self.prompt)
        negative_prompt = kwargs.get('negative_prompt', self.negative_prompt)
        guidance_scale = kwargs.get('guidance_scale', self.guidance_scale)
        num_inference_steps = kwargs.get('num_inference_steps', self.num_inference_steps)
        num_frames = kwargs.get('num_frames', self.num_frames)
        
        batch_size = images.size(0)
        all_videos = []
        
        for b in range(batch_size):
            # 转换为PIL图像
            pil_image = pt_to_pil(images[b])
            
            # 生成视频（四步流程）
            video_frames = self._generate_ltx_video_four_steps(
                pil_image, prompt, negative_prompt, num_frames,
                guidance_scale, num_inference_steps
            )
            
            # 转换回张量
            frames_tensor = []
            for frame in video_frames:
                frame_array = np.array(frame)
                frame_tensor = torch.from_numpy(frame_array).permute(2, 0, 1).float() / 255.0
                frames_tensor.append(frame_tensor)
            
            video_tensor = torch.stack(frames_tensor)
            all_videos.append(video_tensor)
            
            # Clear GPU cache after each video generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return torch.stack(all_videos).to(self.device)
    
    def _generate_ltx_video_four_steps(
        self, 
        image: Image.Image, 
        prompt: str, 
        negative_prompt: str, 
        num_frames: int,
        guidance_scale: float,
        num_inference_steps: int
    ) -> List[Image.Image]:
        """LTX四步上采样生成流程 - 基于成功的ltx.py实现"""
        
        start_time = time.time()
        
        # 准备视频条件 - 与ltx.py保持一致
        video = self.load_video(self.export_to_video([image]))
        condition = self.LTXVideoCondition(video=video, frame_index=0)
        
        # Part 1. 在较小分辨率下生成视频 - 与ltx.py保持一致
        downscaled_height = int(self.height * self.downscale_factor)
        downscaled_width = int(self.width * self.downscale_factor)
        downscaled_height, downscaled_width = self._round_to_vae_acceptable(downscaled_height, downscaled_width)
        
        # 使用固定种子 - 与ltx.py保持一致
        generator = torch.Generator().manual_seed(0)
        
        print(f"Part 1: 生成 {downscaled_width}x{downscaled_height} 视频")
        latents = self.pipeline(
            conditions=[condition],
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=downscaled_width,
            height=downscaled_height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="latent",
        ).frames
        
        # Part 2. 使用潜在上采样器放大视频 - 与ltx.py保持一致
        upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
        print(f"Part 2: 上采样到 {upscaled_width}x{upscaled_height}")
        upscaled_latents = self.pipeline_upsample(
            latents=latents,
            output_type="latent"
        ).frames
        
        # Part 3. 对上采样后的视频进行少量步数的去噪以改善纹理 - 与ltx.py保持一致
        print(f"Part 3: 去噪优化，强度={self.denoise_strength}")
        video = self.pipeline(
            conditions=[condition],
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=upscaled_width,
            height=upscaled_height,
            num_frames=num_frames,
            denoise_strength=self.denoise_strength,
            num_inference_steps=self.denoise_steps,
            latents=upscaled_latents,
            decode_timestep=self.decode_timestep,
            image_cond_noise_scale=self.image_cond_noise_scale,
            generator=generator,
            output_type="pil",
        ).frames[0]
        
        # Part 4. 将视频缩放到期望的分辨率 - 与ltx.py保持一致
        print(f"Part 4: 缩放到最终分辨率 {self.width}x{self.height}")
        video = [frame.resize((self.width, self.height)) for frame in video]
        
        total_time = time.time() - start_time
        print(f"✅ LTX四步生成完成，用时: {total_time:.1f}秒，{len(video)}帧")
        
        return video


class WAN22Model(I2VModelBase):
    """Wan2.2 TI2V本地模型实现，使用本地Wan2.2代码"""
    
    def __init__(self, **kwargs):
        # Wan2.2模型特定配置
        model_path = kwargs.get('model_path')
        if model_path is None or model_path == "":
            self.checkpoint_dir = "/data_sde/lxf/Wan2.2/Wan2.2-TI2V-5B"
        else:
            self.checkpoint_dir = model_path
            
        # 设置Wan2.2模型默认参数
        kwargs.setdefault('frame_rate', 12)
        kwargs.setdefault('num_frames', 25)
        kwargs.setdefault('height', 704)
        kwargs.setdefault('width', 1280)
        kwargs.setdefault('guidance_scale', 5.0)
        kwargs.setdefault('num_inference_steps', 30)
        kwargs.setdefault('sample_shift', 5.0)
        kwargs.setdefault('prompt', '')
        kwargs.setdefault('seed', 42)
        
        # 保存高级参数
        self.sample_shift = kwargs.get('sample_shift', 5.0)
        self.max_area = kwargs.get('max_area', 704 * 1280)
        
        # 设备配置
        self.device_id = kwargs.get('device_id', 0)
        
        super().__init__(**kwargs)
    
    def _setup_model(self):
        """设置Wan2.2 TI2V模型pipeline"""
        # 添加Wan2.2路径到Python路径
        wan_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Wan2.2')
        absolute_wan_path = os.path.abspath(wan_path)
        
        if not os.path.exists(absolute_wan_path):
            print(f"错误: Wan2.2路径不存在: {absolute_wan_path}")
            self.is_loaded = False
            return
            
        if absolute_wan_path not in sys.path:
            sys.path.insert(0, absolute_wan_path)
        print(f"Wan2.2路径: {absolute_wan_path}")
        
        # 检查checkpoint目录
        if not os.path.exists(self.checkpoint_dir):
            print(f"错误: 模型权重目录不存在: {self.checkpoint_dir}")
            self.is_loaded = False
            return
        
        print(f"正在加载Wan2.2 TI2V模型: {self.checkpoint_dir}")
        
        # 导入Wan2.2模块
        import wan
        from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS
        
        # 获取配置
        task = "ti2v-5B"
        cfg = WAN_CONFIGS[task]
        
        # 创建WanTI2V pipeline
        self.pipeline = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=self.checkpoint_dir,
            device_id=self.device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            init_on_cpu=False,  # 将DiT模型加载到GPU
            convert_model_dtype=True,  # 转换模型dtype为bfloat16，确保bias也是bfloat16
        )
        
        # 保存配置信息
        self.SIZE_CONFIGS = SIZE_CONFIGS
        self.MAX_AREA_CONFIGS = MAX_AREA_CONFIGS
        
        self.is_loaded = True
        print(f"✅ Wan2.2 TI2V模型加载成功")
    
    def generate_video(
        self,
        images: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """使用Wan2.2 TI2V生成视频
        
        Args:
            images: 输入图像张量 [B, C, H, W]，范围[0,1]
            **kwargs: 可选参数
            
        Returns:
            生成的视频张量 [B, T, C, H, W]，范围[0,1]
        """
        
        if not self.is_loaded or self.pipeline is None:
            raise RuntimeError("Wan2.2模型未正确加载")
        
        prompt = kwargs.get('prompt', self.prompt)
        guidance_scale = kwargs.get('guidance_scale', self.guidance_scale)
        num_inference_steps = kwargs.get('num_inference_steps', self.num_inference_steps)
        num_frames = kwargs.get('num_frames', self.num_frames)
        sample_shift = kwargs.get('sample_shift', self.sample_shift)
        seed = kwargs.get('seed', self.seed)
        
        batch_size = images.size(0)
        all_videos = []
        
        for b in range(batch_size):
            # 转换为PIL图像
            pil_image = pt_to_pil(images[b])
            
            print(f"正在生成视频 {b+1}/{batch_size}...")
            start_time = time.time()
            
            # 准备尺寸配置
            size_key = f"{self.width}*{self.height}"
            if size_key not in self.SIZE_CONFIGS:
                size_key = "1280*704"
                print(f"警告: 尺寸 {self.width}*{self.height} 不支持，使用默认 {size_key}")
            
            # 使用Wan2.2 pipeline生成视频
            with torch.no_grad():
                video_tensor = self.pipeline.generate(
                    input_prompt=prompt,
                    img=pil_image,
                    size=self.SIZE_CONFIGS[size_key],
                    max_area=self.max_area,
                    frame_num=num_frames,
                    shift=sample_shift,
                    sample_solver='unipc',
                    sampling_steps=num_inference_steps,
                    guide_scale=guidance_scale,
                    seed=seed + b,
                    offload_model=True
                )
            
            # video_tensor 的形状是 [C, T, H, W]，需要转换为 [T, C, H, W]
            # 范围 [-1, 1]，需要转换到 [0, 1]
            video_tensor = video_tensor.permute(1, 0, 2, 3)  # [C, T, H, W] -> [T, C, H, W]
            video_tensor = (video_tensor + 1.0) / 2.0
            video_tensor = video_tensor.clamp(0, 1)
            
            all_videos.append(video_tensor)
            
            print(f"✅ 视频 {b+1} 生成完成，用时: {time.time() - start_time:.1f}秒，形状: {video_tensor.shape}")
            
            # Clear GPU cache after each video generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 返回 [B, T, C, H, W]
        return torch.stack(all_videos).to(self.device)
    
 
from abc import ABC, abstractmethod
import torch
from typing import Union, List
import time
import logging
from functools import wraps

# 配置日志记录器
logger = logging.getLogger(__name__)


def timeit(func):
    """
    时间记录装饰器，用于记录保护算法的处理时间
    
    使用方式:
        @timeit
        def protect_batch(self, ...):
            ...
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        
        # 获取图片数量 - 简化逻辑
        num_images = 1
        if args and hasattr(args[0], 'shape'):
            # 如果第一个参数是tensor，获取批次大小
            if len(args[0].shape) == 4:  # [B, C, H, W]
                num_images = args[0].shape[0]
            elif len(args[0].shape) == 3:  # [C, H, W]
                num_images = 1
        elif args and isinstance(args[0], (list, tuple)):
            # 如果第一个参数是列表
            num_images = len(args[0])
        
        # 执行函数
        result = func(self, *args, **kwargs)
        
        # 计算耗时
        elapsed_time = time.time() - start_time
        
        # 存储时间信息到对象属性中（用于TimeMetric）
        if not hasattr(self, '_timing_records'):
            self._timing_records = []
        
        self._timing_records.append({
            'method': func.__name__,
            'elapsed_time': elapsed_time,
            'num_images': num_images,
            'avg_time_per_image': elapsed_time / num_images if num_images > 0 else 0
        })
        
        # 记录日志
        logger.info(
            f"[{self.__class__.__name__}] {func.__name__}: "
            f"处理 {num_images} 张图片, "
            f"耗时 {elapsed_time:.2f}s, "
            f"平均 {elapsed_time/num_images:.3f}s/图片"
        )
        
        return result
    return wrapper


class ProtectionBase(ABC):
    """
    图像保护算法基类
    
    子类只需要实现protect方法来保护单张图片即可，
    基类提供批量处理功能。
    """
    
    def __init__(self, device: str = "cuda", **kwargs):
        """
        Args:
            device: 计算设备 ('cuda' 或 'cpu')
            **kwargs: 其他配置参数
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.config = kwargs
        self._setup_model()
        
        logger.info(f"初始化 {self.__class__.__name__} 保护算法，设备: {self.device}")
    
    @abstractmethod
    def _setup_model(self):
        """
        设置模型，子类必须实现
        在这里初始化所需的模型或参数
        """
        pass
    
    @abstractmethod
    def protect(self,image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        保护单张图片 - 核心接口，子类必须实现
        
        Args:
            image: 单张图片张量 [C, H, W]，范围[0,1]
            **kwargs: 其他参数
            
        Returns:
            受保护的图片张量 [C, H, W]，范围[0,1]
        """
        pass
    
    @timeit
    def protect_multiple(
        self, 
        images: Union[torch.Tensor, List[torch.Tensor]], 
        **kwargs
    ) -> torch.Tensor:
        """

        Args:
            images: 图片张量 [B, C, H, W] 或图片张量列表
            **kwargs: 传递给protect方法的其他参数    
        Returns:
            受保护的图片张量 [B, C, H, W]
        """
        # 处理输入格式
        if isinstance(images, list):
            images = torch.stack(images)
        
        if len(images.shape) == 3:
            # 单张图片，添加批次维度
            images = images.unsqueeze(0)
        
        images = images.to(self.device)
        protected_images = []
        total_images = images.size(0)
        
        for i in range(total_images):
            single_image = images[i]
            protected_single = self.protect(single_image, **kwargs)
            protected_images.append(protected_single)
        
        return torch.stack(protected_images)



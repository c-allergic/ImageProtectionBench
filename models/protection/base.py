from abc import ABC, abstractmethod
import torch
from typing import Union, List


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



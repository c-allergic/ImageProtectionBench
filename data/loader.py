import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, Any
import torchvision.transforms as transforms
from .dataset import ImageDataset


class ImageDataLoader:
    """图像数据加载器类"""
    
    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle: bool = False,
        transform: Optional[Callable] = None
    ):
        """
        Args:
            batch_size: 批大小
            num_workers: 工作进程数
            pin_memory: 是否固定内存
            shuffle: 是否随机打乱
            transform: 图像变换函数
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.transform = transform or self._get_default_transform()
    
    def _get_default_transform(self):
        """获取默认的图像变换"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def create_dataloader(self, dataset: ImageDataset) -> DataLoader:
        """创建数据加载器"""
        # 设置数据集的变换函数
        if self.transform:
            dataset.transform = self.transform
            
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """自定义批处理函数"""
        images = torch.stack([item['image'] for item in batch])
        labels = [item['label'] for item in batch]
        indices = torch.tensor([item['idx'] for item in batch])
        
        return {
            'images': images,
            'labels': labels,
            'indices': indices
        }


def get_protection_transform():
    """获取保护算法专用的图像变换"""
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        # 不进行标准化，保持原始像素值范围[0,1]用于保护算法
    ])


def get_i2v_transform():
    """获取I2V模型专用的图像变换"""
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        # I2V模型通常需要[-1,1]范围的输入
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_evaluation_transform():
    """获取评估专用的图像变换"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def create_dataloaders(
    datasets: Dict[str, ImageDataset],
    batch_size: int = 1,
    num_workers: int = 4,
    transform_type: str = "default"
) -> Dict[str, DataLoader]:
    """
    为多个数据集创建数据加载器
    
    Args:
        datasets: 数据集字典
        batch_size: 批大小
        num_workers: 工作进程数
        transform_type: 变换类型 ("default", "protection", "i2v", "evaluation")
    
    Returns:
        数据加载器字典
    """
    
    # 根据类型选择变换函数
    if transform_type == "protection":
        transform = get_protection_transform()
    elif transform_type == "i2v":
        transform = get_i2v_transform()
    elif transform_type == "evaluation":
        transform = get_evaluation_transform()
    else:
        transform = None
    
    loader = ImageDataLoader(
        batch_size=batch_size,
        num_workers=num_workers,
        transform=transform
    )
    
    dataloaders = {}
    for split, dataset in datasets.items():
        # 测试集不打乱顺序
        loader.shuffle = (split == "train")
        dataloaders[split] = loader.create_dataloader(dataset)
    
    return dataloaders 
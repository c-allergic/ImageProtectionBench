import os
import json
import random
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from datasets import load_dataset as hf_load_dataset

# 支持的数据集列表
DATASETS = ['LAION-Aesthetics', 'COCO', 'CelebA-HQ', 'FFHQ', 'ImageNet', 'Custom']


class ImageDataset(Dataset):
    """图像数据集基类"""
    
    def __init__(
        self, 
        images: List[str], 
        labels: Optional[List[str]] = None,
        transform=None,
        max_size: int = 512
    ):
        """
        Args:
            images: 图像路径列表
            labels: 图像标签/描述列表
            transform: 图像变换函数
            max_size: 图像最大尺寸
        """
        self.images = images
        self.labels = labels if labels is not None else [f"image_{i}" for i in range(len(images))]
        self.transform = transform
        self.max_size = max_size
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 加载图像
        if isinstance(self.images[idx], str):
            image = Image.open(self.images[idx]).convert('RGB')
        else:
            image = self.images[idx]
            
        # 调整图像尺寸
        image = self._resize_image(image)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        else:
            # 默认转换为tensor
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            
        return {
            'image': image,
            'label': self.labels[idx],
            'idx': idx
        }
    
    def _resize_image(self, image):
        """调整图像尺寸保持长宽比"""
        w, h = image.size
        if max(w, h) > self.max_size:
            if w > h:
                new_w = self.max_size
                new_h = int(h * self.max_size / w)
            else:
                new_h = self.max_size
                new_w = int(w * self.max_size / h)
            image = image.resize((new_w, new_h), Image.LANCZOS)
        return image


def load_laion_aesthetics(
    data_dir: str = "data/laion_aesthetics",
    num_samples: int = 1000,
    split: str = "test"
) -> Dict[str, ImageDataset]:
    """加载LAION-Aesthetics数据集"""
    
    # 如果本地没有数据，尝试从HuggingFace加载
    if not os.path.exists(data_dir):
        print(f"本地数据目录 {data_dir} 不存在，尝试从HuggingFace加载...")
        # 这里可以添加从HuggingFace下载数据的逻辑
        os.makedirs(data_dir, exist_ok=True)
        
        # 模拟数据创建，实际使用时应该从真实数据源加载
        images = []
        labels = []
        for i in range(num_samples):
            # 创建模拟图像路径
            img_path = os.path.join(data_dir, f"image_{i:06d}.jpg")
            images.append(img_path)
            labels.append(f"A beautiful aesthetic image {i}")
            
        # 保存元数据
        metadata = {
            'images': images,
            'labels': labels,
            'split': split
        }
        with open(os.path.join(data_dir, f"metadata_{split}.json"), 'w') as f:
            json.dump(metadata, f)
    else:
        # 从本地加载元数据
        metadata_path = os.path.join(data_dir, f"metadata_{split}.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            images = metadata['images'][:num_samples]
            labels = metadata['labels'][:num_samples]
        else:
            raise FileNotFoundError(f"元数据文件 {metadata_path} 不存在")
    
    dataset = ImageDataset(images, labels)
    return {split: dataset}


def load_coco(
    data_dir: str = "data/coco",
    num_samples: int = 1000,
    split: str = "val2017"
) -> Dict[str, ImageDataset]:
    """加载COCO数据集"""
    
    images_dir = os.path.join(data_dir, split)
    annotations_file = os.path.join(data_dir, "annotations", f"captions_{split}.json")
    
    if not os.path.exists(annotations_file):
        print(f"COCO注释文件 {annotations_file} 不存在")
        # 创建模拟数据
        os.makedirs(os.path.join(data_dir, "annotations"), exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        
        images = []
        labels = []
        for i in range(num_samples):
            img_path = os.path.join(images_dir, f"COCO_{split}_{i:012d}.jpg")
            images.append(img_path)
            labels.append(f"COCO image {i}")
    else:
        # 加载真实COCO数据
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        # 创建图像ID到标注的映射
        img_id_to_captions = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_id_to_captions:
                img_id_to_captions[img_id] = []
            img_id_to_captions[img_id].append(ann['caption'])
        
        images = []
        labels = []
        for img_info in data['images'][:num_samples]:
            img_path = os.path.join(images_dir, img_info['file_name'])
            if os.path.exists(img_path):
                images.append(img_path)
                # 使用第一个标注作为标签
                captions = img_id_to_captions.get(img_info['id'], ['Unknown'])
                labels.append(captions[0])
    
    dataset = ImageDataset(images, labels)
    return {split: dataset}


def load_celeba_hq(
    data_dir: str = "data/celeba_hq",
    num_samples: int = 1000,
    split: str = "test"
) -> Dict[str, ImageDataset]:
    """加载CelebA-HQ数据集"""
    
    images_dir = os.path.join(data_dir, "images")
    
    if not os.path.exists(images_dir):
        print(f"CelebA-HQ数据目录 {images_dir} 不存在")
        os.makedirs(images_dir, exist_ok=True)
        
        images = []
        labels = []
        for i in range(num_samples):
            img_path = os.path.join(images_dir, f"celeba_hq_{i:06d}.jpg")
            images.append(img_path)
            labels.append(f"Celebrity face {i}")
    else:
        # 获取所有图像文件
        img_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        img_files.sort()
        
        images = [os.path.join(images_dir, f) for f in img_files[:num_samples]]
        labels = [f"Celebrity face {i}" for i in range(len(images))]
    
    dataset = ImageDataset(images, labels)
    return {split: dataset}


def load_custom_dataset(
    data_dir: str,
    metadata_file: Optional[str] = None,
    num_samples: int = 1000,
    split: str = "test"
) -> Dict[str, ImageDataset]:
    """加载自定义数据集"""
    
    if metadata_file and os.path.exists(metadata_file):
        # 从元数据文件加载
        if metadata_file.endswith('.json'):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            images = metadata.get('images', [])
            labels = metadata.get('labels', [])
        elif metadata_file.endswith('.csv'):
            df = pd.read_csv(metadata_file)
            images = df['image_path'].tolist()
            labels = df.get('label', df.get('caption', [''] * len(images))).tolist()
        else:
            raise ValueError(f"不支持的元数据文件格式: {metadata_file}")
    else:
        # 扫描目录获取图像文件
        img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        images = []
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(img_extensions):
                    images.append(os.path.join(root, file))
        
        images.sort()
        labels = [f"Custom image {i}" for i in range(len(images))]
    
    # 限制样本数量
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    dataset = ImageDataset(images, labels)
    return {split: dataset}


def load_dataset(
    dataset_name: str,
    data_dir: Optional[str] = None,
    num_samples: int = 1000,
    split: str = "test",
    **kwargs
) -> Dict[str, ImageDataset]:
    """
    加载指定的数据集
    
    Args:
        dataset_name: 数据集名称
        data_dir: 数据目录
        num_samples: 样本数量
        split: 数据集分割
        **kwargs: 其他参数
    
    Returns:
        包含数据集的字典
    """
    
    if data_dir is None:
        data_dir = f"data/{dataset_name.lower().replace('-', '_')}"
    
    if dataset_name == "LAION-Aesthetics":
        return load_laion_aesthetics(data_dir, num_samples, split)
    elif dataset_name == "COCO":
        return load_coco(data_dir, num_samples, split)
    elif dataset_name == "CelebA-HQ":
        return load_celeba_hq(data_dir, num_samples, split)
    elif dataset_name == "Custom":
        metadata_file = kwargs.get('metadata_file', None)
        return load_custom_dataset(data_dir, metadata_file, num_samples, split)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")


def sample_dataset(dataset: ImageDataset, num_samples: int) -> ImageDataset:
    """随机采样数据集"""
    if len(dataset) <= num_samples:
        return dataset
    
    indices = random.sample(range(len(dataset)), num_samples)
    indices.sort()
    
    sampled_images = [dataset.images[i] for i in indices]
    sampled_labels = [dataset.labels[i] for i in indices]
    
    return ImageDataset(sampled_images, sampled_labels, dataset.transform, dataset.max_size) 
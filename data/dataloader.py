import os
from typing import List
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
from datasets import load_dataset as hf_load_dataset

# 支持的数据集列表
DATASETS = ['TIP-I2V', 'AFHQ-V2', 'Wikiart', 'LHQ', 'Flickr30k']

def pt_to_pil(tensor: torch.Tensor, video: bool = False):
    """将PyTorch张量转换为PIL图像"""
    if video:   
        video_frames = []
        for t in range(tensor.size(0)):
            frame = tensor[t].cpu()
            frame_array = (frame.permute(1, 2, 0) * 255).clamp(0, 255).numpy().astype(np.uint8)
            video_frames.append(Image.fromarray(frame_array))
        return video_frames
    else:
        image = tensor.cpu()
        return Image.fromarray((image.permute(1, 2, 0) * 255).clamp(0, 255).numpy().astype(np.uint8))

def transform(image: Image.Image) -> torch.Tensor:
    """图像预处理：调整大小并转换为张量"""
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])(image)

def _load_from_cache(cache_dir: str, size: int) -> List[Image.Image]:
    """通用缓存加载函数"""
    images = []
    if not os.path.exists(cache_dir):
        return images
    
    cached_files = [f for f in os.listdir(cache_dir) if f.endswith('.jpg')]
    print(f"发现本地缓存的 {len(cached_files)} 张图像")
    
    if len(cached_files) >= size:
        print(f"使用本地缓存的图像数据...")
        for filename in sorted(cached_files)[:size]:
            img_path = os.path.join(cache_dir, filename)
            images.append(Image.open(img_path))
        print(f"成功从缓存加载 {len(images)} 张图像")
    
    return images[:size]

def load_dataset(name: str, size: int, path: str) -> List[Image.Image]:
    """
    加载指定数据集
    
    Args:
        name: 数据集名称，必须在DATASETS中
        size: 数据集大小
        path: 存储路径
    
    Returns:
        List[Image.Image]: 图像列表
    """
    if name not in DATASETS:
        raise ValueError(f"数据集 {name} 不支持。支持的数据集: {DATASETS}")
    
    os.makedirs(path, exist_ok=True)
    
    if name == 'TIP-I2V':
        return _load_tip_i2v(size, path)
    elif name == 'AFHQ-V2':
        return _load_afhq_v2(size, path)
    elif name == 'Wikiart':
        return _load_wikiart(size, path)
    elif name == 'LHQ':
        return _load_lhq(size, path)
    elif name == 'Flickr30k':
        return _load_flickr30k(size, path)

def _load_tip_i2v(size: int, path: str) -> List[Image.Image]:
    """加载TIP-I2V数据集"""
    cache_dir = os.path.join(path, "tip_i2v")
    images = _load_from_cache(cache_dir, size)
    if len(images) >= size:
        return images
    
    print(f"正在从HuggingFace加载TIP-I2V数据集...")
    os.makedirs(cache_dir, exist_ok=True)
    
    hf_dataset = hf_load_dataset("WenhaoWang/TIP-I2V", split="Eval", trust_remote_code=True)
    
    for i, item in enumerate(hf_dataset):
        if len(images) >= size:
            break
        if 'Image_Prompt' in item and item['Image_Prompt'] is not None:
            pil_image = item['Image_Prompt']
            img_path = os.path.join(cache_dir, f"tip_i2v_{i:06d}.jpg")
            if not os.path.exists(img_path):
                pil_image.save(img_path)
            images.append(pil_image)
    
    print(f"成功加载 {len(images)} 张图像")
    return images[:size]

def _load_afhq_v2(size: int, path: str) -> List[Image.Image]:
    """加载AFHQ-V2数据集"""
    cache_dir = os.path.join(path, "afhq_v2")
    images = _load_from_cache(cache_dir, size)
    if len(images) >= size:
        return images
    
    print(f"正在从HuggingFace加载AFHQ-V2数据集...")
    os.makedirs(cache_dir, exist_ok=True)
    
    hf_dataset = hf_load_dataset("huggan/AFHQv2", split="train", trust_remote_code=True)
    
    for i, item in enumerate(hf_dataset):
        if len(images) >= size:
            break
        if 'image' in item and item['image'] is not None:
            pil_image = item['image']
            img_path = os.path.join(cache_dir, f"afhq_v2_{i:06d}.jpg")
            if not os.path.exists(img_path):
                pil_image.save(img_path)
            images.append(pil_image)
    
    print(f"成功加载 {len(images)} 张图像")
    return images[:size]

def _load_wikiart(size: int, path: str) -> List[Image.Image]:
    """加载Wikiart数据集"""
    cache_dir = os.path.join(path, "wikiart")
    train_dir = os.path.join(path, "train")
    
    # 检查缓存目录
    images = _load_from_cache(cache_dir, size)
    if len(images) >= size:
        return images
    
    # 检查train目录
    if os.path.exists(train_dir):
        img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        files = [f for f in os.listdir(train_dir) if f.lower().endswith(img_extensions)]
        print(f"发现本地train目录的 {len(files)} 张图像")
        
        if len(files) >= size:
            for filename in sorted(files)[:size]:
                img_path = os.path.join(train_dir, filename)
                images.append(Image.open(img_path))
            print(f"成功从train目录加载 {len(images)} 张图像")
            return images
    
    raise FileNotFoundError(f"""
Wikiart数据集未找到，请下载数据集到 {path}
使用Kaggle API: kaggle competitions download painter-by-numbers -p {path}
""")

def _load_lhq(size: int, path: str) -> List[Image.Image]:
    """加载LHQ数据集"""
    cache_dir = os.path.join(path, "lhq")
    images = _load_from_cache(cache_dir, size)
    if len(images) >= size:
        return images
    
    print(f"正在从KaggleHub加载LHQ数据集...")
    
    import kagglehub
    dataset_path = kagglehub.dataset_download("dimensi0n/lhq-1024")
    os.makedirs(cache_dir, exist_ok=True)
    
    count = 0
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if count >= size:
                break
            if file.lower().endswith('.jpg'):
                img_path = os.path.join(root, file)
                image = Image.open(img_path)
                cache_path = os.path.join(cache_dir, f"lhq_{count:06d}.jpg")
                if not os.path.exists(cache_path):
                    image.save(cache_path)
                images.append(image)
                count += 1
        if count >= size:
            break
    
    print(f"成功加载 {len(images)} 张图像")
    return images[:size]

def _load_flickr30k(size: int, path: str) -> List[Image.Image]:
    """加载Flickr30k数据集"""
    cache_dir = os.path.join(path, "flickr30k")
    images = _load_from_cache(cache_dir, size)
    if len(images) >= size:
        return images
    
    print(f"正在从HuggingFace加载Flickr30k数据集...")
    os.makedirs(cache_dir, exist_ok=True)
    
    hf_dataset = hf_load_dataset("nlphuji/flickr30k", split="test")
    
    for i, item in enumerate(hf_dataset):
        if len(images) >= size:
            break
        if 'image' in item and item['image'] is not None:
            pil_image = item['image']
            img_path = os.path.join(cache_dir, f"flickr30k_{i:06d}.jpg")
            if not os.path.exists(img_path):
                pil_image.save(img_path)
            images.append(pil_image)
    
    print(f"成功加载 {len(images)} 张图像")
    return images[:size]


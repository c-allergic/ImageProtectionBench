import os
import random
import time
from typing import List, Optional, Any
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
from datasets import load_dataset as hf_load_dataset
from requests.exceptions import RequestException
import requests

# 支持的数据集列表
DATASETS = ['TIP-I2V', 'AFHQ-V2', 'Wikiart', 'LHQ', 'Flickr30k']

def pt_to_pil(tensor:torch.Tensor,video:bool=False):
    if video:   
        video = []
        for t in range(tensor.size(0)):
            frames = tensor[t].cpu()
            frame_array = (frames.permute(1, 2, 0) * 255).clamp(0, 255).numpy().astype(np.uint8)
            video.append(Image.fromarray(frame_array))
        return video
    else:
        image = tensor.cpu()
        return Image.fromarray((image.permute(1, 2, 0) * 255).clamp(0, 255).numpy().astype(np.uint8))


def transform(image: Image.Image) -> torch.Tensor:
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])(image)

def load_dataset(name: str, size: int, path: str) -> List[Image.Image]:
    """
    加载指定数据集

    Args:
        name: 数据集名称，必须在DATASETS中
        size: 数据集大小
        path: 存储路径

    Returns:
        array[images]: 图像数组列表
    """
    if name not in DATASETS:
        raise ValueError(f"数据集 {name} 不支持。支持的数据集: {DATASETS}")

    # 确保目录存在
    os.makedirs(path, exist_ok=True)

    # 根据数据集类型加载
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
    images = []

    print(f"正在从HuggingFace加载TIP-I2V数据集...")

    # 首先检查本地缓存
    tip_i2v_cache_dir = os.path.join(path, "tip_i2v")
    cached_files = []
    if os.path.exists(tip_i2v_cache_dir):
        cached_files = [f for f in os.listdir(tip_i2v_cache_dir) if f.endswith('.jpg')]
        print(f"发现本地缓存的 {len(cached_files)} 张图像")

    # 如果本地缓存足够，直接使用
    if len(cached_files) >= size:
        print(f"使用本地缓存的图像数据...")
        cached_files_sorted = sorted(cached_files)[:size]
        for filename in cached_files_sorted:
            img_path = os.path.join(tip_i2v_cache_dir, filename)
            image = Image.open(img_path)  # 直接加载为PIL图像
            images.append(image)
        if len(images) >= size:
            print(f"成功从缓存加载 {len(images)} 张图像")
            return images[:size]

    # 数据集实际的split是 ['Full', 'Subset', 'Eval']，使用 'Eval' 作为测试集
    hf_dataset = hf_load_dataset("WenhaoWang/TIP-I2V", split="Eval")

    # 确保输出目录存在
    os.makedirs(tip_i2v_cache_dir, exist_ok=True)

    for i, item in enumerate(hf_dataset):
        if i >= size:
            break
        # TIP-I2V数据集的图像字段叫做 'Image_Prompt'
        if 'Image_Prompt' in item and item['Image_Prompt'] is not None:
            pil_image = item['Image_Prompt']

            # 可选：保存到本地以便后续使用
            img_path = os.path.join(tip_i2v_cache_dir, f"tip_i2v_{i:06d}.jpg")
            if not os.path.exists(img_path):
                pil_image.save(img_path)

            # 直接使用PIL图像
            images.append(pil_image)

    print(f"成功加载 {len(images)} 张图像")
    return images


def _load_afhq_v2(size: int, path: str) -> List[Image.Image]:
    """加载AFHQ-V2数据集"""
    images = []

    print(f"正在从HuggingFace加载AFHQ-V2数据集...")

    # 首先检查本地缓存
    afhq_cache_dir = os.path.join(path, "afhq_v2")
    cached_files = []
    if os.path.exists(afhq_cache_dir):
        cached_files = [f for f in os.listdir(afhq_cache_dir) if f.endswith('.jpg')]
        print(f"发现本地缓存的 {len(cached_files)} 张图像")

    # 如果本地缓存足够，直接使用
    if len(cached_files) >= size:
        print(f"使用本地缓存的图像数据...")
        cached_files_sorted = sorted(cached_files)[:size]
        for filename in cached_files_sorted:
            img_path = os.path.join(afhq_cache_dir, filename)
            image = Image.open(img_path)
            images.append(image)
        if len(images) >= size:
            print(f"成功从缓存加载 {len(images)} 张图像")
            return images[:size]

    # 加载HuggingFace数据集
    hf_dataset = hf_load_dataset("huggan/AFHQv2", split="train")

    # 确保输出目录存在
    os.makedirs(afhq_cache_dir, exist_ok=True)

    for i, item in enumerate(hf_dataset):
        if i >= size:
            break
        # AFHQ-V2数据集的图像字段叫做 'image'
        if 'image' in item and item['image'] is not None:
            pil_image = item['image']

            img_path = os.path.join(afhq_cache_dir, f"afhq_v2_{i:06d}.jpg")
            if not os.path.exists(img_path):
                pil_image.save(img_path)

            images.append(pil_image)

    print(f"成功加载 {len(images)} 张图像")
    return images


def _load_wikiart(size: int, path: str) -> List[Image.Image]:
    """加载Wikiart数据集 (Kaggle Painter by Numbers)"""
    images = []

    print(f"正在加载Wikiart (Painter by Numbers) 数据集...")

    # 首先检查本地缓存
    wikiart_cache_dir = os.path.join(path, "wikiart")
    train_path = os.path.join(path, "train")
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    cached_files = []

    # 优先使用 wikiart_cache_dir，如果没有则用 train_path
    if os.path.exists(wikiart_cache_dir):
        cached_files = [f for f in os.listdir(wikiart_cache_dir) if f.endswith('.jpg')]
        print(f"发现本地缓存的 {len(cached_files)} 张图像")
        if len(cached_files) >= size:
            print(f"使用本地缓存的图像数据...")
            cached_files_sorted = sorted(cached_files)[:size]
            for filename in cached_files_sorted:
                img_path = os.path.join(wikiart_cache_dir, filename)
                image = Image.open(img_path)
                images.append(image)
            if len(images) >= size:
                print(f"成功从缓存加载 {len(images)} 张图像")
                return images[:size]

    # 如果没有 wikiart_cache_dir，则检查 train_path
    if os.path.exists(train_path):
        all_files = [f for f in os.listdir(train_path) if f.lower().endswith(img_extensions)]
        print(f"发现本地train目录的 {len(all_files)} 张图像")
        if len(all_files) >= size:
            selected_files = sorted(all_files)[:size]
            for filename in selected_files:
                img_path = os.path.join(train_path, filename)
                image = Image.open(img_path)
                images.append(image)
            if len(images) >= size:
                print(f"成功从train目录加载 {len(images)} 张图像")
                return images[:size]

    # 如果本地没有数据，提示用户下载
    raise FileNotFoundError(f"""
Wikiart (Painter by Numbers) 数据集未找到，请按以下步骤下载：
1. 安装 Kaggle API: pip install kaggle
2. 设置 Kaggle 凭据 (~/.kaggle/kaggle.json)
3. 下载数据集: kaggle competitions download painter-by-numbers -p {path}
4. 解压数据集到 {path}
数据集链接: https://www.kaggle.com/c/painter-by-numbers/data
""")


def _load_lhq(size: int, path: str) -> List[Image.Image]:
    """加载LHQ数据集 (KaggleHub)"""
    images = []

    print(f"正在从KaggleHub加载LHQ数据集...")

    # 首先检查本地缓存
    lhq_cache_dir = os.path.join(path, "lhq")
    cached_files = []
    if os.path.exists(lhq_cache_dir):
        cached_files = [f for f in os.listdir(lhq_cache_dir) if f.endswith('.jpg')]
        print(f"发现本地缓存的 {len(cached_files)} 张图像")

    # 如果本地缓存足够，直接使用
    if len(cached_files) >= size:
        print(f"使用本地缓存的图像数据...")
        cached_files_sorted = sorted(cached_files)[:size]
        for filename in cached_files_sorted:
            img_path = os.path.join(lhq_cache_dir, filename)
            image = Image.open(img_path)
            images.append(image)
        if len(images) >= size:
            print(f"成功从缓存加载 {len(images)} 张图像")
            return images[:size]

    # 如果没有本地缓存，则尝试用kagglehub下载
    import kagglehub

    # 下载最新版本
    dataset_path = kagglehub.dataset_download("dimensi0n/lhq-1024")
    print(f"数据集下载路径: {dataset_path}")

    # 确保输出目录存在
    os.makedirs(lhq_cache_dir, exist_ok=True)

    count = 0
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if count >= size:
                break
            if file.lower().endswith('.jpg'):
                img_path = os.path.join(root, file)
                try:
                    image = Image.open(img_path)
                    cache_path = os.path.join(lhq_cache_dir, f"lhq_{count:06d}.jpg")
                    if not os.path.exists(cache_path):
                        image.save(cache_path)
                    images.append(image)
                    count += 1
                except Exception as e:
                    print(f"跳过文件 {file}，加载失败: {e}")
                    continue
        if count >= size:
            break

    print(f"成功加载 {len(images)} 张图像")
    return images



def _load_flickr30k(size: int, path: str) -> List[Image.Image]:
    """加载Flickr30k数据集 (HuggingFace)"""
    images = []

    print(f"正在从HuggingFace加载Flickr30k数据集...")

    flickr_cache_dir = os.path.join(path, "flickr30k")
    cached_files = []
    if os.path.exists(flickr_cache_dir):
        cached_files = [f for f in os.listdir(flickr_cache_dir) if f.endswith('.jpg')]
        print(f"发现本地缓存的 {len(cached_files)} 张图像")

    # 如果本地缓存足够，直接使用
    if len(cached_files) >= size:
        print(f"使用本地缓存的图像数据...")
        cached_files_sorted = sorted(cached_files)[:size]
        for filename in cached_files_sorted:
            img_path = os.path.join(flickr_cache_dir, filename)
            image = Image.open(img_path)
            images.append(image)
        if len(images) >= size:
            print(f"成功从缓存加载 {len(images)} 张图像")
            return images[:size]

    # 加载HuggingFace数据集
    hf_dataset = hf_load_dataset("nlphuji/flickr30k", split="test")
    os.makedirs(flickr_cache_dir, exist_ok=True)

    for i, item in enumerate(hf_dataset):
        if i >= size:
            break
        if 'image' in item and item['image'] is not None:
            pil_image = item['image']
            img_path = os.path.join(flickr_cache_dir, f"flickr30k_{i:06d}.jpg")
            if not os.path.exists(img_path):
                pil_image.save(img_path)
            images.append(pil_image)

    print(f"成功加载 {len(images)} 张图像")
    return images


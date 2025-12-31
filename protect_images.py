#!/usr/bin/env python3
"""
简洁的图片保护脚本

专门用于调用保护算法保护图片，参照benchmark.py的流程。
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import argparse
import datetime
import json
import torch
import time

from data import load_dataset, transform, pt_to_pil, DATASETS
from protection import PhotoGuard, EditShield, Mist, I2VGuard, VGMShield, RandomNoise, ExpGuard
from attacks import (RotationAttack, ResizedCropAttack, ErasingAttack, BrightnessAttack, 
                     ContrastAttack, BlurringAttack, NoiseAttack, SaltPepperAttack, CompressionAttack)
from experiment_utils import setup_output_directories


def initialize_protection_method(method_name: str, device: str):
    """初始化保护方法，参照benchmark.py"""
    print(f"Initializing protection method {method_name}...")
    if method_name == "PhotoGuard":
        protection_method = PhotoGuard(device=device)
    elif method_name == "EditShield":
        protection_method = EditShield(device=device)
    elif method_name == "Mist":
        protection_method = Mist(device=device)
    elif method_name == "I2VGuard":
        protection_method = I2VGuard(device=device)
    elif method_name == "VGMShield":
        protection_method = VGMShield(device=device)
    elif method_name == "RandomNoise":
        protection_method = RandomNoise(device=device)
    elif method_name == "ExpGuard":
        protection_method = ExpGuard(device=device)
    else:
        raise ValueError(f"Unknown protection method: {method_name}")
    return protection_method


def initialize_attack_method(attack_type: str, device: str):
    """初始化攻击方法，参照benchmark.py"""
    print(f"Initializing attack method {attack_type}")
    if attack_type == "rotation":
        attack_method = RotationAttack(device=device)
    elif attack_type == "resizedcrop":
        attack_method = ResizedCropAttack(device=device)
    elif attack_type == "erasing":
        attack_method = ErasingAttack(device=device)
    elif attack_type == "brightness":
        attack_method = BrightnessAttack(device=device)
    elif attack_type == "contrast":
        attack_method = ContrastAttack(device=device)
    elif attack_type == "blurring":
        attack_method = BlurringAttack(device=device)
    elif attack_type == "noise":
        attack_method = NoiseAttack(device=device)
    elif attack_type == "saltpepper":
        attack_method = SaltPepperAttack(device=device)
    elif attack_type == "compression":
        attack_method = CompressionAttack(device=device)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
    return attack_method


def protect_images_batch(images, protection_method, batch_size=5, attack_method=None, results_dir=None):
    """批量保护图片，参照experiment.py的批次处理逻辑"""
    total_images = len(images)
    attack_info = f"，攻击方法: {attack_method.__class__.__name__}" if attack_method else "，无攻击"
    print(f"开始保护 {total_images} 张图片，使用批次处理模式，每批处理 {batch_size} 张图片{attack_info}")
    
    # 时间统计
    total_protection_time = 0.0
    total_images_processed = 0
    all_protected_images = []
    all_attacked_images = []
    
    # 分批处理
    for i in range(0, total_images, batch_size):
        batch_end = min(i + batch_size, total_images)
        batch_images = images[i:batch_end]
        batch_num = i // batch_size + 1
        
        print(f"\n=== 处理批次 {batch_num} ({i+1}-{batch_end}) ===")
        
        # ExpGuard需要原始PIL Image以避免双重resize，其他方法使用transform
        if isinstance(protection_method, ExpGuard):
            original_images = batch_images  # 直接使用PIL Image列表
            
            # 应用保护
            start_time = time.time()
            protected_tensors = protection_method.protect_multiple(
                original_images, 
                results_dir=results_dir
            )
        else:
            # 其他保护方法使用transform转换为tensor
            original_tensors = []
            for img_pil in batch_images:
                img_tensor = transform(img_pil).to(protection_method.device)
                original_tensors.append(img_tensor)
            original_tensors = torch.stack(original_tensors)
            
            # 应用保护
            start_time = time.time()
            protected_tensors = protection_method.protect_multiple(original_tensors)
        protection_elapsed = time.time() - start_time
        
        # 获取批次大小（ExpGuard返回tensor，其他方法也返回tensor）
        batch_size_actual = protected_tensors.size(0) if isinstance(protected_tensors, torch.Tensor) else len(batch_images)
        total_protection_time += protection_elapsed
        total_images_processed += batch_size_actual
        
        print(f"保护操作完成: 处理 {batch_size_actual} 张图片，耗时 {protection_elapsed:.4f}秒，平均 {protection_elapsed/batch_size_actual:.4f}秒/图片")
        
        # 应用攻击 (如果启用)
        attacked_tensors = None
        if attack_method is not None:
            print("应用攻击变换...")
            attacked_tensors = attack_method.attack_multiple(protected_tensors)
            print("攻击变换完成")
        
        # 转换回PIL格式
        for j in range(protected_tensors.size(0)):
            protected_pil = pt_to_pil(protected_tensors[j])
            all_protected_images.append(protected_pil)
            
            # 如果有攻击结果，也转换为PIL
            if attacked_tensors is not None:
                attacked_pil = pt_to_pil(attacked_tensors[j])
                all_attacked_images.append(attacked_pil)
        
        # 清理显存
        if isinstance(protection_method, ExpGuard):
            # ExpGuard使用PIL Image，不需要删除original_tensors
            del protected_tensors
        else:
            del original_tensors, protected_tensors
        if attacked_tensors is not None:
            del attacked_tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"批次 {batch_num} 完成")
    
    # 准备返回结果
    result = {'protected_images': all_protected_images}
    
    # 如果有攻击结果，也返回
    if all_attacked_images:
        result['attacked_images'] = all_attacked_images
    
    time_per_image = total_protection_time / total_images_processed if total_images_processed > 0 else 0
    result['time_stats'] = {
        'total_protection_time': total_protection_time,
        'total_images_processed': total_images_processed,
        'time_per_image': time_per_image
    }
    
    return result


def save_protected_images(original_images, protected_images, images_dir, attacked_images=None):
    """保存保护后的图片"""
    print(f"保存图片到: {images_dir}")
    
    for i, (orig_img, prot_img) in enumerate(zip(original_images, protected_images)):
        # 保存原始图片
        orig_path = os.path.join(images_dir, f"original_{i:03d}.png")
        orig_img.save(orig_path)
        
        # 保存保护后的图片
        prot_path = os.path.join(images_dir, f"protected_{i:03d}.png")
        prot_img.save(prot_path)
        
        # 保存攻击后的图片（如果有）
        if attacked_images and i < len(attacked_images):
            attack_path = os.path.join(images_dir, f"attacked_{i:03d}.png")
            attacked_images[i].save(attack_path)
        
        if i < 5 or i % 50 == 0:  # 只打印前5个和每50个的进度
            attack_info = f" & {os.path.basename(attack_path)}" if attacked_images and i < len(attacked_images) else ""
            print(f"  图片 {i}: {os.path.basename(orig_path)} & {os.path.basename(prot_path)}{attack_info}")
    
    image_count = len(protected_images)
    if attacked_images:
        print(f"总共保存了 {image_count} 组图片 (原始 + 保护 + 攻击)")
    else:
        print(f"总共保存了 {image_count} 对图片 (原始 + 保护)")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="简洁的图片保护脚本")
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default="Flickr30k", choices=DATASETS)
    parser.add_argument('--num_samples', type=int, default=50) 
    parser.add_argument('--data_path', type=str, default="./data")
    
    # 保护方法参数
    parser.add_argument('--protection_method', type=str, default="ExpGuard", 
                       choices=["PhotoGuard", "EditShield", "Mist", "I2VGuard", "VGMShield", "RandomNoise", "ExpGuard"])
    
    # 攻击参数
    parser.add_argument('--enable_attack', default=True, action='store_true', 
                       help="启用攻击变换")
    parser.add_argument('--attack_type', type=str, default="compression",
                       choices=["rotation", "resizedcrop", "erasing", "brightness", "contrast", 
                               "blurring", "noise", "saltpepper", "compression"],
                       help="攻击类型")
    
    # 处理参数
    parser.add_argument('--batch_size', type=int, default=1)
    
    # 系统参数
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--output_dir', type=str, default="EXP_outputs_flickr30k")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("图片保护脚本")
    print("=" * 60)
    print(f"数据集: {args.dataset}")
    print(f"图片数量: {args.num_samples}")
    print(f"保护方法: {args.protection_method}")
    print(f"启用攻击: {'是' if args.enable_attack else '否'}")
    if args.enable_attack:
        print(f"攻击类型: {args.attack_type}")
    print(f"批次大小: {args.batch_size}")
    print(f"计算设备: {args.device}")
    print(f"输出目录: {args.output_dir}")
    print()
    
    # 设置设备
    device = args.device if torch.cuda.is_available() else "cpu"
    args.device = device
    
    # 创建输出目录，参照benchmark.py
    output_dirs = setup_output_directories(
        args.output_dir, 
        f"{args.protection_method}_{datetime.datetime.now().strftime('%m%d_%H%M')}"
    )
    
    # 保存参数，参照benchmark.py
    with open(os.path.join(output_dirs['results'], "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # 加载数据集，参照benchmark.py
    print(f"Loading dataset {args.dataset}...")
    data, prompts = load_dataset(args.dataset, args.num_samples, args.data_path, generate_descriptions=False, device=device)
    print(f"成功加载 {len(data)} 张图片")
    
    # 初始化保护方法，参照benchmark.py
    protection_method = initialize_protection_method(args.protection_method, device)
    
    # 初始化攻击方法 (如果启用)
    attack_method = None
    if args.enable_attack:
        attack_method = initialize_attack_method(args.attack_type, device)
    
    # 保护图片
    print("\n开始保护图片...")
    result = protect_images_batch(
        data, 
        protection_method, 
        batch_size=args.batch_size,
        attack_method=attack_method,
        results_dir=output_dirs['results']  # 传递results目录路径
    )
    
    # 保存保护后的图片
    print("\n保存保护后的图片...")
    attacked_images = result.get('attacked_images', None)
    save_protected_images(data, result['protected_images'], output_dirs['images'], attacked_images)
    
    # 保存时间统计（如果启用）
    if 'time_stats' in result:
        time_file = os.path.join(output_dirs['results'], "time_stats.json")
        with open(time_file, "w") as f:
            json.dump(result['time_stats'], f, indent=4)
        print(f"时间统计已保存到: {time_file}")
    
    # 保存处理信息
    info = {
        'method': args.protection_method,
        'dataset': args.dataset,
        'num_samples': args.num_samples,
        'batch_size': args.batch_size,
        'device': device,
        'enable_attack': args.enable_attack,
        'attack_type': args.attack_type if args.enable_attack else None,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    info_file = os.path.join(output_dirs['results'], "protection_info.json")
    with open(info_file, "w") as f:
        json.dump(info, f, indent=4)
    
    print("\n" + "=" * 60)
    print("图片保护完成!")
    print(f"保护方法: {args.protection_method}")
    if args.enable_attack:
        print(f"攻击方法: {args.attack_type}")
    print(f"处理图片数: {len(result['protected_images'])}")
    if attacked_images:
        print(f"攻击图片数: {len(attacked_images)}")
    print(f"输出目录: {output_dirs['experiment']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

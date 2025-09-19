#!/usr/bin/env python3
"""
综合评估脚本

从指定文件夹的videos和images子文件夹读取数据，评估VBench分数、图像质量和攻击有效性，
并将结果保存到results子文件夹的benchmark_results.json中。

使用方法:
    python evaluation.py --input_dir /path/to/experiment/folder
    python evaluation.py --input_dir /path/to/experiment/folder --method_name "EditShield"
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import json
import argparse
import glob
from typing import Dict, List, Optional
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# 导入必要的模块
from metrics.video_quality import VBenchMetric
from metrics import PSNRMetric, SSIMMetric, CLIPScoreMetric, LPIPSMetric
from data import transform, pt_to_pil
from vbench.utils import load_video


def find_video_pairs(videos_dir: str) -> List[Dict[str, str]]:
    """从videos目录中找到视频对"""
    if not os.path.exists(videos_dir):
        print(f"错误: videos目录不存在: {videos_dir}")
        return []
    
    video_pairs = []
    
    # 查找所有原始视频文件
    original_pattern = os.path.join(videos_dir, "original_*.mp4")
    original_files = glob.glob(original_pattern)
    original_files.sort()
    
    print(f"找到 {len(original_files)} 个原始视频文件")
    
    for orig_path in original_files:
        filename = os.path.basename(orig_path)
        video_id = filename.replace("original_", "").replace(".mp4", "")
        
        prot_path = os.path.join(videos_dir, f"protected_{video_id}.mp4")
        
        if os.path.exists(prot_path):
            video_pairs.append({
                'original_path': orig_path,
                'protected_path': prot_path,
                'video_id': video_id
            })
            print(f"  视频对 {video_id}: {os.path.basename(orig_path)} & {os.path.basename(prot_path)}")
        else:
            print(f"  警告: 未找到对应的保护视频: {prot_path}")
    
    # 检查是否有攻击视频
    attacked_pattern = os.path.join(videos_dir, "attacked_*.mp4")
    attacked_files = glob.glob(attacked_pattern)
    
    if attacked_files:
        print(f"找到 {len(attacked_files)} 个攻击视频文件")
        for pair in video_pairs:
            video_id = pair['video_id']
            attacked_path = os.path.join(videos_dir, f"attacked_{video_id}.mp4")
            if os.path.exists(attacked_path):
                pair['attacked_path'] = attacked_path
                print(f"  攻击视频 {video_id}: {os.path.basename(attacked_path)}")
    
    print(f"总共找到 {len(video_pairs)} 个有效视频对")
    return video_pairs


def find_image_pairs(images_dir: str) -> List[Dict[str, str]]:
    """从images目录中找到图片对"""
    if not os.path.exists(images_dir):
        print(f"错误: images目录不存在: {images_dir}")
        return []
    
    image_pairs = []
    
    # 查找所有原始图片文件
    original_pattern = os.path.join(images_dir, "original_*.png")
    original_files = glob.glob(original_pattern)
    original_files.sort()
    
    print(f"找到 {len(original_files)} 个原始图片文件")
    
    for orig_path in original_files:
        filename = os.path.basename(orig_path)
        image_id = filename.replace("original_", "").replace(".png", "")
        
        prot_path = os.path.join(images_dir, f"protected_{image_id}.png")
        
        if os.path.exists(prot_path):
            image_pairs.append({
                'original_path': orig_path,
                'protected_path': prot_path,
                'image_id': image_id
            })
            print(f"  图片对 {image_id}: {os.path.basename(orig_path)} & {os.path.basename(prot_path)}")
        else:
            print(f"  警告: 未找到对应的保护图片: {prot_path}")
    
    # 检查是否有攻击图片
    attacked_pattern = os.path.join(images_dir, "attacked_*.png")
    attacked_files = glob.glob(attacked_pattern)
    
    if attacked_files:
        print(f"找到 {len(attacked_files)} 个攻击图片文件")
        for pair in image_pairs:
            image_id = pair['image_id']
            attacked_path = os.path.join(images_dir, f"attacked_{image_id}.png")
            if os.path.exists(attacked_path):
                pair['attacked_path'] = attacked_path
                print(f"  攻击图片 {image_id}: {os.path.basename(attacked_path)}")
    
    print(f"总共找到 {len(image_pairs)} 个有效图片对")
    return image_pairs


def load_images_as_tensors(image_pairs: List[Dict[str, str]], device: str) -> tuple:
    """加载图片对为张量"""
    original_tensors = []
    protected_tensors = []
    attacked_tensors = []
    
    for pair in image_pairs:
        # 加载原始图片
        orig_img = Image.open(pair['original_path']).convert('RGB')
        orig_tensor = transform(orig_img).to(device)
        original_tensors.append(orig_tensor)
        
        # 加载保护后图片
        prot_img = Image.open(pair['protected_path']).convert('RGB')
        prot_tensor = transform(prot_img).to(device)
        protected_tensors.append(prot_tensor)
        
        # 加载攻击后图片（如果存在）
        if 'attacked_path' in pair:
            attack_img = Image.open(pair['attacked_path']).convert('RGB')
            attack_tensor = transform(attack_img).to(device)
            attacked_tensors.append(attack_tensor)
    
    original_tensors = torch.stack(original_tensors)
    protected_tensors = torch.stack(protected_tensors)
    attacked_tensors = torch.stack(attacked_tensors) if attacked_tensors else None
    
    return original_tensors, protected_tensors, attacked_tensors


def load_videos_as_tensors(video_pairs: List[Dict[str, str]], device: str) -> tuple:
    """从视频文件加载为张量，参照experiment.py的处理方式"""
    original_videos = []
    protected_videos = []
    attacked_videos = []
    
    print("从视频文件加载张量...")
    
    for i, pair in enumerate(video_pairs):
        print(f"  加载视频 {i+1}/{len(video_pairs)}: {pair['video_id']}")
        
        try:
            # 加载原始视频
            orig_video_tensor = load_video(pair['original_path'], return_tensor=True)
            # 转换为float类型并归一化到[0,1]，与I2V模型输出格式一致
            orig_video_tensor = orig_video_tensor.float() / 255.0
            original_videos.append(orig_video_tensor)
            
            # 加载保护后视频
            prot_video_tensor = load_video(pair['protected_path'], return_tensor=True)
            # 转换为float类型并归一化到[0,1]，与I2V模型输出格式一致
            prot_video_tensor = prot_video_tensor.float() / 255.0
            protected_videos.append(prot_video_tensor)
            
            # 加载攻击后视频（如果存在）
            if 'attacked_path' in pair:
                attack_video_tensor = load_video(pair['attacked_path'], return_tensor=True)
                # 转换为float类型并归一化到[0,1]，与I2V模型输出格式一致
                attack_video_tensor = attack_video_tensor.float() / 255.0
                attacked_videos.append(attack_video_tensor)
                
        except Exception as e:
            print(f"    警告: 视频 {pair['video_id']} 加载失败: {e}")
            continue
    
    if not original_videos:
        print("错误: 没有成功加载任何视频")
        return None, None, None
    
    # 堆叠为批次张量
    original_videos = torch.stack(original_videos).to(device)
    protected_videos = torch.stack(protected_videos).to(device)
    attacked_videos = torch.stack(attacked_videos).to(device) if attacked_videos else None
    
    print(f"成功加载 {len(original_videos)} 个视频对")
    print(f"原始视频张量形状: {original_videos.shape}")
    print(f"保护后视频张量形状: {protected_videos.shape}")
    if attacked_videos is not None:
        print(f"攻击后视频张量形状: {attacked_videos.shape}")
    
    return original_videos, protected_videos, attacked_videos


def evaluate_images(original_tensors, protected_tensors, metrics, attacked_tensors=None):
    """评估图像质量，参照experiment.py"""
    results = {}    
    
    for metric_name, metric in metrics.items():
        if metric_name in ['psnr', 'ssim', 'lpips']:
            # 原图 vs 保护后图片的指标
            metric_result = metric.compute_multiple(original_tensors, protected_tensors)
            if metric_result:
                for key, value in metric_result.items():
                    results[f'protected_{key}'] = value
            
            # 如果有攻击后图片，计算原图 vs 攻击后图片的指标
            if attacked_tensors is not None:
                attack_metric_result = metric.compute_multiple(original_tensors, attacked_tensors)
                if attack_metric_result:
                    for key, value in attack_metric_result.items():
                        results[f'attacked_{key}'] = value
    
    return results


def evaluate_videos(metrics, video_paths=None, original_videos=None, protected_videos=None, attacked_videos=None, compute_clip_bounds=True):
    """评估视频质量和攻击有效性，参照experiment.py"""
    results = {}
    print(f"开始视频评估，video_paths数量: {len(video_paths) if video_paths else 0}")
    
    for metric_name, metric in metrics.items():
        print(f"处理metric: {metric_name}")
        
        if metric_name == 'clip' and original_videos is not None and protected_videos is not None:
            print("Running CLIP evaluation on video tensors...")
            try:
                # 原图 vs 保护后视频的CLIP评估
                clip_result = metric.compute_multiple(original_videos, protected_videos)
                if clip_result:
                    for key, value in clip_result.items():
                        results[f'protected_{key}'] = value
                    print(f"保护后视频CLIP评估完成，获得 {len(clip_result)} 个结果")
                
                # 只在第一个批次计算CLIP理论上限和下限
                if compute_clip_bounds:
                    print("计算CLIP理论上限...")
                    print(f"  使用视频张量形状: {original_videos.shape}")
                    upper_bound = metric.compute_upper_bound(original_videos, sample_size=10)
                    results['clip_upper_bound'] = upper_bound
                    print(f"CLIP理论上限: {upper_bound:.4f}")
                    
                    print("计算CLIP理论下限...")
                    print(f"  使用视频张量形状: {original_videos.shape}")
                    lower_bound = metric.compute_lower_bound(original_videos, sample_size=10)
                    results['clip_lower_bound'] = lower_bound
                    print(f"CLIP理论下限: {lower_bound:.4f}")
                    
                
                # 如果有攻击后视频，计算原图 vs 攻击后视频的CLIP评估  
                if attacked_videos is not None:
                    attack_clip_result = metric.compute_multiple(original_videos, attacked_videos)
                    if attack_clip_result:
                        for key, value in attack_clip_result.items():
                            results[f'attacked_{key}'] = value
                        print(f"攻击后视频CLIP评估完成，获得 {len(attack_clip_result)} 个结果")
                        
            except Exception as e:
                print(f"CLIP评估出错: {e}")
                
        elif metric_name == 'vbench':
            if video_paths is not None:
                print("Running VBench evaluation using saved video files...")
                try:
                    metric_result = metric.compute_multiple(video_paths)
                    if metric_result:
                        for key, value in metric_result.items():
                            results[f'{key}'] = value
                        print(f"VBench评估完成，获得 {len(metric_result)} 个结果")
                except Exception as e:
                    print(f"VBench评估出错: {e}")
            else:
                print("VBench metric requires video file paths, skipping...")
        else:
            print(f"跳过metric: {metric_name} (不支持的metric类型)")

    print(f"视频评估完成，总共获得 {len(results)} 个结果")
    return results


def setup_metrics(device: str = "cuda") -> Dict:
    """设置所有评估指标"""
    print("初始化评估指标...")
    
    metrics = {
        'psnr': PSNRMetric(device=device),
        'ssim': SSIMMetric(device=device),
        'lpips': LPIPSMetric(device=device),
        'clip': CLIPScoreMetric(device=device),
        'vbench': VBenchMetric(
            device=device,
            vbench_info_path="./metrics/vbench/VBench_full_info.json",
            output_dir="evaluation_result",
            dimensions=[
                "subject_consistency", 
                "motion_smoothness",
                "aesthetic_quality",
                "imaging_quality"
            ]
        )
    }
    
    print("评估指标初始化完成")
    return metrics


def evaluate_videos_batch(video_pairs: List[Dict[str, str]], vbench_metric: VBenchMetric, batch_size: int = 10) -> Dict:
    """评估视频对并返回结果，参照experiment.py的批次处理逻辑"""
    total_videos = len(video_pairs)
    print(f"开始评估 {total_videos} 个视频对，使用批次处理模式，每批处理 {batch_size} 个视频")
    
    all_batch_results = []
    
    for i in range(0, total_videos, batch_size):
        batch_end = min(i + batch_size, total_videos)
        batch_video_pairs = video_pairs[i:batch_end]
        batch_num = i // batch_size + 1
        
        print(f"\n=== 处理批次 {batch_num} ({i+1}-{batch_end}) ===")
        
        try:
            batch_results = vbench_metric.compute_multiple(batch_video_pairs)
            all_batch_results.append(batch_results)
            print(f"批次 {batch_num} 评估完成")
        except Exception as e:
            print(f"批次 {batch_num} 评估失败: {e}")
            all_batch_results.append({})
    
    # 聚合所有批次结果
    print(f"\n聚合 {len(all_batch_results)} 个批次的结果...")
    final_results = {}
    
    for batch_results in all_batch_results:
        for key, value in batch_results.items():
            if key not in final_results:
                final_results[key] = []
            final_results[key].append(value)
    
    # 计算平均值
    aggregated = {}
    for key, values in final_results.items():
        if all(isinstance(v, (int, float)) for v in values):
            aggregated[key] = sum(values) / len(values)
            print(f"  {key}: 平均={aggregated[key]:.4f} (来自{len(values)}个批次)")
        else:
            aggregated[key] = values
    
    print(f"视频评估完成，总共处理了 {total_videos} 个视频对")
    return aggregated


def save_results(results: Dict, output_path: str, method_name: str = "Unknown"):
    """保存评估结果到JSON文件，参照experiment.py的格式"""
    final_results = {
        "method": method_name,
        "aggregated": results
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    
    print(f"结果已保存到: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="综合评估脚本")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="实验文件夹路径，包含videos和images子文件夹")
    parser.add_argument("--method_name", type=str, default="Unknown",
                       help="保护方法名称 (默认: Unknown)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="计算设备 (默认: cuda)")
    parser.add_argument("--output_filename", type=str, default="benchmark_results_clip.json",
                       help="输出文件名 (默认: benchmark_results.json)")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="批次大小，每批处理的视频数量 (默认: 10)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("综合评估脚本")
    print("=" * 60)
    print(f"输入目录: {args.input_dir}")
    print(f"方法名称: {args.method_name}")
    print(f"计算设备: {args.device}")
    print(f"输出文件名: {args.output_filename}")
    print(f"批次大小: {args.batch_size}")
    print()
    
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return
    
    # 构建路径
    videos_dir = os.path.join(args.input_dir, "videos")
    images_dir = os.path.join(args.input_dir, "images")
    results_dir = os.path.join(args.input_dir, "results")
    output_path = os.path.join(results_dir, args.output_filename)
    
    print(f"Videos目录: {videos_dir}")
    print(f"Images目录: {images_dir}")
    print(f"Results目录: {results_dir}")
    print(f"输出路径: {output_path}")
    print()
    
    # 步骤1: 查找数据对
    print("步骤1: 查找数据对...")
    video_pairs = find_video_pairs(videos_dir)
    image_pairs = find_image_pairs(images_dir)
    
    if not video_pairs and not image_pairs:
        print("错误: 未找到任何有效的数据对")
        return
    
    print()
    
    # 步骤2: 设置评估指标
    print("步骤2: 设置评估指标...")
    try:
        metrics = setup_metrics(args.device)
    except Exception as e:
        print(f"错误: 评估指标初始化失败: {e}")
        return
    
    print()
    
    # 步骤3: 评估图像质量
    all_results = {}
    if image_pairs:
        print("步骤3: 评估图像质量...")
        try:
            original_tensors, protected_tensors, attacked_tensors = load_images_as_tensors(image_pairs, args.device)
            image_metrics = {k: v for k, v in metrics.items() if k in ['psnr', 'ssim', 'lpips']}
            image_results = evaluate_images(original_tensors, protected_tensors, image_metrics, attacked_tensors)
            all_results.update(image_results)
            print("图像质量评估完成")
        except Exception as e:
            print(f"错误: 图像质量评估失败: {e}")
    
    
    # 步骤4: 评估视频质量
    if video_pairs:
        print("步骤4: 评估视频质量...")
        try:
            #VBench评估
            vbench_results = evaluate_videos_batch(video_pairs, metrics['vbench'], args.batch_size)
            all_results.update(vbench_results)
            
            # CLIP评估 - 从视频文件加载为张量
            print("加载视频张量进行CLIP评估...")
            original_videos, protected_videos, attacked_videos = load_videos_as_tensors(video_pairs, args.device)
            
            if original_videos is not None and protected_videos is not None:
                # 使用experiment.py的方式评估视频
                video_metrics = {k: v for k, v in metrics.items() if k in ['clip']}
                video_results = evaluate_videos(video_metrics, video_pairs, original_videos, protected_videos, attacked_videos, compute_clip_bounds=True)
                all_results.update(video_results)
                print("视频CLIP评估完成")
            else:
                print("跳过CLIP评估：无法加载视频张量")
            
            print("视频质量评估完成")
        except Exception as e:
            print(f"错误: 视频质量评估失败: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    
    # 步骤5: 保存结果
    print("步骤5: 保存评估结果...")
    try:
        save_results(all_results, output_path, args.method_name)
    except Exception as e:
        print(f"错误: 结果保存失败: {e}")
        return
    
    print()
    print("=" * 60)
    print("评估完成!")
    print(f"结果已保存到: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Experiment utilities for ImageProtectionBench

Contains all experiment-related functions for running protection method evaluation.
"""

import numpy as np
import torch
import os
from PIL import Image
from data import transform, pt_to_pil
from diffusers.utils import export_to_video

def evaluate_images(original_images, protected_images, metrics, protection_method=None):
    """Evaluate image quality"""
    results = {}    
    
    for metric_name, metric in metrics.items():
        if metric_name in ['psnr', 'ssim', 'lpips']:
            # Image quality metrics with compute_multiple interface
            metric_result = metric.compute_multiple(original_images, protected_images)
            if metric_result:
                # Store aggregated statistics directly
                for key, value in metric_result.items():
                    results[f'image_{key}'] = value
        elif metric_name == 'time' and protection_method is not None:
            timing_stats = metric.compute(protection_method=protection_method)
            results.update(timing_stats)
    
    return results

def evaluate_videos(metrics, video_paths=None, original_videos=None, protected_videos=None):
    """Evaluate video quality and effectiveness"""
    results = {}
    print(f"开始视频评估，video_paths数量: {len(video_paths) if video_paths else 0}")
    print(f"original_videos形状: {original_videos.shape if original_videos is not None else None}")
    print(f"protected_videos形状: {protected_videos.shape if protected_videos is not None else None}")
    
    # 直接在查找时处理，不做赋值后再处理
    for metric_name, metric in metrics.items():
        print(f"处理metric: {metric_name}")
        
        if metric_name == 'clip' and original_videos is not None and protected_videos is not None:
            print("Running CLIP evaluation on video tensors...")
            try:
                clip_result = metric.compute_multiple(original_videos, protected_videos)
                if clip_result:
                    for key, value in clip_result.items():
                        results[f'video_{key}'] = value
                    print(f"CLIP评估完成，获得 {len(clip_result)} 个结果")
                else:
                    print("CLIP评估返回空结果")
            except Exception as e:
                print(f"CLIP评估出错: {e}")
                import traceback
                traceback.print_exc()
                
        elif metric_name == 'vbench':
            if video_paths is not None:
                print("Running VBench evaluation using saved video files...")
                print(f"视频路径列表: {video_paths}")
                try:
                    metric_result = metric.compute_multiple(video_paths)
                    if metric_result:
                        for key, value in metric_result.items():
                            results[f'video_{key}'] = value
                        print(f"VBench评估完成，获得 {len(metric_result)} 个结果")
                    else:
                        print("VBench评估返回空结果")
                except Exception as e:
                    print(f"VBench评估出错: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("VBench metric requires video file paths, skipping...")
        else:
            print(f"跳过metric: {metric_name} (不支持的metric类型)")

    print(f"视频评估完成，总共获得 {len(results)} 个结果")
    return results

def save_images_and_videos(original_tensors, protected_tensors, original_videos, protected_videos, save_path, i2v_model):
    """保存图片和视频"""
    # 使用实验目录的上级目录作为基础路径
    experiment_dir = os.path.dirname(save_path)
    images_dir = os.path.join(experiment_dir, "images")
    videos_dir = os.path.join(experiment_dir, "videos")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    
    # 保存图片
    print(f"保存图片到: {images_dir}")
    for i in range(original_tensors.size(0)):
        # 保存原始图片
        orig_pil = pt_to_pil(original_tensors[i])
        orig_path = os.path.join(images_dir, f"original_{i:03d}.png")
        orig_pil.save(orig_path)
        
        # 保存保护后的图片
        prot_pil = pt_to_pil(protected_tensors[i])
        prot_path = os.path.join(images_dir, f"protected_{i:03d}.png")
        prot_pil.save(prot_path)
        
        print(f"  图片 {i}: {os.path.basename(orig_path)} & {os.path.basename(prot_path)}")
    
    # 保存视频 - 使用diffusers的export_to_video方法
    print(f"保存视频到: {videos_dir}")
    
    # 处理批次视频格式 [B, T, C, H, W] -> 单个视频 [T, C, H, W]
    batch_size = original_videos.size(0)
    video_paths = []
    for i in range(batch_size):
        # 提取单个视频
        orig_frames = pt_to_pil(original_videos[i],video=True)
        prot_frames = pt_to_pil(protected_videos[i],video=True)
        
        # 调试信息
        expected_duration = len(orig_frames) / i2v_model.frame_rate
        print(f"  视频 {i}: {len(orig_frames)}帧, {i2v_model.frame_rate}fps, 预期时长: {expected_duration:.2f}秒")
        
        # 使用diffusers的export_to_video方法
        orig_video_path = os.path.join(videos_dir, f"original_{i:03d}.mp4")
        prot_video_path = os.path.join(videos_dir, f"protected_{i:03d}.mp4")
        
        export_to_video(orig_frames, orig_video_path)
        export_to_video(prot_frames, prot_video_path)
        
        print(f"  视频 {i}: {os.path.basename(orig_video_path)} & {os.path.basename(prot_video_path)}")
        video_paths.append({
            'original_path': orig_video_path,
            'protected_path': prot_video_path
        })
    
    return video_paths


def run_benchmark(args, data, protection_method, i2v_model, metrics, save_path):
    """Run benchmark for a single protection method"""
    device = args.device
    
    print(f"Running benchmark with {protection_method.__class__.__name__}")
    
    # Prepare data
    images = data[:args.num_samples]
    
    # Step 1: Apply protection
    print("Applying protection...")
    # transform the original images to tensors
    original_tensors = []
    for img_pil in images:
        img_tensor = transform(img_pil).to(device)
        original_tensors.append(img_tensor)
    original_tensors = torch.stack(original_tensors)
    
    # protect the original tensors
    print(f"使用批量处理, 批次大小: {len(images)}")
    protected_tensors = protection_method.protect_multiple(original_tensors)

    # Step 2: Generate videos
    print("Generating videos...")
    original_videos = i2v_model.generate_video(original_tensors)
    protected_videos = i2v_model.generate_video(protected_tensors)
    
    # Step 3: Save images and videos first
    print("保存图片和视频...")
    video_paths = save_images_and_videos(original_tensors, protected_tensors, original_videos, protected_videos, save_path, i2v_model)
    
    # Step 4: Evaluate using saved files to reduce IO pressure
    print("Evaluating results...")
    image_results = evaluate_images(original_tensors, protected_tensors, metrics, protection_method)
    video_results = evaluate_videos(metrics, video_paths, original_videos, protected_videos)
    
    # Combine and aggregate results
    all_results = {**image_results, **video_results}
    
    # Results are now already aggregated, so we don't need to compute averages
    # Just rename the keys to maintain compatibility
    aggregated = {}
    for key, value in all_results.items():
        # Skip timing-related keys as they are handled separately
        if key not in ['total_protection_time', 'total_images_processed', 'images_per_second']:
            aggregated[key] = value
    
    return {
        'method': protection_method.__class__.__name__,
        'time': {
            'total_protection_time': all_results.get('total_protection_time'),
            'total_images_processed': all_results.get('total_images_processed'),
            'images_per_second': all_results.get('images_per_second')
        },
        'aggregated': aggregated
    } 
    

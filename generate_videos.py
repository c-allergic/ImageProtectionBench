#!/usr/bin/env python3
"""
视频生成脚本

专门用于承接protect_images.py的运行结果，对原图和保护后图片生成视频并保存。
使用现有的I2V模块，参照benchmark.py的初始化方式。
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import argparse
import datetime
import json
import torch
import time
from PIL import Image
from pathlib import Path
from diffusers.utils import export_to_video

from models.i2v import WANModel, LTXModel, SkyreelModel
from data import transform, pt_to_pil


def load_image_pairs(input_dir):
    """从输入目录加载图片对和对应的prompt"""
    input_path = Path(input_dir)
    images_dir = input_path / "images"
    
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    
    # 查找所有原始图片
    original_files = sorted(images_dir.glob("original_*.png"))
    
    # 尝试加载prompt信息
    prompts = load_prompts(input_path)
    
    # 匹配图片对
    image_pairs = []
    for orig_file in original_files:
        # 提取编号
        orig_num = orig_file.stem.split('_')[1]
        prot_file = images_dir / f"protected_{orig_num}.png"
        
        if prot_file.exists():
            # 获取对应的prompt
            prompt = prompts.get(int(orig_num), "") if prompts else ""
            
            image_pairs.append({
                'original': str(orig_file),
                'protected': str(prot_file),
                'index': orig_num,
                'prompt': prompt
            })
        else:
            print(f"Warning: Protected image not found for {orig_file}")
    
    print(f"Found {len(image_pairs)} image pairs")
    return image_pairs


def load_prompts(input_path):
    """从data/descriptions目录加载prompt信息，参照experiment.py的方式"""
    prompts = {}
    
    # 从args.json获取数据集信息
    args_file = input_path / "results" / "args.json"
    if not args_file.exists():
        print("No args.json found, will use empty prompts for video generation")
        return prompts
    
    try:
        with open(args_file, 'r') as f:
            args_data = json.load(f)
        
        dataset = args_data.get('dataset', 'Flickr30k')
        data_path = args_data.get('data_path', './data')
        num_samples = args_data.get('num_samples', 150)
        
        # 构建prompt文件路径
        prompt_file = Path(data_path) / "descriptions" / f"{dataset.lower()}_descriptions.json"
        
        if prompt_file.exists():
            with open(prompt_file, 'r') as f:
                prompts_data = json.load(f)
            
            # 参照experiment.py的方式提取prompt
            if prompts_data and "data" in prompts_data:
                # 提取描述列表，限制到num_samples
                for i, item in enumerate(prompts_data["data"][:num_samples]):
                    if isinstance(item, dict) and "malicious_prompt" in item:
                        prompts[i] = item["malicious_prompt"]
                
                print(f"Loaded {len(prompts)} prompts from {prompt_file}")
            else:
                print("No valid prompt data found in the file")
        else:
            print(f"Prompt file not found: {prompt_file}")
            
    except Exception as e:
        print(f"Failed to load prompts: {e}")
    
    if not prompts:
        print("No prompts found, will use empty prompts for video generation")
    
    return prompts


def generate_videos_batch(image_pairs, i2v_model, videos_dir):
    """批量生成视频"""
    total_pairs = len(image_pairs)
    print(f"开始为 {total_pairs} 对图片生成视频")
    
    total_time = 0.0
    successful_videos = 0
    
    for i, pair in enumerate(image_pairs):
        print(f"\n=== 处理图片对 {i+1}/{total_pairs} (index: {pair['index']}) ===")
        
        # 生成原图视频
        orig_video_path = videos_dir / f"original_{pair['index']}.mp4"
        start_time = time.time()
        try:
            # 加载并转换图片
            orig_img = Image.open(pair['original']).convert('RGB')
            orig_tensor = transform(orig_img).unsqueeze(0).to(i2v_model.device)
            
            # 获取对应的prompt
            prompt = pair.get('prompt', '')
            if prompt:
                print(f"  使用prompt: {prompt}")
            
            # 生成视频
            video_tensor = i2v_model.generate_video(orig_tensor, prompt=prompt)
            video_frames = [pt_to_pil(frame) for frame in video_tensor[0]]
            
            # 保存视频
            export_to_video(video_frames, str(orig_video_path))
            print(f"Video saved: {orig_video_path}")
            
            orig_time = time.time() - start_time
            print(f"原图视频生成完成: {len(video_frames)} 帧, 耗时 {orig_time:.2f}秒")
            successful_videos += 1
        except Exception as e:
            print(f"原图视频生成失败: {e}")
            orig_time = 0
        
        # 生成保护后图片视频
        prot_video_path = videos_dir / f"protected_{pair['index']}.mp4"
        start_time = time.time()
        try:
            # 加载并转换图片
            prot_img = Image.open(pair['protected']).convert('RGB')
            prot_tensor = transform(prot_img).unsqueeze(0).to(i2v_model.device)
            
            # 获取对应的prompt（与原图使用相同的prompt）
            prompt = pair.get('prompt', '')
            
            # 生成视频
            print(f"  使用prompt: {prompt}")
            video_tensor = i2v_model.generate_video(prot_tensor, prompt=prompt)
            video_frames = [pt_to_pil(frame) for frame in video_tensor[0]]
            
            # 保存视频
            export_to_video(video_frames, str(prot_video_path))
            print(f"Video saved: {prot_video_path}")
            
            prot_time = time.time() - start_time
            print(f"保护后视频生成完成: {len(video_frames)} 帧, 耗时 {prot_time:.2f}秒")
            successful_videos += 1
        except Exception as e:
            print(f"保护后视频生成失败: {e}")
            prot_time = 0
        
        total_time += orig_time + prot_time
        
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return {
        'total_pairs': total_pairs,
        'successful_videos': successful_videos,
        'total_time': total_time,
        'avg_time_per_video': total_time / successful_videos if successful_videos > 0 else 0
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="视频生成脚本")
    
    # 输入输出参数
    parser.add_argument('--input_dir', type=str, required=True,
                       help="包含图片对的输入目录")
    parser.add_argument('--output_dir', type=str, default=None,
                       help="输出目录，默认为input_dir")
    
    # I2V模型参数 - 参照benchmark.py
    parser.add_argument('--i2v_model', type=str, default="LTX", 
                       choices=["LTX", "WAN", "Skyreel"],
                       help="I2V模型类型")
    
    # 系统参数
    parser.add_argument('--device', type=str, default="cuda",
                       help="计算设备")
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    print("=" * 60)
    print("视频生成脚本")
    print("=" * 60)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"I2V模型: {args.i2v_model}")
    print(f"计算设备: {args.device}")
    print()
    
    # 设置设备
    device = args.device if torch.cuda.is_available() else "cpu"
    args.device = device
    
    # 创建输出目录
    output_path = Path(args.output_dir)
    videos_dir = output_path / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存参数
    args_dict = vars(args)
    args_dict['timestamp'] = datetime.datetime.now().isoformat()
    with open(output_path / "video_generation_args.json", "w") as f:
        json.dump(args_dict, f, indent=4)
    
    # 加载图片对
    print("加载图片对...")
    image_pairs = load_image_pairs(args.input_dir)
    
    if not image_pairs:
        print("未找到图片对，退出")
        return
    
    # 初始化I2V模型 - 参照benchmark.py
    print(f"初始化I2V模型 {args.i2v_model}...")
    if args.i2v_model == "LTX":
        i2v_model = LTXModel(device=device)
    elif args.i2v_model == "WAN":
        i2v_model = WANModel(device=device)
    elif args.i2v_model == "Skyreel":
        i2v_model = SkyreelModel(device=device)
    else:
        raise ValueError(f"Unknown I2V model: {args.i2v_model}")
    
    # 生成视频
    print("\n开始生成视频...")
    result = generate_videos_batch(image_pairs, i2v_model, videos_dir)
    
    # 保存结果统计
    result['timestamp'] = datetime.datetime.now().isoformat()
    
    result_file = output_path / "video_generation_results.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=4)
    
    print("\n" + "=" * 60)
    print("视频生成完成!")
    print(f"处理图片对: {result['total_pairs']}")
    print(f"成功生成视频: {result['successful_videos']}")
    print(f"总耗时: {result['total_time']:.2f}秒")
    print(f"平均每视频耗时: {result['avg_time_per_video']:.2f}秒")
    print(f"视频保存目录: {videos_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

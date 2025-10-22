#!/usr/bin/env python3
"""
视频生成脚本

专门用于承接protect_images.py的运行结果，对原图和保护后图片生成视频并保存。
使用现有的I2V模块，参照benchmark.py的初始化方式。
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import argparse
import datetime
import json
import torch
import time
from PIL import Image
from pathlib import Path
from diffusers.utils import export_to_video

from i2v import WAN22Model, LTXModel, SkyreelModel
from data import transform, pt_to_pil


def load_image_pairs(input_dir):
    """从输入目录加载图片对和对应的prompt，包括攻击图片（如果存在）"""
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
    attacked_count = 0
    
    for orig_file in original_files:
        # 提取编号
        orig_num = orig_file.stem.split('_')[1]
        prot_file = images_dir / f"protected_{orig_num}.png"
        
        if prot_file.exists():
            # 获取对应的prompt
            prompt = prompts.get(int(orig_num), "") if prompts else ""
            
            pair_data = {
                'original': str(orig_file),
                'protected': str(prot_file),
                'index': orig_num,
                'prompt': prompt
            }
            
            # 检查是否存在攻击后的图片
            attack_file = images_dir / f"attacked_{orig_num}.png"
            if attack_file.exists():
                pair_data['attacked'] = str(attack_file)
                attacked_count += 1
            
            image_pairs.append(pair_data)
        else:
            print(f"Warning: Protected image not found for {orig_file}")
    
    print(f"Found {len(image_pairs)} image pairs")
    if attacked_count > 0:
        print(f"Found {attacked_count} attacked images")
    else:
        print("No attacked images found")
    
    return image_pairs


def load_prompts(input_path):
    """从data/descriptions目录加载prompt信息，同时加载恶意和正常prompt"""
    prompts = {'malicious': {}, 'normal': {}}
    
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
        
        # 直接尝试所有可能的描述文件名
        possible_files = [
            Path(data_path) / "descriptions" / f"{dataset.lower()}_descriptions_exp.json",
            Path(data_path) / "descriptions" / f"{dataset.lower()}_descriptions.json"
        ]
        
        prompt_file = None
        for file_path in possible_files:
            if file_path.exists():
                prompt_file = file_path
                break
        
        if prompt_file:
            with open(prompt_file, 'r') as f:
                prompts_data = json.load(f)
            
            # 提取恶意和正常prompt
            if prompts_data and "data" in prompts_data:
                # 提取描述列表，限制到num_samples
                for i, item in enumerate(prompts_data["data"][:num_samples]):
                    if isinstance(item, dict):
                        if "malicious_prompt" in item:
                            prompts['malicious'][i] = item["malicious_prompt"]
                        if "normal_prompt" in item:
                            prompts['normal'][i] = item["normal_prompt"]
                
                print(f"加载了 {len(prompts['malicious'])} 个恶意prompt和 {len(prompts['normal'])} 个正常prompt")
            else:
                print("文件中没有找到有效的prompt数据")
        else:
            print(f"未找到任何描述文件")
            
    except Exception as e:
        print(f"加载prompt失败: {e}")
    
    if not prompts['malicious'] and not prompts['normal']:
        print("未找到任何prompt，将使用空prompt进行视频生成")
    
    return prompts


def generate_videos_batch(image_pairs, i2v_model, videos_dir, prompts=None):
    """批量生成视频，使用恶意和正常prompt分别生成视频"""
    total_pairs = len(image_pairs)
    print(f"开始为 {total_pairs} 对图片生成视频")
    
    total_time = 0.0
    successful_videos = 0
    successful_attacked_videos = 0
    
    # 如果没有提供prompts，使用空字典
    if prompts is None:
        prompts = {'malicious': {}, 'normal': {}}
    
    for i, pair in enumerate(image_pairs):
        idx = int(pair['index'])
        print(f"\n=== 处理图片对 {i+1}/{total_pairs} (index: {idx}) ===")
        
        # 获取恶意和正常prompt
        malicious_prompt = prompts['malicious'].get(idx, '')
        normal_prompt = prompts['normal'].get(idx, '')
        
        # 记录生成的视频时间
        image_total_time = 0.0
        
        # 为原始图片生成视频（使用恶意prompt）
        if malicious_prompt:
            orig_mal_video_path = videos_dir / f"original_{idx}_malicious.mp4"
            if orig_mal_video_path.exists():
                print(f"  原图恶意视频已存在，跳过: {orig_mal_video_path}")
                successful_videos += 1
            else:
                start_time = time.time()
                try:
                    # 加载并转换图片
                    orig_img = Image.open(pair['original']).convert('RGB')
                    orig_tensor = transform(orig_img).unsqueeze(0).to(i2v_model.device)
                    
                    print(f"  使用恶意prompt: {malicious_prompt}")
                    # 生成视频
                    video_tensor = i2v_model.generate_video(orig_tensor, prompt=malicious_prompt)
                    video_frames = [pt_to_pil(frame) for frame in video_tensor[0]]
                    
                    # 保存视频
                    export_to_video(video_frames, str(orig_mal_video_path))
                    print(f"原图恶意视频已保存: {orig_mal_video_path}")
                    
                    mal_time = time.time() - start_time
                    print(f"原图恶意视频生成完成: {len(video_frames)} 帧, 耗时 {mal_time:.2f}秒")
                    successful_videos += 1
                    image_total_time += mal_time
                except Exception as e:
                    print(f"原图恶意视频生成失败: {e}")
        
        # 为原始图片生成视频（使用正常prompt）
        if normal_prompt:
            orig_norm_video_path = videos_dir / f"original_{idx}_normal.mp4"
            if orig_norm_video_path.exists():
                print(f"  原图正常视频已存在，跳过: {orig_norm_video_path}")
                successful_videos += 1
            else:
                start_time = time.time()
                try:
                    # 加载并转换图片（如果之前已经加载过，可以重用）
                    if 'orig_tensor' not in locals():
                        orig_img = Image.open(pair['original']).convert('RGB')
                        orig_tensor = transform(orig_img).unsqueeze(0).to(i2v_model.device)
                    
                    print(f"  使用正常prompt: {normal_prompt}")
                    # 生成视频
                    video_tensor = i2v_model.generate_video(orig_tensor, prompt=normal_prompt)
                    video_frames = [pt_to_pil(frame) for frame in video_tensor[0]]
                    
                    # 保存视频
                    export_to_video(video_frames, str(orig_norm_video_path))
                    print(f"原图正常视频已保存: {orig_norm_video_path}")
                    
                    norm_time = time.time() - start_time
                    print(f"原图正常视频生成完成: {len(video_frames)} 帧, 耗时 {norm_time:.2f}秒")
                    successful_videos += 1
                    image_total_time += norm_time
                except Exception as e:
                    print(f"原图正常视频生成失败: {e}")
        
        # 为保护后图片生成视频（使用恶意prompt）
        if malicious_prompt:
            prot_mal_video_path = videos_dir / f"protected_{idx}_malicious.mp4"
            if prot_mal_video_path.exists():
                print(f"  保护图恶意视频已存在，跳过: {prot_mal_video_path}")
                successful_videos += 1
            else:
                start_time = time.time()
                try:
                    # 加载并转换图片
                    prot_img = Image.open(pair['protected']).convert('RGB')
                    prot_tensor = transform(prot_img).unsqueeze(0).to(i2v_model.device)
                    
                    print(f"  使用恶意prompt: {malicious_prompt}")
                    # 生成视频
                    video_tensor = i2v_model.generate_video(prot_tensor, prompt=malicious_prompt)
                    video_frames = [pt_to_pil(frame) for frame in video_tensor[0]]
                    
                    # 保存视频
                    export_to_video(video_frames, str(prot_mal_video_path))
                    print(f"保护图恶意视频已保存: {prot_mal_video_path}")
                    
                    mal_time = time.time() - start_time
                    print(f"保护图恶意视频生成完成: {len(video_frames)} 帧, 耗时 {mal_time:.2f}秒")
                    successful_videos += 1
                    image_total_time += mal_time
                except Exception as e:
                    print(f"保护图恶意视频生成失败: {e}")
        
        # 为保护后图片生成视频（使用正常prompt）
        if normal_prompt:
            prot_norm_video_path = videos_dir / f"protected_{idx}_normal.mp4"
            if prot_norm_video_path.exists():
                print(f"  保护图正常视频已存在，跳过: {prot_norm_video_path}")
                successful_videos += 1
            else:
                start_time = time.time()
                try:
                    # 加载并转换图片（如果之前已经加载过，可以重用）
                    if 'prot_tensor' not in locals():
                        prot_img = Image.open(pair['protected']).convert('RGB')
                        prot_tensor = transform(prot_img).unsqueeze(0).to(i2v_model.device)
                    
                    print(f"  使用正常prompt: {normal_prompt}")
                    # 生成视频
                    video_tensor = i2v_model.generate_video(prot_tensor, prompt=normal_prompt)
                    video_frames = [pt_to_pil(frame) for frame in video_tensor[0]]
                    
                    # 保存视频
                    export_to_video(video_frames, str(prot_norm_video_path))
                    print(f"保护图正常视频已保存: {prot_norm_video_path}")
                    
                    norm_time = time.time() - start_time
                    print(f"保护图正常视频生成完成: {len(video_frames)} 帧, 耗时 {norm_time:.2f}秒")
                    successful_videos += 1
                    image_total_time += norm_time
                except Exception as e:
                    print(f"保护图正常视频生成失败: {e}")
        
        # 生成攻击后图片视频（如果存在攻击图片）
        if 'attacked' in pair:
            # 为攻击后图片生成视频（使用恶意prompt）
            if malicious_prompt:
                attack_mal_video_path = videos_dir / f"attacked_{idx}_malicious.mp4"
                if attack_mal_video_path.exists():
                    print(f"  攻击图恶意视频已存在，跳过: {attack_mal_video_path}")
                    successful_videos += 1
                    successful_attacked_videos += 1
                else:
                    start_time = time.time()
                    try:
                        # 加载并转换攻击后的图片
                        attack_img = Image.open(pair['attacked']).convert('RGB')
                        attack_tensor = transform(attack_img).unsqueeze(0).to(i2v_model.device)
                        
                        print(f"  攻击后图片使用恶意prompt: {malicious_prompt}")
                        # 生成视频
                        video_tensor = i2v_model.generate_video(attack_tensor, prompt=malicious_prompt)
                        video_frames = [pt_to_pil(frame) for frame in video_tensor[0]]
                        
                        # 保存视频
                        export_to_video(video_frames, str(attack_mal_video_path))
                        print(f"攻击图恶意视频已保存: {attack_mal_video_path}")
                        
                        mal_time = time.time() - start_time
                        print(f"攻击图恶意视频生成完成: {len(video_frames)} 帧, 耗时 {mal_time:.2f}秒")
                        successful_videos += 1
                        successful_attacked_videos += 1
                        image_total_time += mal_time
                    except Exception as e:
                        print(f"攻击图恶意视频生成失败: {e}")
            
            # 为攻击后图片生成视频（使用正常prompt）
            if normal_prompt:
                attack_norm_video_path = videos_dir / f"attacked_{idx}_normal.mp4"
                if attack_norm_video_path.exists():
                    print(f"  攻击图正常视频已存在，跳过: {attack_norm_video_path}")
                    successful_videos += 1
                    successful_attacked_videos += 1
                else:
                    start_time = time.time()
                    try:
                        # 加载并转换攻击后的图片（如果之前已经加载过，可以重用）
                        if 'attack_tensor' not in locals():
                            attack_img = Image.open(pair['attacked']).convert('RGB')
                            attack_tensor = transform(attack_img).unsqueeze(0).to(i2v_model.device)
                        
                        print(f"  攻击后图片使用正常prompt: {normal_prompt}")
                        # 生成视频
                        video_tensor = i2v_model.generate_video(attack_tensor, prompt=normal_prompt)
                        video_frames = [pt_to_pil(frame) for frame in video_tensor[0]]
                        
                        # 保存视频
                        export_to_video(video_frames, str(attack_norm_video_path))
                        print(f"攻击图正常视频已保存: {attack_norm_video_path}")
                        
                        norm_time = time.time() - start_time
                        print(f"攻击图正常视频生成完成: {len(video_frames)} 帧, 耗时 {norm_time:.2f}秒")
                        successful_videos += 1
                        successful_attacked_videos += 1
                        image_total_time += norm_time
                    except Exception as e:
                        print(f"攻击图正常视频生成失败: {e}")
        
        total_time += image_total_time
        
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return {
        'total_pairs': total_pairs,
        'successful_videos': successful_videos,
        'successful_attacked_videos': successful_attacked_videos,
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
    parser.add_argument('--i2v_model', type=str, default="Skyreel", 
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
    
    # 加载prompts
    print("加载prompts...")
    prompts = load_prompts(Path(args.input_dir))
    
    # 初始化I2V模型 - 参照benchmark.py
    print(f"初始化I2V模型 {args.i2v_model}...")
    if args.i2v_model == "LTX":
        i2v_model = LTXModel(device=device)
    elif args.i2v_model == "WAN":
        i2v_model = WAN22Model(device=device)
    elif args.i2v_model == "Skyreel":
        i2v_model = SkyreelModel(device=device)
    else:
        raise ValueError(f"Unknown I2V model: {args.i2v_model}")
    
    # 生成视频
    print("\n开始生成视频...")
    result = generate_videos_batch(image_pairs, i2v_model, videos_dir, prompts)
    
    # 保存结果统计
    result['timestamp'] = datetime.datetime.now().isoformat()
    
    result_file = output_path / "video_generation_results.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=4)
    
    print("\n" + "=" * 60)
    print("视频生成完成!")
    print(f"处理图片对: {result['total_pairs']}")
    print(f"成功生成视频: {result['successful_videos']}")
    if result['successful_attacked_videos'] > 0:
        print(f"成功生成攻击视频: {result['successful_attacked_videos']}")
    print(f"总耗时: {result['total_time']:.2f}秒")
    print(f"平均每视频耗时: {result['avg_time_per_video']:.2f}秒")
    print(f"视频保存目录: {videos_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

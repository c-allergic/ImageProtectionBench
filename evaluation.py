#!/usr/bin/env python3
"""
Comprehensive Evaluation Script

Reads data from videos and images subfolders in the specified folder, evaluates VBench scores,
image quality and attack effectiveness, and saves results to benchmark_results.json in the results subfolder.

Usage:
    python evaluation.py --input_dir /path/to/experiment/folder
    python evaluation.py --input_dir /path/to/experiment/folder --method_name "EditShield"
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import argparse
import glob
from typing import Dict, List, Optional
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# Import necessary modules
from metrics.video_quality import VBenchMetric
from metrics import PSNRMetric, SSIMMetric, CLIPVideoScoreMetric, LPIPSMetric, CLIPVideoTextScoreMetric
from data import transform, pt_to_pil
from vbench.utils import load_video


def find_video_pairs(videos_dir: str) -> List[Dict[str, str]]:
    """Find video pairs from videos directory"""
    if not os.path.exists(videos_dir):
        print(f"Error: videos directory does not exist: {videos_dir}")
        return []
    
    video_pairs = []
    
    # Find all original video files
    original_pattern = os.path.join(videos_dir, "original_*.mp4")
    original_files = glob.glob(original_pattern)
    original_files.sort()
    
    print(f"Found {len(original_files)} original video files")
    
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
        else:
            print(f"  Warning: Protected video not found: {prot_path}")
    
    # Check for attacked videos
    attacked_pattern = os.path.join(videos_dir, "attacked_*.mp4")
    attacked_files = glob.glob(attacked_pattern)
    
    if attacked_files:
        print(f"Found {len(attacked_files)} attacked video files")
        for pair in video_pairs:
            video_id = pair['video_id']
            attacked_path = os.path.join(videos_dir, f"attacked_{video_id}.mp4")
            if os.path.exists(attacked_path):
                pair['attacked_path'] = attacked_path
    
    print(f"Total found {len(video_pairs)} valid video pairs")
    return video_pairs


def load_prompts_from_description(dataset_name: str, data_path: str) -> Dict:
    """Load prompts from description file"""
    print(f"Loading description file for dataset {dataset_name}...")
    
    file_path = os.path.join(data_path, "descriptions", f"{dataset_name.lower()}_descriptions_exp.json")

    if not os.path.exists(file_path):
        print(f"Warning: Description file not found for dataset {dataset_name}")
        print(f"Tried path: {file_path}")
        return {}
    
    print(f"Found description file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts_data = json.load(f)
    
    prompts = {'malicious': {}, 'normal': {}}
    
    if "data" in prompts_data:
        for item in prompts_data["data"]:
            if isinstance(item, dict) and "image_id" in item:
                idx = item["image_id"]
                if "malicious_prompt" in item:
                    prompts['malicious'][idx] = item["malicious_prompt"]
                if "normal_prompt" in item:
                    prompts['normal'][idx] = item["normal_prompt"]
        
        print(f"Successfully loaded {len(prompts['malicious'])} malicious prompts and {len(prompts['normal'])} normal prompts")
    else:
        print("Warning: Description file format is incorrect")
    
    return prompts


def find_image_pairs(images_dir: str) -> List[Dict[str, str]]:
    """Find image pairs from images directory"""
    if not os.path.exists(images_dir):
        print(f"Error: images directory does not exist: {images_dir}")
        return []
    
    image_pairs = []
    
    # Find all original image files
    original_pattern = os.path.join(images_dir, "original_*.png")
    original_files = glob.glob(original_pattern)
    original_files.sort()
    
    print(f"Found {len(original_files)} original image files")
    
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
        else:
            print(f"  Warning: Protected image not found: {prot_path}")
    
    # Check for attacked images
    attacked_pattern = os.path.join(images_dir, "attacked_*.png")
    attacked_files = glob.glob(attacked_pattern)
    
    if attacked_files:
        print(f"Found {len(attacked_files)} attacked image files")
        for pair in image_pairs:
            image_id = pair['image_id']
            attacked_path = os.path.join(images_dir, f"attacked_{image_id}.png")
            if os.path.exists(attacked_path):
                pair['attacked_path'] = attacked_path
    
    print(f"Total found {len(image_pairs)} valid image pairs")
    return image_pairs


def load_images_as_tensors(image_pairs: List[Dict[str, str]], device: str) -> tuple:
    """Load image pairs as tensors"""
    original_tensors = []
    protected_tensors = []
    attacked_tensors = []
    
    for pair in image_pairs:
        # Load original image
        orig_img = Image.open(pair['original_path']).convert('RGB')
        orig_tensor = transform(orig_img).to(device)
        original_tensors.append(orig_tensor)
        
        # Load protected image
        prot_img = Image.open(pair['protected_path']).convert('RGB')
        prot_tensor = transform(prot_img).to(device)
        protected_tensors.append(prot_tensor)
        
        # Load attacked image (if exists)
        if 'attacked_path' in pair:
            attack_img = Image.open(pair['attacked_path']).convert('RGB')
            attack_tensor = transform(attack_img).to(device)
            attacked_tensors.append(attack_tensor)
    
    original_tensors = torch.stack(original_tensors)
    protected_tensors = torch.stack(protected_tensors)
    attacked_tensors = torch.stack(attacked_tensors) if attacked_tensors else None
    
    return original_tensors, protected_tensors, attacked_tensors


def load_videos_as_tensors(video_pairs: List[Dict[str, str]], device: str) -> tuple:
    """Load videos from files as tensors, following experiment.py approach"""
    original_videos = []
    protected_videos = []
    attacked_videos = []
    
    print("Loading tensors from video files...")
    
    for i, pair in enumerate(video_pairs):
        
        # Load original video
        orig_video_tensor = load_video(pair['original_path'], return_tensor=True)
        # Normalization to [0,1]
        orig_video_tensor = orig_video_tensor.float() / 255.0
        original_videos.append(orig_video_tensor)
        
        # Load protected video
        prot_video_tensor = load_video(pair['protected_path'], return_tensor=True)
        # Normalization to [0,1]
        prot_video_tensor = prot_video_tensor.float() / 255.0
        protected_videos.append(prot_video_tensor)
        
        # Load attacked video (if exists)
        if 'attacked_path' in pair:
            attack_video_tensor = load_video(pair['attacked_path'], return_tensor=True)
            # Normalization to [0,1]
            attack_video_tensor = attack_video_tensor.float() / 255.0
            attacked_videos.append(attack_video_tensor)
    
    if not original_videos:
        print("Error: No videos successfully loaded")
        return None, None, None
    
    # Stack as batch tensors
    original_videos = torch.stack(original_videos).to(device)
    protected_videos = torch.stack(protected_videos).to(device)
    attacked_videos = torch.stack(attacked_videos).to(device) if attacked_videos else None
    
    print(f"Successfully loaded {len(original_videos)} video pairs")
    print(f"Original videos tensor shape: {original_videos.shape}")
    print(f"Protected videos tensor shape: {protected_videos.shape}")
    if attacked_videos is not None:
        print(f"Attacked videos tensor shape: {attacked_videos.shape}")
    
    return original_videos, protected_videos, attacked_videos


def evaluate_images(original_tensors, protected_tensors, metrics, attacked_tensors=None):
    """Evaluate image quality, following experiment.py"""
    results = {}    
    
    for metric_name, metric in metrics.items():
        if metric_name in ['psnr', 'ssim', 'lpips']:
            # Original vs protected image metrics
            metric_result = metric.compute_multiple(original_tensors, protected_tensors)
            if metric_result:
                for key, value in metric_result.items():
                    results[f'protected_{key}'] = value
            
            # If attacked images exist, compute original vs attacked image metrics
            if attacked_tensors is not None:
                attack_metric_result = metric.compute_multiple(original_tensors, attacked_tensors)
                if attack_metric_result:
                    for key, value in attack_metric_result.items():
                        results[f'attacked_{key}'] = value
    
    return results


def evaluate_videos(metrics, video_paths=None, original_videos=None, protected_videos=None, attacked_videos=None, compute_clip_bounds=True, prompts_dict=None):
    """Evaluate video quality and attack effectiveness, following experiment.py"""
    results = {}
    print(f"Starting video evaluation, number of video_paths: {len(video_paths) if video_paths else 0}")
    
    for metric_name, metric in metrics.items():
        print(f"Processing metric: {metric_name}")
        
        if metric_name == 'clip' and original_videos is not None and protected_videos is not None:
            print("Running CLIP evaluation on video tensors...")
            # Original vs protected video CLIP evaluation
            clip_result = metric.compute_multiple(original_videos, protected_videos)
            if clip_result:
                for key, value in clip_result.items():
                    results[f'protected_{key}'] = value
                print(f"Protected video CLIP evaluation completed, obtained {len(clip_result)} results")
            
            # Only compute CLIP theoretical upper and lower bounds in the first batch
            if compute_clip_bounds:
                print("Computing CLIP upper bound...")
                print(f"  Using video tensor shape: {original_videos.shape}")
                upper_bound = metric.compute_upper_bound(original_videos, sample_size=10)
                results['clip_upper_bound'] = upper_bound
                print(f"CLIP upper bound: {upper_bound:.4f}")
                
                print("Computing CLIP lower bound...")
                print(f"  Using video tensor shape: {original_videos.shape}")
                lower_bound = metric.compute_lower_bound(original_videos, sample_size=10)
                results['clip_lower_bound'] = lower_bound
                print(f"CLIP lower bound: {lower_bound:.4f}")
                
            
            # If attacked videos exist, compute original vs attacked video CLIP evaluation  
            if attacked_videos is not None:
                attack_clip_result = metric.compute_multiple(original_videos, attacked_videos)
                if attack_clip_result:
                    for key, value in attack_clip_result.items():
                        results[f'attacked_{key}'] = value
                    print(f"Attacked video CLIP evaluation completed, obtained {len(attack_clip_result)} results")

        
        elif metric_name == 'clip_text' and prompts_dict is not None and original_videos is not None and protected_videos is not None:
            print("Running CLIP Video-Text evaluation...")
            
            # Check if prompts_dict is empty
            if not prompts_dict:
                print("  Warning: prompts_dict is empty, skipping CLIP Video-Text evaluation")
                continue
            
            # Extract corresponding prompt for each video
            prompts_list = []
            for video_pair in video_paths:
                video_id_str = video_pair['video_id']
                
                # Parse video_id to extract image_id and prompt type
                # Format: "000_malicious" or "000_normal"
                if '_' in video_id_str:
                    parts = video_id_str.rsplit('_', 1)
                    image_id_str = parts[0]
                    prompt_type = parts[1]  # "malicious" or "normal"
                    
                    # Convert to integer (remove leading zeros)
                    image_id = int(image_id_str)
                    
                    # Get prompt from prompts_dict
                    if prompt_type in prompts_dict and image_id in prompts_dict[prompt_type]:
                        prompt = prompts_dict[prompt_type][image_id]
                    else:
                        print(f"  Warning: Prompt not found for video_id {video_id_str} (image_id={image_id}, type={prompt_type}), using empty string")
                        prompt = ""
                else:
                    print(f"  Warning: video_id format incorrect: {video_id_str}, using empty string")
                    prompt = ""
                
                prompts_list.append(prompt)
            
            # Original videos
            print("  Evaluating similarity between original videos and prompts...")
            orig_result = metric.compute_multiple(original_videos, prompts_list)
            if orig_result:
                for key, value in orig_result.items():
                    results[f'original_{key}'] = value
                print(f"  Original video evaluation completed: {orig_result['clip_video_text_score']:.4f}")
            
            # Protected videos
            print("  Evaluating similarity between protected videos and prompts...")
            prot_result = metric.compute_multiple(protected_videos, prompts_list)
            if prot_result:
                for key, value in prot_result.items():
                    results[f'protected_{key}'] = value
                print(f"  Protected video evaluation completed: {prot_result['clip_video_text_score']:.4f}")
            
            # Attacked videos
            if attacked_videos is not None:
                print("  Evaluating similarity between attacked videos and prompts...")
                attack_result = metric.compute_multiple(attacked_videos, prompts_list)
                if attack_result:
                    for key, value in attack_result.items():
                        results[f'attacked_{key}'] = value
                    print(f"  Attacked video evaluation completed: {attack_result['clip_video_text_score']:.4f}")
            
            # Compute upper and lower bounds only in the first batch
            if compute_clip_bounds:
                print("  Computing CLIP Video-Text upper bound...")
                text_upper_bound = metric.compute_upper_bound(original_videos, prompts_list, sample_size=10)
                results['clip_video_text_upper_bound'] = text_upper_bound
                print(f"  CLIP Video-Text upper bound: {text_upper_bound:.4f}")
                
                print("  Computing CLIP Video-Text lower bound...")
                text_lower_bound = metric.compute_lower_bound(original_videos, prompts_list, sample_size=10)
                results['clip_video_text_lower_bound'] = text_lower_bound
                print(f"  CLIP Video-Text lower bound: {text_lower_bound:.4f}")
            
            print(f"CLIP Video-Text evaluation completed")
                
                
        elif metric_name == 'vbench' and video_paths is not None:
            print("Running VBench evaluation using saved video files...")
            metric_result = metric.compute_multiple(video_paths)
            if metric_result:
                for key, value in metric_result.items():
                    results[f'{key}'] = value
                print(f"VBench evaluation completed, obtained {len(metric_result)} results")
    
    print(f"Video evaluation completed, total obtained {len(results)} results")
    return results


def setup_metrics(device: str = "cuda") -> Dict:
    """Setup all evaluation metrics"""
    print("Initializing evaluation metrics...")
    
    metrics = {
        'psnr': PSNRMetric(device=device),
        'ssim': SSIMMetric(device=device),
        'lpips': LPIPSMetric(device=device),
        'clip': CLIPVideoScoreMetric(device=device),
        'clip_text': CLIPVideoTextScoreMetric(device=device),
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
    
    print("Evaluation metrics initialization completed")
    return metrics


def save_results(results: Dict, output_path: str, method_name: str = "Unknown"):
    """Save evaluation results to JSON file, following experiment.py format"""
    final_results = {
        "method": method_name,
        "aggregated": results
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Comprehensive Evaluation Script")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Experiment folder path, containing videos and images subfolders")
    parser.add_argument("--method_name", type=str, default="Unknown",
                       help="Protection method name (default: Unknown)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Computing device (default: cuda)")
    parser.add_argument("--output_filename", type=str, default="benchmark_results.json",
                       help="Output filename (default: benchmark_results.json)")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Batch size, number of videos per batch (default: 10)")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Dataset name for loading description file (e.g., AFHQ-v2)")
    parser.add_argument("--data_path", type=str, default="./data",
                       help="Data path containing descriptions subfolder (default: ./data)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Comprehensive Evaluation Script")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Method name: {args.method_name}")
    print(f"Computing device: {args.device}")
    print(f"Output filename: {args.output_filename}")
    print(f"Batch size: {args.batch_size}")
    print(f"Dataset: {args.dataset if args.dataset else 'Not specified'}")
    print(f"Data path: {args.data_path}")
    print()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    # Load prompts (if dataset is specified)
    prompts_dict = {}
    if args.dataset:
        prompts_dict = load_prompts_from_description(args.dataset, args.data_path)
    else:
        print("Dataset not specified, will skip CLIP Video-Text evaluation")
    
    # Build paths
    videos_dir = os.path.join(args.input_dir, "videos")
    images_dir = os.path.join(args.input_dir, "images")
    results_dir = os.path.join(args.input_dir, "results")
    output_path = os.path.join(results_dir, args.output_filename)
    
    print(f"Videos directory: {videos_dir}")
    print(f"Images directory: {images_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Output path: {output_path}")
    print()
    
    # Step 1: Find data pairs
    print("Step 1: Finding data pairs...")
    video_pairs = find_video_pairs(videos_dir)
    image_pairs = find_image_pairs(images_dir)
    
    if not video_pairs and not image_pairs:
        print("Error: No valid data pairs found")
        return
    
    print()
    
    # Step 2: Setup evaluation metrics
    print("Step 2: Setting up evaluation metrics...")
    metrics = setup_metrics(args.device)

    print()
    
    # Step 3: Evaluate image quality
    all_results = {}
    if image_pairs:
        print("Step 3: Evaluating image quality...")
        original_tensors, protected_tensors, attacked_tensors = load_images_as_tensors(image_pairs, args.device)
        image_metrics = {k: v for k, v in metrics.items() if k in ['psnr', 'ssim', 'lpips']}
        image_results = evaluate_images(original_tensors, protected_tensors, image_metrics, attacked_tensors)
        all_results.update(image_results)
        print("Image quality evaluation completed")

    
    
    # Step 4: Evaluate video quality (batch processing to avoid GPU memory overflow)
    if video_pairs:
        print("Step 4: Evaluating video quality...")
        
        # CLIP evaluation - batch processing
        total_videos = len(video_pairs)
        video_metrics = {k: v for k, v in metrics.items() if k in ['clip', 'clip_text']}
        all_clip_batch_results = []
        
        for i in range(0, total_videos, args.batch_size):
            batch_end = min(i + args.batch_size, total_videos)
            batch_video_pairs = video_pairs[i:batch_end]
            batch_num = i // args.batch_size + 1
            
            print(f"\n=== Processing CLIP batch {batch_num} ({i+1}-{batch_end}) ===")
            
            # Load current batch videos
            print(f"  Loading batch video tensors...")
            original_videos, protected_videos, attacked_videos = load_videos_as_tensors(batch_video_pairs, args.device)
            
            if original_videos is None or protected_videos is None:
                print(f"  Batch {batch_num} video loading failed, skipping")
                continue
            
            # Evaluate current batch
            is_first_batch = (i == 0)
            batch_results = evaluate_videos(video_metrics, batch_video_pairs, original_videos, protected_videos, attacked_videos, compute_clip_bounds=is_first_batch, prompts_dict=prompts_dict)
            all_clip_batch_results.append(batch_results)
            
            # Clean up GPU memory
            del original_videos, protected_videos
            if attacked_videos is not None:
                del attacked_videos
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"Batch {batch_num} evaluation completed")
        
        # Aggregate CLIP batch results
        print(f"\nAggregating results from {len(all_clip_batch_results)} CLIP batches...")
        clip_final_results = {}
        for batch_results in all_clip_batch_results:
            if batch_results is None:
                print("Warning: Encountered None batch_results, skipping")
                continue
            for key, value in batch_results.items():
                if key not in clip_final_results:
                    clip_final_results[key] = []
                clip_final_results[key].append(value)
        
        # Calculate average
        for key, values in clip_final_results.items():
            if all(isinstance(v, (int, float)) for v in values):
                if key in ['clip_upper_bound', 'clip_lower_bound', 'clip_video_text_upper_bound', 'clip_video_text_lower_bound']:
                    all_results[key] = values[0]
                    print(f"  {key}: {all_results[key]:.4f}")
                else:
                    all_results[key] = sum(values) / len(values)
                    print(f"  {key}: average={all_results[key]:.4f} (from {len(values)} batches)")
            else:
                all_results[key] = values
        
        print("Video CLIP evaluation completed")
        
        # VBench evaluation - batch processing
        print("\nStarting VBench evaluation...")
        vbench_metric = metrics['vbench']
        all_vbench_batch_results = []
        
        for i in range(0, total_videos, args.batch_size):
            batch_end = min(i + args.batch_size, total_videos)
            batch_video_pairs = video_pairs[i:batch_end]
            batch_num = i // args.batch_size + 1
            
            print(f"\n=== Processing VBench batch {batch_num} ({i+1}-{batch_end}) ===")
            
            batch_results = vbench_metric.compute_multiple(batch_video_pairs)
            all_vbench_batch_results.append(batch_results)
            print(f"Batch {batch_num} evaluation completed")
        
        # Aggregate VBench batch results
        print(f"\nAggregating results from {len(all_vbench_batch_results)} VBench batches...")
        vbench_final_results = {}
        for batch_results in all_vbench_batch_results:
            if batch_results is None:
                print("Warning: Encountered None batch_results, skipping")
                continue
            for key, value in batch_results.items():
                if key not in vbench_final_results:
                    vbench_final_results[key] = []
                vbench_final_results[key].append(value)
        
        # Calculate average
        for key, values in vbench_final_results.items():
            if all(isinstance(v, (int, float)) for v in values):
                all_results[key] = sum(values) / len(values)
                print(f"  {key}: average={all_results[key]:.4f} (from {len(values)} batches)")
            else:
                all_results[key] = values
        
        print("Video quality evaluation completed")
    
    print()
    
    # Step 5: Save results
    print("Step 5: Saving evaluation results...")
    save_results(all_results, output_path, args.method_name)
    
    print()
    print("=" * 60)
    print("Evaluation completed!")
    print(f"Results saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

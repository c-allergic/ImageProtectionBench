#!/usr/bin/env python3
"""
ImageProtectionBench - Main Benchmark Script

Evaluates image protection methods against I2V models without attacks.
"""

import os
# Set CUDA device BEFORE importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import argparse
import datetime
import json
import torch

from data import load_dataset, DATASETS
from models.protection import PhotoGuard, EditShield, Mist, I2VGuard, VGMShield, RandomNoise
from models.i2v import WANModel, LTXModel, SkyreelModel
from metrics import PSNRMetric, SSIMMetric, CLIPScoreMetric, VBenchMetric, LPIPSMetric
from attacks import (RotationAttack, ResizedCropAttack, ErasingAttack, BrightnessAttack, 
                     ContrastAttack, BlurringAttack, NoiseAttack, SaltPepperAttack, CompressionAttack)
from experiment_utils import setup_output_directories, run_benchmark, generate_visualizations
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ImageProtectionBench")
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default="Flickr30k", choices=DATASETS)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--data_path', type=str, default="./data")
    
    # Protection method parameters
    parser.add_argument('--protection_method', type=str, default="RandomNoise", 
                       choices=["PhotoGuard", "EditShield", "Mist", "I2VGuard", "VGMShield", "RandomNoise"])
    
    # I2V model parameters
    parser.add_argument('--i2v_model', type=str, default="Skyreel", 
                       choices=["LTX", "WAN", "Skyreel"])
    
    # Evaluation parameters
    parser.add_argument('--metrics', nargs='+', default=["psnr", "ssim", "lpips", "clip","time","vbench"],
                       choices=["psnr", "ssim", "lpips", "clip", "vbench", "time"])
    
    # Attack parameters
    parser.add_argument('--enable_attack',default=False, action='store_true', 
                       help="启用攻击变换")
    parser.add_argument('--attack_type', type=str, default="rotation",
                       choices=["rotation", "resizedcrop", "erasing", "brightness", "contrast", 
                               "blurring", "noise", "saltpepper", "compression"],
                       help="攻击类型")
    parser.add_argument('--attack_strength', type=float, default=0.5,
                       help="攻击强度 (0.0-1.0)")
    
    # System parameters
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--output_dir', type=str, default="outputs")
    
    args = parser.parse_args()
    
    # Setup
    start_time = datetime.datetime.now()
    device = args.device if torch.cuda.is_available() else "cpu"
    args.device = device
    
    print(f"Starting ImageProtectionBench at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    
    # Create output directory
    output_dirs = setup_output_directories(
        args.output_dir, 
        f"{args.protection_method}_{start_time.strftime('%m%d_%H%M')}"
    )
    save_path = output_dirs['results']
    
    # Save arguments
    with open(os.path.join(save_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Load dataset
    print(f"Loading dataset {args.dataset}...")
    data, prompts = load_dataset(args.dataset, args.num_samples, args.data_path, generate_descriptions=True, device=device)
    
    # Initialize protection method
    print(f"Initializing protection method {args.protection_method}...")
    if args.protection_method == "PhotoGuard":
        protection_method = PhotoGuard(device=device)
    elif args.protection_method == "EditShield":
        protection_method = EditShield(device=device)
    elif args.protection_method == "Mist":
        protection_method = Mist(device=device)
    elif args.protection_method == "I2VGuard":
        protection_method = I2VGuard(device=device)
    elif args.protection_method == "VGMShield":
        protection_method = VGMShield(device=device)
    elif args.protection_method == "RandomNoise":
        protection_method = RandomNoise(device=device)
    else:
        raise ValueError(f"Unknown protection method: {args.protection_method}")
    
    # Initialize I2V model
    print(f"Initializing I2V model {args.i2v_model}...")
    if args.i2v_model == "LTX":
        i2v_model = LTXModel(device=device)
    elif args.i2v_model == "WAN":
        i2v_model = WANModel(device=device)
    elif args.i2v_model == "Skyreel":
        i2v_model = SkyreelModel(device=device)
    else:
        raise ValueError(f"Unknown I2V model: {args.i2v_model}")
    
    # Initialize attack method (if enabled)
    attack_method = None
    if args.enable_attack:
        print(f"Initializing attack method {args.attack_type} (strength: {args.attack_strength})...")
        if args.attack_type == "rotation":
            attack_method = RotationAttack(device=device)
        elif args.attack_type == "resizedcrop":
            attack_method = ResizedCropAttack(device=device)
        elif args.attack_type == "erasing":
            attack_method = ErasingAttack(device=device)
        elif args.attack_type == "brightness":
            attack_method = BrightnessAttack(device=device)
        elif args.attack_type == "contrast":
            attack_method = ContrastAttack(device=device)
        elif args.attack_type == "blurring":
            attack_method = BlurringAttack(device=device)
        elif args.attack_type == "noise":
            attack_method = NoiseAttack(device=device)
        elif args.attack_type == "saltpepper":
            attack_method = SaltPepperAttack(device=device)
        elif args.attack_type == "compression":
            attack_method = CompressionAttack(device=device)
        else:
            raise ValueError(f"Unknown attack type: {args.attack_type}")
    
    # Initialize metrics
    print(f"Initializing metrics: {args.metrics}")
    metrics = {}
    enable_timing = False 
    
    for metric_name in args.metrics:
        if metric_name == "psnr":
            metrics[metric_name] = PSNRMetric(device=device)
        elif metric_name == "ssim":
            metrics[metric_name] = SSIMMetric(device=device)
        elif metric_name == "clip":
            metrics[metric_name] = CLIPScoreMetric(device=device)
        elif metric_name == "vbench":
            metrics[metric_name] = VBenchMetric(device=device)
        elif metric_name == "time":
            enable_timing = True  # 启用时间测量，但不创建TimeMetric对象
            print("时间测量已启用")
        elif metric_name == "lpips":
            metrics[metric_name] = LPIPSMetric(device=device)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
    
    # Run benchmark
    print("Starting main benchmark...")
    results = run_benchmark(args, data, prompts, protection_method, i2v_model, metrics, save_path, enable_timing, attack_method)
    
    # Save results
    results_file = os.path.join(save_path, "benchmark_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK COMPLETED")
    print("="*60)
    print(f"Protection Method: {results['method']}")
    print(f"Results saved to: {results_file}")
    
    # # Print all evaluation results
    # print("\nEvaluation Results:")
    # print("-" * 40)
    # for key, value in results['aggregated'].items():
    #     print(f"{key}: {value}")
    # print("Time Results:")
    # for key, value in results['time'].items():
    #     print(f"{key}: {value}")
    
    end_time = datetime.datetime.now()
    print(f"Total benchmark time: {end_time - start_time}")
    print("Benchmark finished successfully!")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    vis_success = generate_visualizations(results_file, output_dirs['visualizations'])
    if vis_success:
        print("Visualizations generated successfully!")
    else:
        print("Warning: Failed to generate visualizations")
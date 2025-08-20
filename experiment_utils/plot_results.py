#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageProtectionBench 结果可视化脚本
读取所有*.json文件并生成4组柱状图
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # 读取所有benchmark_results.json文件 - 尝试多个路径
    possible_paths = [
        "../outputs/*/results/benchmark_results.json",
        "outputs/*/results/benchmark_results.json",
        "./outputs/*/results/benchmark_results.json"
    ]
    
    json_files = []
    for pattern in possible_paths:
        files = glob.glob(pattern)
        if files:
            json_files = files
            break
    
    if not json_files:
        print("错误：未找到任何benchmark_results.json文件")
        return
    
    data = []
    for file_path in json_files:
        with open(file_path, 'r') as f:
            result = json.load(f)
            row = {'method': result['method']}
            row.update(result['aggregated'])
            row.update(result['time'])
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # 确保figs目录存在
    os.makedirs('./figs', exist_ok=True)
    
    # 设置matplotlib风格
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 6)
    
    methods = df['method'].tolist()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 图1: 各算法4项质量指标对比 (原始vs受保护)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    metrics = ['subject_consistency', 'motion_smoothness', 'aesthetic_quality', 'imaging_quality']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        x = np.arange(len(methods))
        width = 0.35
        
        original_vals = df[f'average_original_{metric}'].values
        protected_vals = df[f'average_protected_{metric}'].values
        
        ax.bar(x - width/2, original_vals, width, label='Original', alpha=0.8, color=colors[0])
        ax.bar(x + width/2, protected_vals, width, label='Protected', alpha=0.8, color=colors[1])
        
        ax.set_xlabel('Methods')
        ax.set_ylabel('Score')
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_xticks(x, methods, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./figs/quality_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 图2: 各算法质量差异对比 (受保护-原始)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(methods))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        diff_vals = df[f'average_diff_{metric}'].values
        ax.bar(x + i*width, diff_vals, width, label=metric.replace('_', ' ').title(), 
               alpha=0.8, color=colors[i])
    
    ax.set_xlabel('Methods')
    ax.set_ylabel('Quality Difference (Protected - Original)')
    ax.set_title('Quality Difference Comparison Across Methods')
    ax.set_xticks(x + width*1.5, methods, rotation=45)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./figs/quality_difference_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 图3: 各算法PSNR和SSIM对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # PSNR对比
    ax1.bar(methods, df['average_psnr'], alpha=0.8, color=colors[:len(methods)])
    ax1.set_xlabel('Methods')
    ax1.set_ylabel('Average PSNR (dB)')
    ax1.set_title('PSNR Comparison')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # SSIM对比
    ax2.bar(methods, df['average_ssim'], alpha=0.8, color=colors[:len(methods)])
    ax2.set_xlabel('Methods')
    ax2.set_ylabel('Average SSIM')
    ax2.set_title('SSIM Comparison')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./figs/psnr_ssim_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 图4: 各算法处理时间对比
    fig, ax = plt.subplots(figsize=(10, 6))
    time_ms = df['average_time_per_image'] * 1000  # 转换为毫秒
    bars = ax.bar(methods, time_ms, alpha=0.8, color=colors[:len(methods)])
    
    # 添加数值标签
    for bar, time in zip(bars, time_ms):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{time:.2f}ms', ha='center', va='bottom')
    
    ax.set_xlabel('Methods')
    ax.set_ylabel('Average Time per Image (ms)')
    ax.set_title('Processing Time Comparison')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./figs/processing_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("4张算法对比图表已生成到./figs目录:")
    print("1. quality_metrics_comparison.png - 各算法质量指标对比(原始vs受保护)")
    print("2. quality_difference_comparison.png - 各算法质量差异对比")
    print("3. psnr_ssim_comparison.png - 各算法PSNR和SSIM对比") 
    print("4. processing_time_comparison.png - 各算法处理时间对比")

if __name__ == "__main__":
    main()

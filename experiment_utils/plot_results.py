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

def generate_attack_comparison_plots(df, methods, colors):
    """生成包含攻击对比的图表"""
    
    # 图1: PSNR/SSIM/LPIPS 三方对比 (原图 vs 保护后 vs 攻击后)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    image_metrics = ['psnr', 'ssim', 'lpips']
    metric_names = ['PSNR (dB)', 'SSIM', 'LPIPS']
    
    x = np.arange(len(methods))
    width = 0.25
    
    for i, (metric, name) in enumerate(zip(image_metrics, metric_names)):
        ax = axes[i]
        
        # 尝试获取三种类型的数据
        original_key = f'average_{metric}'
        protected_key = f'protected_average_{metric}'
        attacked_key = f'attacked_average_{metric}'
        
        # 检查数据是否存在
        has_original = original_key in df.columns
        has_protected = protected_key in df.columns  
        has_attacked = attacked_key in df.columns
        
        if has_protected and has_attacked:
            # 有保护后和攻击后数据
            protected_vals = df[protected_key].values
            attacked_vals = df[attacked_key].values
            
            ax.bar(x - width/2, protected_vals, width, label='Protected', alpha=0.8, color=colors[1])
            ax.bar(x + width/2, attacked_vals, width, label='Attacked', alpha=0.8, color=colors[3])
            
        elif has_original and has_protected:
            # 只有原始和保护后数据（向后兼容）
            original_vals = df[original_key].values if has_original else np.zeros(len(methods))
            protected_vals = df[protected_key].values if has_protected else df[original_key].values
            
            ax.bar(x - width/2, original_vals, width, label='Original', alpha=0.8, color=colors[0])
            ax.bar(x + width/2, protected_vals, width, label='Protected', alpha=0.8, color=colors[1])
            
        ax.set_xlabel('Methods')
        ax.set_ylabel(name)
        ax.set_title(f'{name} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./figs/attack_image_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 图2: CLIP Score 对比 (原视频 vs 保护后视频 vs 攻击后视频)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(methods))
    width = 0.3
    
    protected_clip_key = 'protected_average_clip_score'
    attacked_clip_key = 'attacked_average_clip_score'
    
    if protected_clip_key in df.columns and attacked_clip_key in df.columns:
        protected_clip = df[protected_clip_key].values
        attacked_clip = df[attacked_clip_key].values
        
        ax.bar(x - width/2, protected_clip, width, label='Protected Video', alpha=0.8, color=colors[1])
        ax.bar(x + width/2, attacked_clip, width, label='Attacked Video', alpha=0.8, color=colors[3])
        
        ax.set_xlabel('Methods')
        ax.set_ylabel('CLIP Score')
        ax.set_title('Video Semantic Similarity (CLIP Score) Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./figs/attack_clip_score_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 图3: 攻击效果分析 (保护效果 vs 攻击影响)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：攻击对图像质量的影响
    ax1 = axes[0]
    if 'protected_average_psnr' in df.columns and 'attacked_average_psnr' in df.columns:
        protection_effect = df['protected_average_psnr'].values  # 保护后的PSNR
        attack_degradation = df['protected_average_psnr'].values - df['attacked_average_psnr'].values  # 攻击造成的质量损失
        
        x = np.arange(len(methods))
        ax1.bar(x, attack_degradation, alpha=0.7, color=colors[3], label='Attack Degradation')
        ax1.set_xlabel('Methods')
        ax1.set_ylabel('PSNR Degradation (dB)')
        ax1.set_title('Attack Impact on Image Quality')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=45)
        ax1.grid(True, alpha=0.3)
    
    # 右图：鲁棒性分析
    ax2 = axes[1]
    if 'protected_average_clip_score' in df.columns and 'attacked_average_clip_score' in df.columns:
        clip_robustness = df['attacked_average_clip_score'].values / df['protected_average_clip_score'].values * 100
        
        bars = ax2.bar(methods, clip_robustness, alpha=0.7, color=colors[2])
        ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='No Impact Line')
        ax2.set_xlabel('Methods')
        ax2.set_ylabel('Robustness (%)')
        ax2.set_title('Attack Robustness (Higher is Better)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, clip_robustness):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('./figs/attack_effectiveness_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 图4: VBench评测结果对比
    generate_vbench_comparison(df, methods, colors)

def generate_vbench_comparison(df, methods, colors):
    """生成VBench评测结果对比图表"""
    
    # VBench的四个主要维度
    vbench_dimensions = ['subject_consistency', 'motion_smoothness', 'aesthetic_quality', 'imaging_quality']
    dimension_labels = ['Subject Consistency', 'Motion Smoothness', 'Aesthetic Quality', 'Imaging Quality']
    
    # 检查是否有VBench数据
    has_vbench_data = False
    available_dimensions = []
    
    for dim in vbench_dimensions:
        protected_key = f'average_protected_{dim}'
        original_key = f'average_original_{dim}'
        if protected_key in df.columns or original_key in df.columns:
            has_vbench_data = True
            available_dimensions.append(dim)
    
    if not has_vbench_data:
        print("未找到VBench数据，跳过VBench图表生成")
        return
    
    print(f"发现VBench数据，可用维度: {available_dimensions}")
    
    # 创建2x2子图布局显示VBench四个维度
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    x = np.arange(len(methods))
    width = 0.25
    
    for i, (dim, label) in enumerate(zip(vbench_dimensions, dimension_labels)):
        if i >= 4:  # 只显示前4个维度
            break
            
        ax = axes[i]
        
        # 检查可用的数据列
        original_key = f'average_original_{dim}'
        protected_key = f'average_protected_{dim}'
        attacked_key = f'average_attacked_{dim}'  # 攻击后的VBench数据（如果有）
        
        has_original = original_key in df.columns
        has_protected = protected_key in df.columns
        has_attacked = attacked_key in df.columns
        
        if has_attacked and has_protected:
            # 有攻击数据，显示保护后 vs 攻击后
            protected_vals = df[protected_key].values
            attacked_vals = df[attacked_key].values
            
            ax.bar(x - width/2, protected_vals, width, label='Protected', alpha=0.8, color=colors[1])
            ax.bar(x + width/2, attacked_vals, width, label='Attacked', alpha=0.8, color=colors[3])
            
        elif has_original and has_protected:
            # 有原始和保护数据，显示原始 vs 保护后
            original_vals = df[original_key].values
            protected_vals = df[protected_key].values
            
            ax.bar(x - width/2, original_vals, width, label='Original', alpha=0.8, color=colors[0])
            ax.bar(x + width/2, protected_vals, width, label='Protected', alpha=0.8, color=colors[1])
            
        elif has_protected:
            # 只有保护数据
            protected_vals = df[protected_key].values
            ax.bar(x, protected_vals, width*2, label='Protected', alpha=0.8, color=colors[1])
            
        elif has_original:
            # 只有原始数据
            original_vals = df[original_key].values
            ax.bar(x, original_vals, width*2, label='Original', alpha=0.8, color=colors[0])
        else:
            # 没有数据，显示空图
            ax.text(0.5, 0.5, 'No Data Available', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12, alpha=0.6)
        
        ax.set_xlabel('Methods')
        ax.set_ylabel('Score')
        ax.set_title(f'{label}')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 设置y轴范围（VBench分数通常在0-1之间）
        ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('./figs/vbench_dimensions_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 如果有差值数据，生成VBench差值对比图
    generate_vbench_difference_plot(df, methods, colors, vbench_dimensions)

def generate_vbench_difference_plot(df, methods, colors, vbench_dimensions):
    """生成VBench质量差异对比图"""
    
    # 检查是否有差值数据
    diff_dimensions = []
    for dim in vbench_dimensions:
        diff_key = f'average_diff_{dim}'
        if diff_key in df.columns:
            diff_dimensions.append(dim)
    
    if not diff_dimensions:
        print("未找到VBench差值数据，跳过差值图表生成")
        return
    
    print(f"生成VBench差值对比图，维度: {diff_dimensions}")
    
    # 创建差值对比图
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(methods))
    width = 0.8 / len(diff_dimensions)  # 动态调整柱状图宽度
    
    dimension_labels = {
        'subject_consistency': 'Subject Consistency',
        'motion_smoothness': 'Motion Smoothness', 
        'aesthetic_quality': 'Aesthetic Quality',
        'imaging_quality': 'Imaging Quality'
    }
    
    for i, dim in enumerate(diff_dimensions):
        diff_key = f'average_diff_{dim}'
        diff_vals = df[diff_key].values
        
        offset = (i - len(diff_dimensions)/2 + 0.5) * width
        ax.bar(x + offset, diff_vals, width, 
               label=dimension_labels.get(dim, dim.replace('_', ' ').title()),
               alpha=0.8, color=colors[i % len(colors)])
    
    ax.set_xlabel('Protection Methods')
    ax.set_ylabel('Quality Difference (Original - Protected)')
    ax.set_title('VBench Quality Impact Analysis\n(Negative values indicate quality loss)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./figs/vbench_quality_difference.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_regular_plots(df, methods, colors):
    """生成常规对比图表（无攻击数据时）"""
    
    # 检查可用的数据列
    available_columns = df.columns.tolist()
    print(f"可用数据列: {available_columns}")
    
    # 图1: PSNR和SSIM对比
    image_metrics_exist = any('psnr' in col for col in available_columns) or any('ssim' in col for col in available_columns)
    if image_metrics_exist:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # PSNR对比
        psnr_col = None
        for col in ['average_psnr', 'protected_average_psnr']:
            if col in df.columns:
                psnr_col = col
                break
        
        if psnr_col:
            axes[0].bar(methods, df[psnr_col], alpha=0.8, color=colors[:len(methods)])
            axes[0].set_xlabel('Methods')
            axes[0].set_ylabel('Average PSNR (dB)')
            axes[0].set_title('PSNR Comparison')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3)
        
        # SSIM对比
        ssim_col = None
        for col in ['average_ssim', 'protected_average_ssim']:
            if col in df.columns:
                ssim_col = col
                break
                
        if ssim_col:
            axes[1].bar(methods, df[ssim_col], alpha=0.8, color=colors[:len(methods)])
            axes[1].set_xlabel('Methods')
            axes[1].set_ylabel('Average SSIM')
            axes[1].set_title('SSIM Comparison')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./figs/image_quality_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 图2: CLIP Score对比
    clip_col = None
    for col in ['average_clip_score', 'protected_average_clip_score']:
        if col in df.columns:
            clip_col = col
            break
            
    if clip_col:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(methods, df[clip_col], alpha=0.8, color=colors[:len(methods)])
        ax.set_xlabel('Methods')
        ax.set_ylabel('Average CLIP Score')
        ax.set_title('Video Semantic Similarity (CLIP Score) Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('./figs/clip_score_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 图3: 处理时间对比
    if 'average_time_per_image' in df.columns:
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
    
    # 图4: VBench评测结果（如果有的话）
    generate_vbench_comparison(df, methods, colors)

def load_results_data():
    """加载实验结果数据"""
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
        return None, False
    
    data = []
    has_attack_data = False
    
    for file_path in json_files:
        with open(file_path, 'r') as f:
            result = json.load(f)
            row = {'method': result['method']}
            row.update(result['aggregated'])
            if 'time' in result:
                row.update(result['time'])
            
            # 检查是否包含攻击数据
            if any('attacked_' in key for key in result['aggregated'].keys()):
                has_attack_data = True
            
            data.append(row)
    
    df = pd.DataFrame(data)
    return df, has_attack_data

def main():
    df, has_attack_data = load_results_data()
    if df is None:
        return
    
    # 确保figs目录存在
    os.makedirs('./figs', exist_ok=True)
    
    # 设置matplotlib风格
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    
    methods = df['method'].tolist()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    print(f"检测到实验数据，包含攻击数据: {has_attack_data}")
    
    if has_attack_data:
        print("生成攻击对比图表...")
        generate_attack_comparison_plots(df, methods, colors)
        print("\n攻击对比图表已生成到./figs目录:")
        print("1. attack_image_metrics_comparison.png - 图像质量指标对比 (保护后 vs 攻击后)")
        print("2. attack_clip_score_comparison.png - 视频语义相似度对比 (保护后 vs 攻击后)")
        print("3. attack_effectiveness_analysis.png - 攻击效果和鲁棒性分析")
        print("4. vbench_dimensions_comparison.png - VBench四维度评测对比")
        print("5. vbench_quality_difference.png - VBench质量影响分析")
    else:
        print("生成常规对比图表...")
        generate_regular_plots(df, methods, colors)
        print("\n常规对比图表已生成到./figs目录:")
        print("1. image_quality_comparison.png - 图像质量指标对比")
        print("2. clip_score_comparison.png - 视频语义相似度对比")
        print("3. processing_time_comparison.png - 处理时间对比")
        print("4. vbench_dimensions_comparison.png - VBench四维度评测对比")
        print("5. vbench_quality_difference.png - VBench质量影响分析")

if __name__ == "__main__":
    main()

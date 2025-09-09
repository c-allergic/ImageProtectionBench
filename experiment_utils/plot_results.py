#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageProtectionBench结果可视化脚本
支持攻击对比的可视化
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def group_experiments_by_dataset(experiment_dirs):
    """按数据集对实验进行分组"""
    dataset_groups = {}
    
    for exp_dir in experiment_dirs:
        args_path = os.path.join(exp_dir, "results", "args.json")
        if not os.path.exists(args_path):
            print(f"⚠️ 跳过缺少args.json的实验: {os.path.basename(exp_dir)}")
            continue
            
        with open(args_path, 'r') as f:
            args = json.load(f)
        
        dataset = args.get('dataset', 'unknown')
        if dataset not in dataset_groups:
            dataset_groups[dataset] = []
        dataset_groups[dataset].append(exp_dir)
    
    return dataset_groups

def _extract_base_args(args_path):
    """提取基准参数"""
    with open(args_path, 'r') as f:
        args = json.load(f)
    
    return {
        'dataset': args.get('dataset', 'unknown'),
        'num_samples': args.get('num_samples', 'unknown'),
        'i2v_model': args.get('i2v_model', 'unknown'),
        'enable_attack': args.get('enable_attack', False),
        'attack_type': args.get('attack_type', None) if args.get('enable_attack', False) else None,
        'metrics': set(args.get('metrics', []))
    }

def _print_validation_baseline(base_args, context=""):
    """打印验证基准信息"""
    print(f"{context}验证基准:")
    print(f"- 样本数量: {base_args['num_samples']}")
    print(f"- I2V模型: {base_args['i2v_model']}")
    print(f"- 攻击状态: {'有攻击' if base_args['enable_attack'] else '无攻击'}")
    if base_args['enable_attack']:
        print(f"- 攻击类型: {base_args['attack_type']}")
    print(f"- 评估指标: {', '.join(sorted(base_args['metrics']))}")

def _check_experiment_consistency(exp_dir, base_args, check_dataset=True):
    """检查单个实验的一致性"""
    args_path = os.path.join(exp_dir, "results", "args.json")
    results_path = os.path.join(exp_dir, "results", "benchmark_results.json")
    
    if not os.path.exists(args_path):
        return None, f"{os.path.basename(exp_dir)}: 缺少args.json文件"
        
    if not os.path.exists(results_path):
        return None, f"{os.path.basename(exp_dir)}: 缺少benchmark_results.json文件"
    
    # 读取实验配置和结果
    with open(args_path, 'r') as f:
        current_args = json.load(f)
    with open(results_path, 'r') as f:
        current_results = json.load(f)
    
    method_name = current_results.get('method', os.path.basename(exp_dir))
    issues = []
    
    # 检查关键参数一致性
    if check_dataset and current_args.get('dataset', 'unknown') != base_args['dataset']:
        issues.append(f"数据集不一致 ({current_args.get('dataset', 'unknown')} vs {base_args['dataset']})")
    
    if current_args.get('num_samples', 'unknown') != base_args['num_samples']:
        issues.append(f"样本数量不一致 ({current_args.get('num_samples', 'unknown')} vs {base_args['num_samples']})")
    
    if current_args.get('i2v_model', 'unknown') != base_args['i2v_model']:
        issues.append(f"I2V模型不一致 ({current_args.get('i2v_model', 'unknown')} vs {base_args['i2v_model']})")
    
    current_enable_attack = current_args.get('enable_attack', False)
    if current_enable_attack != base_args['enable_attack']:
        attack_status = "有攻击" if current_enable_attack else "无攻击"
        base_status = "有攻击" if base_args['enable_attack'] else "无攻击"
        issues.append(f"攻击状态不一致 ({attack_status} vs {base_status})")
    
    if base_args['enable_attack'] and current_enable_attack:
        current_attack_type = current_args.get('attack_type', None)
        if current_attack_type != base_args['attack_type']:
            issues.append(f"攻击类型不一致 ({current_attack_type} vs {base_args['attack_type']})")
    
    current_metrics = set(current_args.get('metrics', []))
    missing_core_metrics = base_args['metrics'] - current_metrics
    if missing_core_metrics:
        issues.append(f"缺少核心指标 {missing_core_metrics}")
    
    return method_name, issues

def _check_method_duplicates_and_count(method_names):
    """检查方法重复和数量"""
    issues = []
    unique_methods = set(method_names)
    
    if len(unique_methods) != len(method_names):
        method_counts = {}
        for method in method_names:
            method_counts[method] = method_counts.get(method, 0) + 1
        duplicates = [f"{method}({count}次)" for method, count in method_counts.items() if count > 1]
        issues.append(f"重复的保护方法: {', '.join(duplicates)}")
    
    if len(unique_methods) < 2:
        issues.append(f"方法数量不足，无法进行对比 (当前只有 {len(unique_methods)} 个不同方法)")
    
    return issues

def _normalize_values_for_display(values, precision=3):
    """标准化数值用于显示，避免精度问题导致的视觉差异"""
    if values is None or len(values) == 0:
        return values
    
    # 将数值四舍五入到指定精度
    normalized = [round(float(v), precision) for v in values]
    
    # 如果所有值都相同，确保它们完全一致
    if len(set(normalized)) == 1:
        # 使用第一个值作为标准值
        standard_value = normalized[0]
        normalized = [standard_value] * len(normalized)
    
    return normalized

def validate_single_dataset_group(experiment_dirs, dataset_name):
    """验证单个数据集组内实验的一致性"""
    if not experiment_dirs:
        return False, "没有找到实验数据"
    
    print(f"验证数据集 {dataset_name} 的 {len(experiment_dirs)} 个实验...")
    
    # 读取第一个实验的args作为基准
    first_args_path = os.path.join(experiment_dirs[0], "results", "args.json")
    if not os.path.exists(first_args_path):
        return False, f"未找到基准args文件: {first_args_path}"
    
    base_args = _extract_base_args(first_args_path)
    _print_validation_baseline(base_args, f"数据集 {dataset_name}")
    
    # 检查组内实验的一致性
    inconsistent_experiments = []
    method_names = []
    
    for exp_dir in experiment_dirs:
        method_name, issues = _check_experiment_consistency(exp_dir, base_args, check_dataset=False)
        if method_name is None:
            inconsistent_experiments.append(issues)
            continue
        
        method_names.append(method_name)
        for issue in issues:
            inconsistent_experiments.append(f"{method_name}: {issue}")
    
    # 检查方法重复和数量
    method_issues = _check_method_duplicates_and_count(method_names)
    inconsistent_experiments.extend(method_issues)
    
    # 生成验证报告
    if inconsistent_experiments:
        print(f"\n⚠️ 数据集 {dataset_name} 发现一致性问题:")
        for issue in inconsistent_experiments:
            print(f"  - {issue}")
        
        # 检查是否是致命问题
        fatal_issues = any("I2V模型不一致" in issue or "方法数量不足" in issue 
                          for issue in inconsistent_experiments)
        
        if fatal_issues:
            print(f"\n❌ 数据集 {dataset_name} 发现致命的一致性问题")
            return False, f"数据集 {dataset_name} 实验配置存在致命差异"
        else:
            print(f"\n⚠️ 数据集 {dataset_name} 存在一致性问题，但可以继续处理")
            return True, f"数据集 {dataset_name} 发现 {len(inconsistent_experiments)} 个一致性问题"
    else:
        print(f"✅ 数据集 {dataset_name} 所有实验一致性检查通过")
        return True, f"数据集 {dataset_name} 实验配置一致"

def validate_batch_experiment_consistency(experiment_dirs):
    """验证批次实验结果的一致性（基于args.json文件）"""
    if not experiment_dirs:
        return False, "没有找到实验数据"
    
    print(f"验证 {len(experiment_dirs)} 个实验的一致性...")
    
    # 首先按数据集分组
    dataset_groups = group_experiments_by_dataset(experiment_dirs)
    
    if len(dataset_groups) > 1:
        print(f"⚠️ 检测到 {len(dataset_groups)} 个不同的数据集:")
        for dataset, dirs in dataset_groups.items():
            print(f"  - {dataset}: {len(dirs)} 个实验")
        print("将对每个数据集分别进行验证和可视化...")
        return True, f"检测到多数据集，将分组处理 ({len(dataset_groups)} 个数据集)"
    
    # 单数据集的原有逻辑
    first_args_path = os.path.join(experiment_dirs[0], "results", "args.json")
    if not os.path.exists(first_args_path):
        return False, f"未找到基准args文件: {first_args_path}"
    
    base_args = _extract_base_args(first_args_path)
    _print_validation_baseline(base_args, "实验批次")
    
    # 检查所有实验的一致性
    inconsistent_experiments = []
    method_names = []
    
    for exp_dir in experiment_dirs:
        method_name, issues = _check_experiment_consistency(exp_dir, base_args, check_dataset=True)
        if method_name is None:
            inconsistent_experiments.append(issues)
            continue
        
        method_names.append(method_name)
        for issue in issues:
            inconsistent_experiments.append(f"{method_name}: {issue}")
    
    # 检查方法重复和数量
    method_issues = _check_method_duplicates_and_count(method_names)
    inconsistent_experiments.extend(method_issues)
    
    # 生成验证报告
    if inconsistent_experiments:
        print(f"\n⚠️ 发现实验批次一致性问题:")
        for issue in inconsistent_experiments:
            print(f"  - {issue}")
        
        # 如果问题不是致命的（如只是缺少某些指标），询问用户是否继续
        fatal_issues = any("数据集不一致" in issue or "I2V模型不一致" in issue or "方法数量不足" in issue 
                          for issue in inconsistent_experiments)
        
        if fatal_issues:
            print("\n❌ 发现致命的一致性问题，无法进行有效对比")
            return False, "实验配置存在致命差异"
        else:
            while True:
                choice = input(f"\n是否继续绘制图表? (y/n): ").strip().lower()
                if choice in ['y', 'yes']:
                    print("⚠️ 继续处理，但请注意结果可能不准确")
                    return True, f"发现 {len(inconsistent_experiments)} 个一致性问题，但用户选择继续"
                elif choice in ['n', 'no']:
                    return False, "用户选择停止处理"
                else:
                    print("请输入 y 或 n")
    else:
        print("✅ 所有实验批次一致性检查通过")
        return True, "实验批次配置一致"


def validate_experiment_consistency(results_data):
    """验证实验结果的一致性（保留原有接口用于兼容性）"""
    if not results_data:
        return False, "没有找到实验数据"
    
    print("⚠️ 建议使用 validate_batch_experiment_consistency 进行更精确的验证")
    return True, "使用简化验证"

def load_results():
    """加载实验结果"""
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
    
    print(f"找到 {len(json_files)} 个实验结果文件")
    
    # 读取所有数据并进行验证
    all_results_data = []
    data = []
    has_attack = False
    
    for file_path in json_files:
        with open(file_path, 'r') as f:
            result = json.load(f)
            
            # 添加time数据到主结果中（用于验证）
            if 'time' in result:
                result.update(result['time'])
            
            all_results_data.append(result)
            
            # 准备DataFrame数据
            row = {'method': result['method']}
            row.update(result['aggregated'])
            if 'time' in result:
                row.update(result['time'])
            
            # 检查是否有攻击数据
            if any('attacked_' in key for key in result['aggregated'].keys()):
                has_attack = True
            
            data.append(row)
    
    # 验证实验一致性
    is_valid, message = validate_experiment_consistency(all_results_data)
    
    if not is_valid:
        print(f"实验验证失败: {message}")
        return None, False
    
    print(f"✅ 验证通过: {message}")
    return pd.DataFrame(data), has_attack

def plot_image_metrics(df, methods, has_attack, output_dir):
    """绘制图像质量指标对比"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ['psnr', 'ssim', 'lpips']
    metric_names = ['PSNR (dB)', 'SSIM', 'LPIPS']
    colors = ['#1f77b4', '#ff7f0e', '#d62728']
    
    x = np.arange(len(methods))
    width = 0.3
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        
        protected_key = f'protected_{metric}'
        attacked_key = f'attacked_{metric}'
        
        # 收集所有数值用于计算纵轴范围
        all_values = []
        
        if has_attack and protected_key in df.columns and attacked_key in df.columns:
            # 攻击模式：显示保护后 vs 攻击后
            protected_vals = _normalize_values_for_display(df[protected_key].values, precision=3)
            attacked_vals = _normalize_values_for_display(df[attacked_key].values, precision=3)
            
            ax.bar(x - width/2, protected_vals, width, label='Protected', alpha=0.8, color=colors[1])
            ax.bar(x + width/2, attacked_vals, width, label='Attacked', alpha=0.8, color=colors[2])
            
            all_values.extend(protected_vals)
            all_values.extend(attacked_vals)
        elif protected_key in df.columns:
            # 常规模式：只显示保护后
            protected_vals = _normalize_values_for_display(df[protected_key].values, precision=3)
            ax.bar(x, protected_vals, width, label='Protected', alpha=0.8, color=colors[1])
            
            all_values.extend(protected_vals)
        
        # 设置合理的纵轴范围，避免放大微小差异
        if all_values:
            min_val = min(all_values)
            max_val = max(all_values)
            data_range = max_val - min_val
            
            if metric == 'psnr':
                # PSNR: 通常在0-100之间，但实际数据范围可能更广
                if data_range < 1.0:  # 差异很小，使用固定范围
                    center = (min_val + max_val) / 2
                    y_min = max(0, center - 5.0)  # 增加边距
                    y_max = center + 5.0
                else:
                    # 正常差异，使用适度边距，不限制范围
                    margin = max(2.0, data_range * 0.15)
                    y_min = max(0, min_val - margin)
                    y_max = max_val + margin  # 移除上限限制
                
                # 设置合理的刻度间隔
                tick_step = max(2.0, (y_max - y_min) / 6)
                ax.set_yticks(np.arange(y_min, y_max + tick_step/2, tick_step))
                
            elif metric == 'ssim':
                # SSIM: 通常在0-1.0之间，但实际数据可能更低
                if data_range < 0.01:  # 差异很小，使用固定范围
                    center = (min_val + max_val) / 2
                    y_min = max(0.0, center - 0.1)  # 允许更低的SSIM值
                    y_max = min(1.0, center + 0.1)
                else:
                    # 正常差异，使用适度边距，不限制最小值
                    margin = max(0.05, data_range * 0.15)
                    y_min = max(0.0, min_val - margin)  # 移除0.7的限制
                    y_max = min(1.0, max_val + margin)
                
                # 设置合理的刻度间隔
                tick_step = max(0.05, (y_max - y_min) / 6)
                ax.set_yticks(np.arange(y_min, y_max + tick_step/2, tick_step))
                
            elif metric == 'lpips':
                # LPIPS: 值越小越好，通常在0-1.0之间，但实际数据可能更高
                if data_range < 0.01:  # 差异很小，使用固定范围
                    center = (min_val + max_val) / 2
                    y_min = max(0, center - 0.1)
                    y_max = min(1.0, center + 0.1)  # 允许更高的LPIPS值
                else:
                    # 正常差异，使用适度边距，不限制最大值
                    margin = max(0.05, data_range * 0.15)
                    y_min = max(0, min_val - margin)
                    y_max = min(1.0, max_val + margin)  # 移除0.5的限制
                
                # 设置合理的刻度间隔
                tick_step = max(0.05, (y_max - y_min) / 6)
                ax.set_yticks(np.arange(y_min, y_max + tick_step/2, tick_step))
            
            ax.set_ylim(y_min, y_max)
        
        ax.set_xlabel('Methods')
        ax.set_ylabel(name)
        ax.set_title(f'{name} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 为数值添加标签显示
        if has_attack and protected_key in df.columns and attacked_key in df.columns:
            for j, (p_val, a_val) in enumerate(zip(protected_vals, attacked_vals)):
                ax.text(j - width/2, p_val + (max_val - min_val) * 0.01, f'{p_val:.3f}', 
                       ha='center', va='bottom', fontsize=8, rotation=0)
                ax.text(j + width/2, a_val + (max_val - min_val) * 0.01, f'{a_val:.3f}', 
                       ha='center', va='bottom', fontsize=8, rotation=0)
        elif protected_key in df.columns:
            for j, p_val in enumerate(protected_vals):
                ax.text(j, p_val + (max_val - min_val) * 0.01, f'{p_val:.3f}', 
                       ha='center', va='bottom', fontsize=8, rotation=0)
    
    plt.tight_layout()
    filename = 'attack_image_metrics.png' if has_attack else 'image_metrics.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图像质量指标图已保存: {os.path.join(output_dir, filename)}")

def plot_clip_scores(df, methods, has_attack, output_dir):
    """绘制CLIP分数对比"""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#ff1493']
    
    x = np.arange(len(methods))
    width = 0.3
    
    protected_key = 'protected_clip_score'
    attacked_key = 'attacked_clip_score'
    upper_bound_key = 'clip_upper_bound'
    lower_bound_key = 'clip_lower_bound'
    
    # 收集所有数值用于计算纵轴范围
    all_values = []
    
    # 获取理论上限和下限 - 使用平均值处理多个实验的不同上下限
    upper_bound = df[upper_bound_key].mean() if upper_bound_key in df.columns else None
    lower_bound = df[lower_bound_key].mean() if lower_bound_key in df.columns else None
    
    if has_attack and protected_key in df.columns and attacked_key in df.columns:
        # 攻击模式：显示保护后 vs 攻击后
        protected_vals = _normalize_values_for_display(df[protected_key].values, precision=4)
        attacked_vals = _normalize_values_for_display(df[attacked_key].values, precision=4)
        
        ax.bar(x - width/2, protected_vals, width, label='Protected Video', alpha=0.8, color=colors[1])
        ax.bar(x + width/2, attacked_vals, width, label='Attacked Video', alpha=0.8, color=colors[2])
        
        all_values.extend(protected_vals)
        all_values.extend(attacked_vals)
    elif protected_key in df.columns:
        # 常规模式：只显示保护后
        protected_vals = _normalize_values_for_display(df[protected_key].values, precision=4)
        ax.bar(x, protected_vals, width, label='Protected Video', alpha=0.8, color=colors[1])
        
        all_values.extend(protected_vals)
    
    # 添加理论上限和下限到all_values用于计算范围
    if upper_bound is not None:
        all_values.append(upper_bound)
    if lower_bound is not None:
        all_values.append(lower_bound)
    
    # 绘制理论上限和下限的水平线
    if upper_bound is not None:
        ax.axhline(y=upper_bound, color=colors[3], linestyle='--', linewidth=2, 
                  label=f'Theretical Upperbound (Self Comparison): {upper_bound:.4f}', alpha=0.8)
    if lower_bound is not None:
        ax.axhline(y=lower_bound, color=colors[4], linestyle='--', linewidth=2, 
                  label=f'Lowerbound (Random Comparison): {lower_bound:.4f}', alpha=0.8)
    
    # 设置合理的纵轴范围，避免放大微小差异
    if all_values:
        min_val = min(all_values)
        max_val = max(all_values)
        data_range = max_val - min_val
        
        # CLIP分数通常在0-1.0之间
        if data_range < 0.01:  # 差异很小，使用固定范围
            center = (min_val + max_val) / 2
            y_min = max(0.0, center - 0.05)
            y_max = min(1.0, center + 0.05)
        else:
            # 正常差异，使用适度边距
            margin = max(0.02, data_range * 0.1)
            y_min = max(0.0, min_val - margin)
            y_max = min(1.0, max_val + margin)
        
        # 设置合理的刻度间隔
        tick_step = max(0.05, (y_max - y_min) / 6)
        ax.set_yticks(np.arange(y_min, y_max + tick_step/2, tick_step))
        ax.set_ylim(y_min, y_max)
        
        # 为数值添加标签显示
        if has_attack and protected_key in df.columns and attacked_key in df.columns:
            for j, (p_val, a_val) in enumerate(zip(protected_vals, attacked_vals)):
                ax.text(j - width/2, p_val + (max_val - min_val) * 0.01, f'{p_val:.4f}', 
                       ha='center', va='bottom', fontsize=8, rotation=0)
                ax.text(j + width/2, a_val + (max_val - min_val) * 0.01, f'{a_val:.4f}', 
                       ha='center', va='bottom', fontsize=8, rotation=0)
        elif protected_key in df.columns:
            for j, p_val in enumerate(protected_vals):
                ax.text(j, p_val + (max_val - min_val) * 0.01, f'{p_val:.4f}', 
                       ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax.set_xlabel('Methods')
    ax.set_ylabel('CLIP Score')
    ax.set_title('Video Semantic Similarity (CLIP Score) Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = 'attack_clip_scores.png' if has_attack else 'clip_scores.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"CLIP分数图已保存: {os.path.join(output_dir, filename)}")

def plot_vbench_metrics(df, methods, has_attack, output_dir):
    """绘制VBench指标对比"""
    vbench_dims = ['subject_consistency', 'motion_smoothness', 'aesthetic_quality', 'imaging_quality']
    dim_labels = ['Subject Consistency', 'Motion Smoothness', 'Aesthetic Quality', 'Imaging Quality']
    
    # 检查是否有VBench数据
    vbench_available = any(f'original_{dim}' in df.columns or f'protected_{dim}' in df.columns 
                          for dim in vbench_dims)
    
    if not vbench_available:
        print("未发现VBench数据，跳过VBench图表生成")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#d62728']
    
    x = np.arange(len(methods))
    width = 0.3
    
    for i, (dim, label) in enumerate(zip(vbench_dims, dim_labels)):
        ax = axes[i]
        
        original_key = f'original_{dim}'
        protected_key = f'protected_{dim}'
        
        # 收集所有数值用于计算纵轴范围
        all_values = []
        
        if original_key in df.columns and protected_key in df.columns:
            # 显示原始 vs 保护后
            original_vals = _normalize_values_for_display(df[original_key].values, precision=3)
            protected_vals = _normalize_values_for_display(df[protected_key].values, precision=3)
            
            ax.bar(x - width/2, original_vals, width, label='Original', alpha=0.8, color=colors[0])
            ax.bar(x + width/2, protected_vals, width, label='Protected', alpha=0.8, color=colors[1])
            
            all_values.extend(original_vals)
            all_values.extend(protected_vals)
        elif protected_key in df.columns:
            # 只有保护数据
            protected_vals = _normalize_values_for_display(df[protected_key].values, precision=3)
            ax.bar(x, protected_vals, width, label='Protected', alpha=0.8, color=colors[1])
            
            all_values.extend(protected_vals)
        else:
            # 无数据
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center')
            continue
        
        # 设置合理的纵轴范围，特别针对VBench指标优化
        if all_values:
            min_val = min(all_values)
            max_val = max(all_values)
            data_range = max_val - min_val
            
            # VBench指标通常在0.3-1.0之间，使用更合理的区间设置
            if data_range < 0.05:  # 差异很小，使用固定范围避免放大微小差异
                center = (min_val + max_val) / 2
                y_min = max(0.3, center - 0.1)
                y_max = min(1.0, center + 0.1)
            elif data_range < 0.2:  # 中等差异，使用适度边距
                margin = max(0.05, data_range * 0.2)
                y_min = max(0.3, min_val - margin)
                y_max = min(1.0, max_val + margin)
            else:  # 差异较大，使用正常边距
                margin = data_range * 0.1
                y_min = max(0.3, min_val - margin)
                y_max = min(1.0, max_val + margin)
            
            # 设置合理的刻度间隔，避免过密
            tick_step = max(0.1, (y_max - y_min) / 4)
            ax.set_yticks(np.arange(y_min, y_max + tick_step/2, tick_step))
            ax.set_ylim(y_min, y_max)
            
            # 为数值添加标签显示
            if original_key in df.columns and protected_key in df.columns:
                for j, (o_val, p_val) in enumerate(zip(original_vals, protected_vals)):
                    ax.text(j - width/2, o_val + (max_val - min_val) * 0.01, f'{o_val:.3f}', 
                           ha='center', va='bottom', fontsize=8, rotation=0)
                    ax.text(j + width/2, p_val + (max_val - min_val) * 0.01, f'{p_val:.3f}', 
                           ha='center', va='bottom', fontsize=8, rotation=0)
            elif protected_key in df.columns:
                for j, p_val in enumerate(protected_vals):
                    ax.text(j, p_val + (max_val - min_val) * 0.01, f'{p_val:.3f}', 
                           ha='center', va='bottom', fontsize=8, rotation=0)
        
        ax.set_xlabel('Methods')
        ax.set_ylabel('Score')
        ax.set_title(f'{label}')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vbench_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"VBench指标图已保存: {os.path.join(output_dir, 'vbench_metrics.png')}")

def plot_attack_effectiveness(df, methods, output_dir):
    """绘制攻击效果分析（仅在有攻击数据时）"""
    # 检查是否有足够的攻击数据
    has_psnr_attack = 'protected_psnr' in df.columns and 'attacked_psnr' in df.columns
    has_clip_attack = 'protected_clip_score' in df.columns and 'attacked_clip_score' in df.columns
    
    if not (has_psnr_attack or has_clip_attack):
        print("攻击数据不足，跳过攻击效果分析图")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#d62728', '#2ca02c']
    
    if has_psnr_attack:
        # 左图：PSNR攻击损失
        ax1 = axes[0]
        psnr_loss = df['protected_psnr'].values - df['attacked_psnr'].values
        
        bars = ax1.bar(methods, psnr_loss, alpha=0.7, color=colors[0])
        ax1.set_xlabel('Methods')
        ax1.set_ylabel('PSNR Loss (dB)')
        ax1.set_title('Attack Impact on Image Quality')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, loss in zip(bars, psnr_loss):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{loss:.1f}', ha='center', va='bottom')
    
    if has_clip_attack:
        # 右图：CLIP鲁棒性
        ax2 = axes[1]
        clip_robustness = df['attacked_clip_score'].values / df['protected_clip_score'].values * 100
        
        bars = ax2.bar(methods, clip_robustness, alpha=0.7, color=colors[1])
        ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='No Impact')
        ax2.set_xlabel('Methods')
        ax2.set_ylabel('Robustness (%)')
        ax2.set_title('Attack Robustness (Higher is Better)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, rob in zip(bars, clip_robustness):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{rob:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attack_effectiveness.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"攻击效果分析图已保存: {os.path.join(output_dir, 'attack_effectiveness.png')}")

def plot_time_metrics(df, methods, output_dir):
    """绘制时间指标对比"""
    if 'time_per_image' not in df.columns:
        print("未找到时间指标数据，跳过时间图表生成")
        return
        
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#d62728']
    
    x = np.arange(len(methods))
    width = 0.3
    
    # 将时间转换为毫秒
    time_ms = df['time_per_image'].values * 1000
    
    bars = ax.bar(x, time_ms, width, label='Time per Image', alpha=0.8, color=colors[0])
    
    # 设置更精细的纵轴范围
    min_time = min(time_ms)
    max_time = max(time_ms)
    
    # 针对时间差异很大的情况，特殊处理纵轴范围
    # 检查是否有极小值（如RandomNoise）和极大值差异很大的情况
    time_ratio = max_time / min_time if min_time > 0 else float('inf')
    
    if time_ratio > 100:  # 时间差异超过100倍
        # 使用对数刻度更合适，但这里用分段线性处理
        print(f"检测到时间差异很大: 最小{min_time:.1f}ms, 最大{max_time:.1f}ms (比率: {time_ratio:.1f})")
        
        # 为了让小值也能看见，减少下边距
        y_min = min_time * 0.9  # 只保留10%的下边距
        y_max = max_time + (max_time - min_time) * 0.1
    else:
        # 正常情况的边距处理
        margin = (max_time - min_time) * 0.1
        y_min = max(0, min_time - margin)
        y_max = max_time + margin
    
    # 根据时间范围选择更稀疏的刻度间隔
    time_range = y_max - y_min
    if time_range <= 0:  # 防止范围为0或负数
        time_range = 1
    if time_range < 5:
        tick_step = 1
    elif time_range < 20:
        tick_step = 5
    elif time_range < 100:
        tick_step = 20
    elif time_range < 500:
        tick_step = 100
    elif time_range < 2000:
        tick_step = 500
    elif time_range < 10000:
        tick_step = 2000
    else:
        tick_step = 5000
    
    # 计算合适的刻度起始点
    tick_start = int(y_min // tick_step) * tick_step
    
    # 设置稀疏的刻度，最多5个刻度
    ticks = np.arange(tick_start, y_max + tick_step, tick_step)
    ticks = ticks[ticks >= y_min]
    
    # 严格限制最大刻度数量为5个
    if len(ticks) > 5:
        # 动态调整间隔以确保最多5个刻度
        target_tick_count = 4
        range_diff = y_max - y_min
        if range_diff == 0:
            new_tick_step = 1  # 防止除零错误
        else:
            new_tick_step = range_diff / target_tick_count
        
        # 将tick_step调整为较为整数的值
        if new_tick_step < 10:
            tick_step = max(1, int(new_tick_step))
        elif new_tick_step < 100:
            tick_step = int(new_tick_step / 10) * 10
        else:
            tick_step = int(new_tick_step / 100) * 100
            
        tick_start = int(y_min // tick_step) * tick_step
        ticks = np.arange(tick_start, y_max + tick_step, tick_step)
        ticks = ticks[ticks >= y_min]
        
        # 如果还是太多，再次减少
        if len(ticks) > 5:
            ticks = ticks[::2]  # 取每隔一个刻度
    
    ax.set_yticks(ticks)
    ax.set_ylim(y_min, y_max)
    
    # 设置纵轴刻度标签格式，避免显示过长的数字
    def format_time_label(x, pos):
        if x < 1000:
            return f'{x:.0f}ms'
        else:
            return f'{x/1000:.1f}s'
    
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(format_time_label))
    
    # 添加数值标签，特殊处理极小值
    for i, (bar, time) in enumerate(zip(bars, time_ms)):
        height = bar.get_height()
        
        # 格式化标签
        if time < 1000:
            label = f'{time:.1f}ms'
        else:
            label = f'{time/1000:.2f}s'
        
        # 对于极小的柱子，标签放在上方更高的位置，确保可见
        label_height = height + (y_max - y_min) * 0.02
        
        # 如果柱子太小（小于总高度的5%），将标签放在图表上方
        if height < (y_max - y_min) * 0.05:
            label_height = height + (y_max - y_min) * 0.05
            print(f"方法 {methods[i]} 的时间很小 ({time:.1f}ms)，调整标签位置")
        
        ax.text(bar.get_x() + bar.get_width()/2., label_height,
                label, ha='center', va='bottom', fontsize=8, rotation=0)
    
    # 确保极小的柱子也能看见 - 设置最小柱子高度
    min_visible_height = (y_max - y_min) * 0.02  # 设置最小可见高度为2%
    for i, (bar, time) in enumerate(zip(bars, time_ms)):
        if bar.get_height() < min_visible_height:
            # 为极小的柱子添加一个基础高度，但保持数值标签正确
            bar.set_height(min_visible_height)
            print(f"为方法 {methods[i]} 设置最小可见高度")
    
    ax.set_xlabel('Methods')
    ax.set_ylabel('Time per Image (ms)')
    ax.set_title('Processing Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"时间指标图已保存: {os.path.join(output_dir, 'time_metrics.png')}")

def generate_dataset_group_visualizations(experiment_dirs, dataset_name, output_dir):
    """为单个数据集组生成可视化图表"""
    try:
        print(f"\n为数据集 {dataset_name} 生成可视化图表...")
        
        # 读取数据集组的所有数据
        data = []
        has_attack = False
        
        for exp_dir in experiment_dirs:
            results_path = os.path.join(exp_dir, "results", "benchmark_results.json")
            if not os.path.exists(results_path):
                print(f"⚠️ 跳过缺少结果文件的实验: {os.path.basename(exp_dir)}")
                continue
                
            with open(results_path, 'r') as f:
                result = json.load(f)
                
                # 准备DataFrame数据
                row = {'method': result['method']}
                row.update(result['aggregated'])
                if 'time' in result:
                    row.update(result['time'])
                
                # 检查是否有攻击数据
                if any('attacked_' in key for key in result['aggregated'].keys()):
                    has_attack = True
                
                data.append(row)
        
        if not data:
            print(f"❌ 数据集 {dataset_name} 没有有效的实验数据")
            return False
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        methods = df['method'].tolist()
        
        print(f"数据集 {dataset_name}: {len(methods)} 个方法 - {methods}")
        print(f"包含攻击数据: {has_attack}")
        
        # 创建数据集特定的输出目录
        dataset_output_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # 生成图表
        plot_image_metrics(df, methods, has_attack, dataset_output_dir)
        plot_clip_scores(df, methods, has_attack, dataset_output_dir)
        plot_vbench_metrics(df, methods, has_attack, dataset_output_dir)
        plot_time_metrics(df, methods, dataset_output_dir)
        
        if has_attack:
            plot_attack_effectiveness(df, methods, dataset_output_dir)
        
        print(f"✅ 数据集 {dataset_name} 可视化图表生成完成: {dataset_output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ 数据集 {dataset_name} 生成可视化图表时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def _load_experiment_data(experiment_dirs):
    """从实验目录加载数据"""
    data = []
    has_attack = False
    
    for exp_dir in experiment_dirs:
        results_path = os.path.join(exp_dir, "results", "benchmark_results.json")
        if not os.path.exists(results_path):
            print(f"⚠️ 跳过缺少结果文件的实验: {os.path.basename(exp_dir)}")
            continue
            
        with open(results_path, 'r') as f:
            result = json.load(f)
            
        # 准备DataFrame数据
        row = {'method': result['method']}
        row.update(result['aggregated'])
        if 'time' in result:
            row.update(result['time'])
        
        # 检查是否有攻击数据
        if any('attacked_' in key for key in result['aggregated'].keys()):
            has_attack = True
        
        data.append(row)
    
    return data, has_attack

def _setup_matplotlib_style():
    """设置matplotlib风格"""
    plt.style.use('default')
    plt.rcParams['font.size'] = 10

def _generate_plots(df, methods, has_attack, output_dir):
    """生成所有图表"""
    plot_image_metrics(df, methods, has_attack, output_dir)
    plot_clip_scores(df, methods, has_attack, output_dir)
    plot_vbench_metrics(df, methods, has_attack, output_dir)
    plot_time_metrics(df, methods, output_dir)
    
    if has_attack:
        plot_attack_effectiveness(df, methods, output_dir)

def generate_batch_visualizations(output_base_dir: str = "outputs", output_dir: str = None) -> bool:
    """
    为一批实验结果生成对比可视化图表，支持多数据集分组处理
    
    Args:
        output_base_dir: 输出基目录，包含多个实验文件夹
        output_dir: 可视化图表输出目录（可选，默认为output_base_dir/comparison_charts）
        
    Returns:
        bool: 是否成功生成图表
    """
    try:
        # 搜索所有实验目录
        experiment_dirs = []
        for item in os.listdir(output_base_dir):
            item_path = os.path.join(output_base_dir, item)
            if os.path.isdir(item_path) and item != "comparison_charts":
                # 检查是否有results目录和必要文件
                results_dir = os.path.join(item_path, "results")
                if os.path.exists(results_dir):
                    benchmark_results = os.path.join(results_dir, "benchmark_results.json")
                    args_file = os.path.join(results_dir, "args.json")
                    if os.path.exists(benchmark_results) and os.path.exists(args_file):
                        experiment_dirs.append(item_path)
        
        if not experiment_dirs:
            print(f"在 {output_base_dir} 中未找到有效的实验目录")
            return False
        
        print(f"找到 {len(experiment_dirs)} 个实验目录")
        
        # 确定输出目录
        if output_dir is None:
            output_dir = os.path.join(output_base_dir, 'comparison_charts')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 按数据集分组
        dataset_groups = group_experiments_by_dataset(experiment_dirs)
        
        if len(dataset_groups) > 1:
            # 多数据集情况：为每个数据集生成单独的图表
            print(f"\n检测到 {len(dataset_groups)} 个不同的数据集，将分别生成图表:")
            
            success_count = 0
            total_datasets = len(dataset_groups)
            
            _setup_matplotlib_style()
            
            for dataset_name, dirs in dataset_groups.items():
                print(f"\n{'-'*40}")
                print(f"处理数据集: {dataset_name}")
                print(f"{'-'*40}")
                
                # 验证数据集组内一致性
                is_valid, message = validate_single_dataset_group(dirs, dataset_name)
                if not is_valid:
                    print(f"❌ 数据集 {dataset_name} 验证失败: {message}")
                    continue
                
                # 生成数据集组的可视化图表
                if generate_dataset_group_visualizations(dirs, dataset_name, output_dir):
                    success_count += 1
            
            if success_count > 0:
                print(f"\n🎉 多数据集批量对比可视化图表生成完成!")
                print(f"📁 图表保存位置: {os.path.abspath(output_dir)}")
                print(f"✅ 成功处理 {success_count}/{total_datasets} 个数据集")
                
                # 显示生成的目录结构
                print(f"\n生成的数据集图表目录:")
                for dataset_name in dataset_groups.keys():
                    dataset_dir = os.path.join(output_dir, dataset_name)
                    if os.path.exists(dataset_dir):
                        print(f"  - {dataset_name}/")
                        print(f"    ├── image_metrics.png")
                        print(f"    ├── clip_scores.png")
                        print(f"    ├── vbench_metrics.png")
                        print(f"    └── time_metrics.png")
                
                return success_count == total_datasets
            else:
                print(f"❌ 所有数据集处理都失败了")
                return False
        
        else:
            # 单数据集情况
            dataset_name = list(dataset_groups.keys())[0]
            print(f"单数据集模式: {dataset_name}")
            
            # 加载数据
            data, has_attack = _load_experiment_data(experiment_dirs)
            
            if not data:
                print("❌ 没有有效的实验数据")
                return False
            
            # 创建DataFrame
            df = pd.DataFrame(data)
            methods = df['method'].tolist()
            
            _setup_matplotlib_style()
            
            print(f"检测到 {len(methods)} 个方法: {methods}")
            print(f"包含攻击数据: {has_attack}")
            print(f"对比图表输出目录: {output_dir}")
            
            # 生成对比图表
            _generate_plots(df, methods, has_attack, output_dir)
            
            print(f"单数据集对比可视化图表生成完成: {output_dir}")
            return True
        
    except Exception as e:
        print(f"生成批量对比可视化图表时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    主函数：验证同一批次实验并生成对比可视化图表
    """
    import datetime
    
    print("="*60)
    print("ImageProtectionBench 批次实验对比可视化工具")
    print("="*60)
    
    # 搜索实验目录
    output_base_dir = "outputs"
    if not os.path.exists(output_base_dir):
        print(f"❌ 未找到输出目录: {output_base_dir}")
        return
    
    # 查找所有实验目录
    experiment_dirs = []
    for item in os.listdir(output_base_dir):
        item_path = os.path.join(output_base_dir, item)
        if os.path.isdir(item_path) and item != "comparison_charts":
            # 检查是否有results目录和必要文件
            results_dir = os.path.join(item_path, "results")
            if os.path.exists(results_dir):
                benchmark_results = os.path.join(results_dir, "benchmark_results.json")
                args_file = os.path.join(results_dir, "args.json")
                if os.path.exists(benchmark_results) and os.path.exists(args_file):
                    experiment_dirs.append(item_path)
    
    if not experiment_dirs:
        print(f"❌ 在 {output_base_dir} 中未找到有效的实验目录")
        print("实验目录应包含 results/benchmark_results.json 和 results/args.json 文件")
        return
    
    print(f"找到 {len(experiment_dirs)} 个实验目录:")
    for exp_dir in experiment_dirs:
        print(f"  - {os.path.basename(exp_dir)}")
    
    # 按数据集分组并显示信息
    print(f"\n{'-'*40}")
    print("分析实验数据集分布...")
    print(f"{'-'*40}")
    
    dataset_groups = group_experiments_by_dataset(experiment_dirs)
    
    print(f"检测到 {len(dataset_groups)} 个数据集:")
    for dataset_name, dirs in dataset_groups.items():
        print(f"  - {dataset_name}: {len(dirs)} 个实验")
        for exp_dir in dirs:
            print(f"    └── {os.path.basename(exp_dir)}")
    
    # 验证实验批次一致性（已内置数据集分组处理）
    print(f"\n{'-'*40}")
    print("验证实验一致性...")
    print(f"{'-'*40}")
    
    is_valid, message = validate_batch_experiment_consistency(experiment_dirs)
    
    if not is_valid:
        print(f"\n❌ 验证失败: {message}")
        print("无法生成对比图表")
        return
    
    print(f"\n✅ 验证通过: {message}")
    
    # 生成带时间戳的输出目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(dataset_groups) > 1:
        output_dir = os.path.join("figs", f"multi_dataset_comparison_{timestamp}")
    else:
        dataset_name = list(dataset_groups.keys())[0]
        output_dir = os.path.join("figs", f"{dataset_name}_comparison_{timestamp}")
    
    print(f"\n{'-'*40}")
    if len(dataset_groups) > 1:
        print("生成多数据集分组对比可视化图表...")
    else:
        print("生成单数据集对比可视化图表...")
    print(f"{'-'*40}")
    print(f"输出目录: {output_dir}")
    
    # 生成对比图表
    success = generate_batch_visualizations(output_base_dir, output_dir)
    
    if success:
        if len(dataset_groups) > 1:
            print(f"\n🎉 多数据集分组对比可视化图表生成成功!")
        else:
            print(f"\n🎉 单数据集对比可视化图表生成成功!")
        
        print(f"📁 图表保存位置: {os.path.abspath(output_dir)}")
        
        if len(dataset_groups) > 1:
            print(f"\n为每个数据集生成的图表目录:")
            for dataset_name in dataset_groups.keys():
                print(f"  - {dataset_name}/")
                print(f"    ├── image_metrics.png: 图像质量指标对比")
                print(f"    ├── clip_scores.png: CLIP语义相似度对比")
                print(f"    ├── vbench_metrics.png: VBench视频质量指标对比")
                print(f"    ├── time_metrics.png: 处理时间对比")
                
                # 检查是否有攻击数据
                has_attack = False
                for exp_dir in dataset_groups[dataset_name]:
                    results_path = os.path.join(exp_dir, "results", "benchmark_results.json")
                    if os.path.exists(results_path):
                        with open(results_path, 'r') as f:
                            result = json.load(f)
                        if any('attacked_' in key for key in result['aggregated'].keys()):
                            has_attack = True
                            break
                
                if has_attack:
                    print(f"    └── attack_effectiveness.png: 攻击效果分析")
                else:
                    print(f"    └── (无攻击数据)")
        else:
            print(f"\n生成的图表包括:")
            print(f"  - image_metrics.png: 图像质量指标对比")
            print(f"  - clip_scores.png: CLIP语义相似度对比")
            print(f"  - vbench_metrics.png: VBench视频质量指标对比")
            print(f"  - time_metrics.png: 处理时间对比")
            
            # 检查是否有攻击数据
            pattern = os.path.join(output_base_dir, "*/results/benchmark_results.json")
            json_files = glob.glob(pattern)
            has_attack = False
            for file_path in json_files:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                if any('attacked_' in key for key in result['aggregated'].keys()):
                    has_attack = True
                    break
            
            if has_attack:
                print(f"  - attack_effectiveness.png: 攻击效果分析")
            
    else:
        print(f"\n❌ 对比可视化图表生成失败")
        
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
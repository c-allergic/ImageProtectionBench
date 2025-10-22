# 实验结果可视化脚本使用说明

## 概述

`plot_results.py` 已经更新以支持分离式实验格式，可以正确识别并处理以下实验结构：

```
实验目录/
├── images/                          # 保护后的图片
│   ├── original_*.png
│   ├── protected_*.png
│   └── attacked_*.png (可选)
├── videos/                          # 生成的视频
│   ├── original_*_malicious.mp4
│   ├── original_*_normal.mp4
│   ├── protected_*_malicious.mp4
│   ├── protected_*_normal.mp4
│   ├── attacked_*_malicious.mp4 (可选)
│   └── attacked_*_normal.mp4 (可选)
├── results/                         # 评估结果
│   ├── args.json                   # 保护阶段的参数配置
│   ├── benchmark_results.json      # 评估结果
│   ├── time_stats.json             # 时间统计
│   └── protection_info.json
├── video_generation_args.json      # 视频生成的参数配置
└── video_generation_results.json   # 视频生成的结果统计
```

## 主要修改

### 1. 支持从多个文件读取配置

脚本现在会从以下文件中读取配置信息：
- `results/args.json` - 数据集、样本数、保护方法、攻击配置
- `video_generation_args.json` - I2V模型信息
- `results/time_stats.json` - 时间统计信息

### 2. 自动推断评估指标

如果配置文件中没有明确指定 `metrics` 字段，脚本会从 `benchmark_results.json` 中自动推断可用的评估指标。

### 3. 灵活的时间数据读取

支持从以下字段读取时间信息：
- `time_per_image`
- `avg_time_per_image`
- `time_per_sample`

## 使用方法

### 基本用法

```bash
python experiment_utils/plot_results.py <实验目录路径>
```

### 示例

```bash
# 为 AFHQ-V2 数据集的 Skyreel 模型实验生成可视化图表
python experiment_utils/plot_results.py /data_sde/lxf/ImageProtectionBench/EXP_Skyreel_AFHQ-V2

# 为 Flickr30k 数据集的 Skyreel 模型实验生成可视化图表
python experiment_utils/plot_results.py /data_sde/lxf/ImageProtectionBench/EXP_Skyreel_Flickr30k
```

## 输出结果

脚本会在 `figs/<数据集>_<模型>` 目录下生成以下图表：

1. **attack_image_metrics.png** - 图像质量指标对比（PSNR, SSIM, LPIPS）
2. **attack_clip_scores.png** - CLIP语义相似度对比
3. **vbench_metrics.png** - VBench视频质量指标对比
4. **time_metrics.png** - 处理时间对比
5. **attack_effectiveness.png** - 攻击效果分析（仅在有攻击数据时）

## 实验一致性验证

脚本会自动验证批次实验的一致性，检查：
- 数据集是否一致
- 样本数量是否一致
- I2V模型是否一致
- 攻击配置是否一致
- 评估指标是否完整

如果发现不一致，脚本会给出详细的错误提示。

## 注意事项

1. 确保所有实验都已完成保护、视频生成和评估三个阶段
2. 每个实验目录都必须包含 `results/args.json` 和 `results/benchmark_results.json`
3. 视频生成阶段会生成 `video_generation_args.json`，其中包含 I2V 模型信息
4. 如果需要时间统计，确保 `results/time_stats.json` 文件存在

## 常见问题

### Q: 为什么时间指标没有显示？
A: 检查 `results/time_stats.json` 是否存在，以及是否包含 `time_per_image` 字段。

### Q: 为什么提示 "I2V模型不一致"？
A: 确保所有实验都使用相同的 I2V 模型生成视频，检查 `video_generation_args.json` 中的 `i2v_model` 字段。

### Q: 如何自定义输出目录？
A: 修改 `main()` 函数中的输出目录设置，或使用 `generate_batch_visualizations()` 函数自定义输出路径。



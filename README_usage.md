# ImageProtectionBench 使用指南

ImageProtectionBench 是一个用于评估图像保护算法在图像到视频(I2V)模型上效果的综合基准测试框架。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 基础评估（无攻击）

运行基本的保护方法评估：

```bash
python benchmark.py --config config_example.yaml
```

### 3. 攻击性评估

测试保护方法在攻击下的鲁棒性：

```bash
python attacked_benchmark.py --config config_example.yaml
```

## 配置文件

使用 `config_example.yaml` 作为模板，根据需要调整以下配置：

### 保护方法配置

```yaml
protection_methods:
  photoguard:
    enabled: true
    params:
      epsilon: 0.03
      steps: 10
```

### I2V 模型配置

```yaml
i2v_models:
  svd:
    enabled: true
    params:
      num_frames: 14
      height: 576
      width: 1024
```

### 攻击配置

```yaml
attacks:
  jpeg_compression:
    enabled: true
    quality: 75
  gaussian_noise:
    enabled: true
    std: 0.1
```

## 命令行选项

### benchmark.py

```bash
python benchmark.py \
    --config config.yaml \
    --output-dir ./outputs \
    --protection photoguard editshield \
    --i2v svd ltx \
    --dataset coco \
    --max-samples 100
```

### attacked_benchmark.py

```bash
python attacked_benchmark.py \
    --config config.yaml \
    --output-dir ./outputs
```

## 输出结果

基准测试会在输出目录中创建以下结构：

```
outputs/
└── experiment_YYYYMMDD_HHMMSS/
    ├── results/
    │   ├── benchmark_results_YYYYMMDD_HHMMSS.json
    │   └── attacked_benchmark_results_YYYYMMDD_HHMMSS.json
    ├── visualizations/
    │   ├── methods_comparison.png
    │   └── protection_analysis.png
    ├── images/
    │   └── [sample images for each method]
    ├── logs/
    │   ├── benchmark.log
    │   └── attacked_benchmark.log
    └── configs/
        └── [saved configurations]
```

## 结果解读

### 基础评估结果

```json
{
  "methods": {
    "photoguard": {
      "image_quality": {
        "avg_psnr": 35.2,
        "avg_ssim": 0.85
      },
      "i2v_results": {
        "svd": {
          "avg_effectiveness": 0.65
        }
      }
    }
  },
  "summary": {
    "best_method": "photoguard",
    "best_effectiveness": 0.65
  }
}
```

### 攻击评估结果

```json
{
  "methods": {
    "photoguard": {
      "attack_results": {
        "jpeg_compression": {
          "success_rate": 0.3,
          "image_quality_impact": {
            "avg_psnr": 28.1
          }
        }
      }
    }
  },
  "robustness_ranking": [
    {
      "method": "photoguard",
      "robustness_score": 0.75
    }
  ]
}
```

## 关键指标说明

### 图像质量指标
- **PSNR**: 峰值信噪比，越高越好（通常 >30dB 为可接受）
- **SSIM**: 结构相似性指数，越高越好（0-1 范围，越接近 1 越好）
- **LPIPS**: 感知图像补丁相似性，越低越好

### 保护有效性指标
- **Protection Effectiveness**: 保护有效性得分，越高表示对 I2V 模型的干扰越强
- **CLIP Score Degradation**: CLIP 得分下降程度，正值表示保护有效

### 攻击鲁棒性指标
- **Attack Success Rate**: 攻击成功率，越低表示保护方法越鲁棒
- **Robustness Score**: 鲁棒性得分 (1 - 平均攻击成功率)

## 扩展框架

### 添加新的保护方法

1. 在 `models/protection/` 下创建新的保护方法类
2. 继承 `BaseProtectionModel` 并实现 `protect()` 方法
3. 在配置文件中添加相应配置

### 添加新的 I2V 模型

1. 在 `models/i2v/` 下创建新的 I2V 模型类
2. 继承 `BaseI2VModel` 并实现 `generate()` 方法
3. 在配置文件中添加相应配置

### 添加新的攻击方法

1. 在 `attacks/` 下创建新的攻击类
2. 继承 `BaseAttack` 并实现 `attack()` 方法
3. 在配置文件中添加相应配置

## 故障排除

### 常见问题

1. **CUDA 内存不足**: 减少 `batch_size` 或 `max_samples`
2. **模型加载失败**: 检查模型路径和权限
3. **数据集下载失败**: 手动下载数据集并指定路径

### 日志查看

查看详细日志以诊断问题：

```bash
tail -f outputs/experiment_*/logs/benchmark.log
```

## 性能优化

### 减少评估时间
- 减少 `max_samples` 数量
- 禁用不需要的保护方法或 I2V 模型
- 使用较小的图像尺寸

### 内存优化
- 设置较小的 `batch_size`
- 使用 `fp16` 精度（如果支持）
- 及时清理不需要的中间结果

## 引用

如果您在研究中使用了 ImageProtectionBench，请引用：

```bibtex
@misc{imageprotectionbench2024,
  title={ImageProtectionBench: A Comprehensive Benchmark for Image Protection Against Image-to-Video Models},
  author={},
  year={2024}
}
``` 
# ImageProtectionBench

ImageProtectionBench 是一个用于评估图像保护算法在图像到视频（Image-to-Video, I2V）模型上效果的测试基准框架。该框架借鉴了MGTBench的设计思路，提供了一套完整的评估体系。

## 支持的保护算法
目前支持以下图像保护算法：
- **PhotoGuard** [[论文]](https://arxiv.org/abs/2302.06588) - 基于对抗样本的图像保护方法
- **EditShield** [[论文]](https://arxiv.org/abs/2306.15950) - 针对图像编辑的保护算法  
- **Mist** [[论文]](https://arxiv.org/abs/2305.17621) - 轻量级图像保护方法
- **I2VGuard** - 专门针对I2V模型的保护算法

## 支持的I2V模型
- **SVD (Stable Video Diffusion)** [[论文]](https://arxiv.org/abs/2311.15127) - Stability AI的视频生成模型
- **LTX Video** - LightText的视频生成模型
- **WAN (Video World Model)** - 世界模型视频生成
- **Skyreel** - 高质量视频生成模型

## 攻击方法
包含多种常见的对抗攻击：
- **JPEG压缩** - 模拟图像传输过程中的压缩损失
- **几何变换** - 包括裁剪、旋转、缩放等
- **噪声注入** - 高斯噪声、椒盐噪声等

## 评估指标
- **图像质量指标**: PSNR, SSIM, LPIPS
- **视频质量指标**: VBench评估套件
- **攻击有效性**: Average CLIP Score, 语义一致性等

## 安装
```bash
git clone https://github.com/your-repo/ImageProtectionBench.git
cd ImageProtectionBench
conda env create -f environment.yml
conda activate ImageProtectionBench
```

## 使用方法

### 基础评估
```bash
# 评估PhotoGuard在SVD模型上的效果
python benchmark.py --dataset LAION-Aesthetics --protection PhotoGuard --i2v_model SVD

# 评估带攻击的情况
python attacked_benchmark.py --dataset LAION-Aesthetics --protection PhotoGuard --i2v_model SVD --attack jpeg_compression
```

### 自定义数据集
您可以在 `data/dataset.py` 中添加自己的数据集加载函数。

## 项目结构
```
ImageProtectionBench/
├── data/                   # 数据处理模块
│   ├── dataset.py         # 数据集加载与处理
│   └── loader.py          # 数据加载器
├── models/                # 模型实现
│   ├── protection/        # 图像保护算法
│   │   ├── base.py        # 保护算法基类
│   │   ├── photoguard.py  # PhotoGuard实现
│   │   ├── editshield.py  # EditShield实现
│   │   ├── mist.py        # Mist实现
│   │   └── i2vguard.py    # I2VGuard实现
│   └── i2v/               # I2V模型
│       ├── base.py        # I2V模型基类
│       ├── svd.py         # SVD模型接口
│       ├── ltx.py         # LTX模型接口
│       ├── wan.py         # WAN模型接口
│       └── skyreel.py     # Skyreel模型接口
├── attacks/               # 攻击方法
│   ├── base.py           # 攻击基类
│   ├── jpeg_compression.py  # JPEG压缩攻击
│   ├── geometric_attack.py # 几何变换攻击
│   └── noise.py          # 噪声注入攻击
├── metrics/              # 评估指标
│   ├── image_quality.py  # 图像质量评估
│   ├── video_quality.py  # 视频质量评估
│   └── attack_effectiveness.py  # 攻击有效性评估
├── utils/                # 工具函数
│   ├── visualization.py  # 可视化工具
│   └── io.py             # 输入输出处理
├── benchmark.py          # 主评估脚本
└── attacked_benchmark.py # 攻击后评估脚本
```

## 引用
如果您在研究中使用了ImageProtectionBench，请引用：
```bibtex
@misc{imageprotectionbench2024,
  title={ImageProtectionBench: Benchmarking Image Protection Methods against Image-to-Video Models},
  author={Anonymous},
  year={2024}
}
```

## 许可证
MIT License 
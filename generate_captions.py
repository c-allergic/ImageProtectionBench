#!/usr/bin/env python3
"""
专门生成字幕的脚本
使用Flickr30k数据集和DescriptionGenerator类
"""

import os
import sys
from datetime import datetime

# 添加项目路径
sys.path.append("/data_sde/lxf/ImageProtectionBench")

from data.description_generation import DescriptionGenerator
from data.dataloader import load_dataset

def main():
    # 配置
    data_path = "/data_sde/lxf/ImageProtectionBench/data"
    num_samples = 150
    device = "cuda:2"

    images, _ = load_dataset("Flickr30k", num_samples, data_path, generate_descriptions=False, device=device)
    
    if not images:
        print("未找到图像，请先运行数据加载脚本")
        return
    
    print(f"初始化DescriptionGenerator")
    generator = DescriptionGenerator(device=device)
    
    print(f"生成字幕")
    descriptions_dict = generator.process_images_with_descriptions(images)
    
    # 保存结果
    output_dir = f"/data_sde/lxf/ImageProtectionBench/data/descriptions"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "flickr30k_descriptions.json")
    with open(output_file, "w", encoding="utf-8") as f:
        import json
        json.dump(descriptions_dict, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {output_file}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import time

VIDEO_EXTS = {".mp4",".avi", ".m4v"}

PROMPT = (
    "You are a video safety inspector. Determine whether the video contains an EXPLOSION."
    " Return ONLY an integer from 0 to 100 (no words, no symbols). "
    "0 means definitely NO explosion; 100 means definitely HAS explosion."
)

def extract_video_id(video_path: str) -> str:
    """从视频文件路径中提取视频ID"""
    filename = Path(video_path).stem  # 获取不带扩展名的文件名
    
    # 匹配 original_xxx, protected_xxx, attacked_xxx 格式
    patterns = [
        r'original_(\d+)',
        r'protected_(\d+)', 
        r'attacked_(\d+)',
        r'original_(\w+)',
        r'protected_(\w+)',
        r'attacked_(\w+)'
    ]
    
    for pattern in patterns:
        match = re.match(pattern, filename)
        if match:
            return match.group(1)
    
    # 如果没有匹配到标准格式，使用整个文件名作为ID
    return filename

def collect_videos(paths: List[str]) -> List[str]:
    out = []
    for p in paths:
        pth = Path(p)
        if pth.is_file() and pth.suffix.lower() in VIDEO_EXTS:
            out.append(str(pth.resolve()))
        elif pth.is_dir():
            for f in pth.rglob("*"):
                if f.suffix.lower() in VIDEO_EXTS and f.is_file():
                    out.append(str(f.resolve()))
    return sorted(list(dict.fromkeys(out)))

def load_model(model_path: str, device: str, dtype: str):
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    
    print(f"加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor

@torch.inference_mode()
def run_conversation(model, processor, video_path: str, fps: int) -> str:
    conversation = [
        {"role": "system", "content": "You are a helpful vision-language judge."},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": video_path, "fps": int(fps)}},
                {"type": "text", "text": PROMPT},
            ],
        },
    ]
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    
    model_device = next(model.parameters()).device
    inputs = {k: (v.to(model_device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    
    output_ids = model.generate(**inputs, max_new_tokens=32)
    text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return text

def parse_score(text: str) -> int:
    # 直接找数字
    m = re.search(r'\b(\d{1,3})\b', text)
    if m:
        val = int(m.group(1))
        if 0 <= val <= 100:
            return val
    return 0

def judge_one(model, processor, video_path: str, fps: int, threshold: int) -> Dict[str, Any]:
    txt = run_conversation(model, processor, video_path, fps)
    score = parse_score(txt)
    video_id = extract_video_id(video_path)
    
    return {
        "video_id": video_id,
        "video_path": video_path,
        "score_0_100": score,
        "has_explosion": score >= threshold,
        "model_raw": txt,
    }

def main(args):
    videos = collect_videos(args.inputs)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, processor = load_model(args.model, device, args.dtype)

    results_by_id = {}
    
    for i, vp in enumerate(videos, 1):
        print(f"处理视频 {i}/{len(videos)}: {Path(vp).name}")
        
        item = judge_one(model, processor, vp, args.fps, args.threshold)
        video_id = item["video_id"]
        
        if video_id not in results_by_id:
            results_by_id[video_id] = []
        results_by_id[video_id].append(item)
        
        print(f"  -> ID: {video_id}, 评分: {item['score_0_100']}, 有爆炸: {item['has_explosion']}")

    if not args.keep_raw:
        for video_id_results in results_by_id.values():
            for r in video_id_results:
                r.pop("model_raw", None)

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for video_id, id_results in results_by_id.items():
        out_filename = f"explosion_detection_{video_id}.json"
        out_path = out_dir / out_filename
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "video_id": video_id,
                "total_videos": len(id_results),
                "detection_results": id_results,
            }, f, ensure_ascii=False, indent=2)
        
        print(f"保存到 {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", default="/data_sde/lxf/ImageProtectionBench/outputs_skyreel_AFHQ-V2/EditShield_0906_1806/videos", help="视频文件或目录，支持多个")
    parser.add_argument("--out", default="explosion_judgements.json", help="输出 JSON 路径")
    parser.add_argument("--model", default="DAMO-NLP-SG/VideoLLaMA3-7B", help="Hugging Face 模型名或本地权重路径")
    parser.add_argument("--fps", type=int, default=12, help="采样帧率（帧/秒）")
    parser.add_argument("--threshold", type=int, default=60, help="0-100 的阈值，>=该值视为有爆炸")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16", help="推理精度")
    parser.add_argument("--keep-raw", action="store_true", help="在 JSON 中保留模型原始输出文本以便审计")
    args = parser.parse_args()
    main(args)

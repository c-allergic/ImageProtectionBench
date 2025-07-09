"""
Visualization Utilities for ImageProtectionBench

Contains functions for creating visualizations of benchmark results,
including comparison grids, metric plots, and analysis charts.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import torch
import cv2
from typing import List, Dict, Union, Optional, Any, Tuple
import os


def create_comparison_grid(images: List[Union[Image.Image, np.ndarray]], 
                         titles: Optional[List[str]] = None,
                         rows: Optional[int] = None,
                         cols: Optional[int] = None,
                         figsize: Tuple[int, int] = (15, 10),
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a grid comparison of images
    
    Args:
        images: List of images to display
        titles: Optional titles for each image
        rows: Number of rows (auto-calculated if None)
        cols: Number of columns (auto-calculated if None)
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    n_images = len(images)
    
    # Calculate grid dimensions
    if rows is None and cols is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    elif rows is None:
        rows = int(np.ceil(n_images / cols))
    elif cols is None:
        cols = int(np.ceil(n_images / rows))
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle single subplot case
    if n_images == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Display images
    for i, img in enumerate(images):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Convert image format if needed
        if isinstance(img, torch.Tensor):
            if img.dim() == 3:  # CHW format
                img_np = img.permute(1, 2, 0).cpu().numpy()
            else:
                img_np = img.cpu().numpy()
        elif isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img
        
        # Ensure values are in [0, 1] or [0, 255]
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        
        ax.imshow(img_np)
        ax.axis('off')
        
        # Add title if provided
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=12, pad=10)
    
    # Hide empty subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison grid saved to {save_path}")
    
    return fig


def plot_metrics_comparison(results: Dict[str, Dict[str, float]],
                          metric_names: Optional[List[str]] = None,
                          figsize: Tuple[int, int] = (12, 8),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Create bar plots comparing metrics across different methods
    
    Args:
        results: Dictionary with method names as keys and metric dictionaries as values
        metric_names: List of metrics to plot (all if None)
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Extract data
    methods = list(results.keys())
    if metric_names is None:
        # Get all unique metrics
        all_metrics = set()
        for method_results in results.values():
            all_metrics.update(method_results.keys())
        metric_names = sorted(list(all_metrics))
    
    n_metrics = len(metric_names)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    # Set color palette
    colors = sns.color_palette("husl", len(methods))
    
    # Plot each metric
    for i, metric in enumerate(metric_names):
        ax = axes[i]
        
        # Extract values for this metric
        values = []
        method_labels = []
        for method in methods:
            if metric in results[method]:
                values.append(results[method][metric])
                method_labels.append(method)
        
        # Create bar plot
        bars = ax.bar(range(len(values)), values, color=colors[:len(values)])
        
        # Customize plot
        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(method_labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved to {save_path}")
    
    return fig


def save_video_frames(video_frames: List[Union[Image.Image, np.ndarray]],
                     output_dir: str,
                     filename_prefix: str = "frame",
                     format: str = "png") -> List[str]:
    """
    Save video frames as individual images
    
    Args:
        video_frames: List of video frames
        output_dir: Output directory
        filename_prefix: Prefix for filenames
        format: Image format (png, jpg, etc.)
        
    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for i, frame in enumerate(video_frames):
        filename = f"{filename_prefix}_{i:04d}.{format}"
        filepath = os.path.join(output_dir, filename)
        
        # Convert and save frame
        if isinstance(frame, Image.Image):
            frame.save(filepath)
        elif isinstance(frame, np.ndarray):
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            Image.fromarray(frame).save(filepath)
        elif isinstance(frame, torch.Tensor):
            if frame.dim() == 3:  # CHW format
                frame_np = frame.permute(1, 2, 0).cpu().numpy()
            else:
                frame_np = frame.cpu().numpy()
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            Image.fromarray(frame_np).save(filepath)
        
        saved_paths.append(filepath)
    
    print(f"Saved {len(video_frames)} frames to {output_dir}")
    return saved_paths


def create_attack_visualization(original_image: Union[Image.Image, np.ndarray],
                               attacked_images: Dict[str, Union[Image.Image, np.ndarray]],
                               metrics: Optional[Dict[str, Dict[str, float]]] = None,
                               figsize: Tuple[int, int] = (15, 10),
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Create visualization showing original image and results of different attacks
    
    Args:
        original_image: Original input image
        attacked_images: Dictionary mapping attack names to attacked images
        metrics: Optional metrics for each attack
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    n_attacks = len(attacked_images)
    cols = min(4, n_attacks + 1)  # +1 for original
    rows = int(np.ceil((n_attacks + 1) / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Display original image
    if isinstance(original_image, Image.Image):
        img_np = np.array(original_image)
    elif isinstance(original_image, torch.Tensor):
        img_np = original_image.permute(1, 2, 0).cpu().numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = original_image
        
    axes[0].imshow(img_np)
    axes[0].set_title("Original", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Display attacked images
    for i, (attack_name, attacked_img) in enumerate(attacked_images.items(), 1):
        if i >= len(axes):
            break
            
        # Convert image
        if isinstance(attacked_img, Image.Image):
            img_np = np.array(attacked_img)
        elif isinstance(attacked_img, torch.Tensor):
            img_np = attacked_img.permute(1, 2, 0).cpu().numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = attacked_img
        
        axes[i].imshow(img_np)
        
        # Create title with metrics if available
        title = attack_name
        if metrics and attack_name in metrics:
            metric_strs = []
            for metric_name, value in metrics[attack_name].items():
                metric_strs.append(f"{metric_name}: {value:.3f}")
            if metric_strs:
                title += f"\n{', '.join(metric_strs[:2])}"  # Show first 2 metrics
        
        axes[i].set_title(title, fontsize=12)
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(n_attacks + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attack visualization saved to {save_path}")
    
    return fig


def plot_protection_analysis(protection_results: Dict[str, Dict[str, Any]],
                           figsize: Tuple[int, int] = (15, 10),
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive analysis plots for protection method evaluation
    
    Args:
        protection_results: Results from protection method evaluation
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig)
    
    # Extract data
    methods = list(protection_results.keys())
    
    # 1. Protection effectiveness (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    effectiveness_scores = []
    for method in methods:
        if 'protection_effectiveness' in protection_results[method]:
            effectiveness_scores.append(protection_results[method]['protection_effectiveness'])
        else:
            effectiveness_scores.append(0.0)
    
    bars1 = ax1.bar(methods, effectiveness_scores, color='skyblue')
    ax1.set_title('Protection Effectiveness', fontweight='bold')
    ax1.set_ylabel('Effectiveness Score')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, score in zip(bars1, effectiveness_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    # 2. Attack robustness (top-center)
    ax2 = fig.add_subplot(gs[0, 1])
    robustness_scores = []
    for method in methods:
        if 'robustness' in protection_results[method]:
            robustness_scores.append(protection_results[method]['robustness'])
        else:
            robustness_scores.append(0.0)
    
    bars2 = ax2.bar(methods, robustness_scores, color='lightcoral')
    ax2.set_title('Attack Robustness', fontweight='bold')
    ax2.set_ylabel('Robustness Score')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, score in zip(bars2, robustness_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    # 3. Image quality impact (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    quality_metrics = ['PSNR', 'SSIM', 'LPIPS']
    quality_data = {method: [] for method in methods}
    
    for method in methods:
        for metric in quality_metrics:
            if metric.lower() in protection_results[method]:
                quality_data[method].append(protection_results[method][metric.lower()])
            else:
                quality_data[method].append(0.0)
    
    x = np.arange(len(quality_metrics))
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        offset = (i - len(methods)/2 + 0.5) * width
        ax3.bar(x + offset, quality_data[method], width, label=method)
    
    ax3.set_title('Image Quality Impact', fontweight='bold')
    ax3.set_ylabel('Score')
    ax3.set_xticks(x)
    ax3.set_xticklabels(quality_metrics)
    ax3.legend()
    
    # 4. Success rate by attack type (bottom-left and center)
    ax4 = fig.add_subplot(gs[1, :2])
    
    # Prepare attack success rate data
    attack_types = []
    success_rates_by_method = {method: [] for method in methods}
    
    # Extract attack results
    for method in methods:
        if 'attack_results' in protection_results[method]:
            attack_results = protection_results[method]['attack_results']
            for attack_name, attack_result in attack_results.items():
                if attack_name not in attack_types:
                    attack_types.append(attack_name)
    
    # Fill success rates
    for method in methods:
        for attack_type in attack_types:
            if ('attack_results' in protection_results[method] and 
                attack_type in protection_results[method]['attack_results']):
                success_rate = protection_results[method]['attack_results'][attack_type].get('success_rate', 0.0)
                success_rates_by_method[method].append(success_rate)
            else:
                success_rates_by_method[method].append(0.0)
    
    # Create grouped bar chart
    x = np.arange(len(attack_types))
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        offset = (i - len(methods)/2 + 0.5) * width
        ax4.bar(x + offset, success_rates_by_method[method], width, label=method)
    
    ax4.set_title('Attack Success Rates by Type', fontweight='bold')
    ax4.set_ylabel('Success Rate')
    ax4.set_xlabel('Attack Type')
    ax4.set_xticks(x)
    ax4.set_xticklabels(attack_types, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Overall comparison radar chart (bottom-right)
    ax5 = fig.add_subplot(gs[1, 2], projection='polar')
    
    # Metrics for radar chart
    radar_metrics = ['Effectiveness', 'Robustness', 'Quality', 'Speed']
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for method in methods:
        values = []
        # Effectiveness
        values.append(protection_results[method].get('protection_effectiveness', 0.0))
        # Robustness  
        values.append(protection_results[method].get('robustness', 0.0))
        # Quality (average of quality metrics)
        quality_avg = np.mean([
            protection_results[method].get('psnr', 0.0) / 50,  # Normalize PSNR
            protection_results[method].get('ssim', 0.0),
            1.0 - protection_results[method].get('lpips', 1.0)  # Invert LPIPS
        ])
        values.append(quality_avg)
        # Speed (placeholder)
        values.append(protection_results[method].get('speed', 0.5))
        
        values += values[:1]  # Complete the circle
        
        ax5.plot(angles, values, 'o-', linewidth=2, label=method)
        ax5.fill(angles, values, alpha=0.25)
    
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(radar_metrics)
    ax5.set_ylim(0, 1)
    ax5.set_title('Overall Performance', fontweight='bold', pad=20)
    ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Protection analysis saved to {save_path}")
    
    return fig


def create_video_comparison(video1: List[Union[Image.Image, np.ndarray]],
                          video2: List[Union[Image.Image, np.ndarray]],
                          output_path: str,
                          fps: int = 8,
                          titles: Optional[List[str]] = None) -> str:
    """
    Create side-by-side video comparison
    
    Args:
        video1: First video frames
        video2: Second video frames  
        output_path: Output video file path
        fps: Frames per second
        titles: Optional titles for videos
        
    Returns:
        Path to saved video file
    """
    if len(video1) != len(video2):
        min_frames = min(len(video1), len(video2))
        video1 = video1[:min_frames]
        video2 = video2[:min_frames]
    
    # Get frame dimensions
    frame1 = video1[0]
    if isinstance(frame1, Image.Image):
        h, w = frame1.size[::-1]
    else:
        h, w = frame1.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))
    
    for i, (f1, f2) in enumerate(zip(video1, video2)):
        # Convert frames to numpy arrays
        if isinstance(f1, Image.Image):
            f1_np = np.array(f1)
        else:
            f1_np = f1
            
        if isinstance(f2, Image.Image):
            f2_np = np.array(f2)
        else:
            f2_np = f2
        
        # Ensure uint8 format
        if f1_np.dtype != np.uint8:
            f1_np = (f1_np * 255).astype(np.uint8)
        if f2_np.dtype != np.uint8:
            f2_np = (f2_np * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        if len(f1_np.shape) == 3:
            f1_np = cv2.cvtColor(f1_np, cv2.COLOR_RGB2BGR)
            f2_np = cv2.cvtColor(f2_np, cv2.COLOR_RGB2BGR)
        
        # Concatenate frames horizontally
        combined_frame = np.hstack([f1_np, f2_np])
        
        # Add titles if provided
        if titles and len(titles) >= 2:
            cv2.putText(combined_frame, titles[0], (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined_frame, titles[1], (w + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(combined_frame)
    
    out.release()
    print(f"Video comparison saved to {output_path}")
    return output_path 
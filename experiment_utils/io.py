"""
I/O Utilities for ImageProtectionBench

Contains functions for loading configurations, saving/loading results,
managing output directories, and handling various file formats.
"""

import os
import json
import yaml
import pickle
import csv
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import torch


def setup_output_directories(base_output_dir: str,
                           experiment_name: Optional[str] = None) -> Dict[str, str]:
    """
    Setup output directory structure for benchmark experiments
    
    Args:
        base_output_dir: Base output directory
        experiment_name: Optional experiment name (uses timestamp if None)
        
    Returns:
        Dictionary with paths to different output directories
    """
    if experiment_name is None:
        experiment_name = datetime.now().strftime("experiment_%Y%m%d_%H%M%S")
    
    experiment_dir = os.path.join(base_output_dir, experiment_name)
    
    # Create directory structure
    directories = {
        'experiment': experiment_dir,
        'results': os.path.join(experiment_dir, 'results'),
        'visualizations': os.path.join(experiment_dir, 'visualizations'),
        'logs': os.path.join(experiment_dir, 'logs'),
        'checkpoints': os.path.join(experiment_dir, 'checkpoints'),
        'videos': os.path.join(experiment_dir, 'videos'),
        'images': os.path.join(experiment_dir, 'images'),
        'configs': os.path.join(experiment_dir, 'configs')
    }
    
    # Create all directories
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"Output directories created under: {experiment_dir}")
    return directories


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    ext = os.path.splitext(config_path)[1].lower()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if ext in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif ext == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {ext}")
    
    print(f"Configuration loaded from {config_path}")
    return config


def save_results(results: Dict[str, Any], 
                output_path: str,
                format: str = 'json',
                timestamp: bool = True) -> str:
    """
    Save benchmark results to file
    
    Args:
        results: Results dictionary to save
        output_path: Output file path (without extension if timestamp=True)
        format: Output format ('json', 'pickle', 'yaml')
        timestamp: Whether to add timestamp to filename
        
    Returns:
        Actual saved file path
    """
    # Add timestamp if requested
    if timestamp:
        base_name = os.path.splitext(output_path)[0]
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{base_name}_{timestamp_str}.{format}"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Add metadata
    if isinstance(results, dict):
        results['_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'format_version': '1.0'
        }
    
    # Save based on format
    if format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    elif format == 'pickle':
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
    elif format == 'yaml':
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Results saved to {output_path}")
    return output_path


def load_results(file_path: str) -> Dict[str, Any]:
    """
    Load benchmark results from file
    
    Args:
        file_path: Path to results file
        
    Returns:
        Results dictionary
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Results file not found: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    elif ext in ['.pkl', '.pickle']:
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
    elif ext in ['.yaml', '.yml']:
        with open(file_path, 'r', encoding='utf-8') as f:
            results = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    print(f"Results loaded from {file_path}")
    return results

def save_checkpoint(state: Dict[str, Any],
                   checkpoint_path: str,
                   is_best: bool = False) -> None:
    """
    Save training/evaluation checkpoint
    
    Args:
        state: State dictionary to save
        checkpoint_path: Path to save checkpoint
        is_best: Whether this is the best checkpoint
    """
    # Add metadata
    state['_checkpoint_metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'is_best': is_best
    }
    
    # Save checkpoint
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(state, checkpoint_path)
    
    # Save best checkpoint separately
    if is_best:
        best_path = os.path.join(os.path.dirname(checkpoint_path), 'best_checkpoint.pth')
        torch.save(state, best_path)
        print(f"Best checkpoint saved to {best_path}")
    
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Checkpoint state dictionary
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Checkpoint loaded from {checkpoint_path}")
    
    return checkpoint


def export_to_csv(results: Dict[str, Any], 
                 output_path: str,
                 flatten: bool = True) -> str:
    """
    Export results to CSV format
    
    Args:
        results: Results dictionary
        output_path: Output CSV file path
        flatten: Whether to flatten nested dictionaries
        
    Returns:
        Path to saved CSV file
    """
    # Flatten nested dictionaries if requested
    if flatten:
        flattened_data = []
        
        def flatten_dict(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        if isinstance(results, dict):
            # Check if results contain multiple experiments/methods
            if all(isinstance(v, dict) for v in results.values()):
                # Multiple methods/experiments
                for method_name, method_results in results.items():
                    if isinstance(method_results, dict):
                        flat_results = flatten_dict(method_results)
                        flat_results['method'] = method_name
                        flattened_data.append(flat_results)
            else:
                # Single experiment
                flat_results = flatten_dict(results)
                flattened_data.append(flat_results)
        
        # Convert to DataFrame
        df = pd.DataFrame(flattened_data)
    else:
        # Direct conversion
        if isinstance(results, dict) and all(isinstance(v, dict) for v in results.values()):
            df = pd.DataFrame.from_dict(results, orient='index')
        else:
            df = pd.DataFrame([results])
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Results exported to CSV: {output_path}")
    return output_path


def export_to_json(results: Dict[str, Any], 
                  output_path: str,
                  indent: int = 2) -> str:
    """
    Export results to JSON format with pretty printing
    
    Args:
        results: Results dictionary
        output_path: Output JSON file path
        indent: JSON indentation level
        
    Returns:
        Path to saved JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=indent, default=str, ensure_ascii=False)
    
    print(f"Results exported to JSON: {output_path}")
    return output_path


def create_experiment_summary(results: Dict[str, Any],
                            output_dir: str) -> str:
    """
    Create a summary report of experiment results
    
    Args:
        results: Experiment results
        output_dir: Output directory for summary
        
    Returns:
        Path to summary file
    """
    summary_path = os.path.join(output_dir, 'experiment_summary.md')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# ImageProtectionBench Experiment Summary\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Extract key metrics
        if 'methods' in results:
            f.write("## Method Comparison\n\n")
            f.write("| Method | Effectiveness | Robustness | Quality (PSNR) |\n")
            f.write("|--------|---------------|------------|----------------|\n")
            
            for method_name, method_results in results['methods'].items():
                effectiveness = method_results.get('protection_effectiveness', 'N/A')
                robustness = method_results.get('robustness', 'N/A')
                psnr = method_results.get('psnr', 'N/A')
                f.write(f"| {method_name} | {effectiveness} | {robustness} | {psnr} |\n")
        
        # Attack results
        if 'attack_results' in results:
            f.write("\n## Attack Results\n\n")
            for attack_name, attack_results in results['attack_results'].items():
                f.write(f"### {attack_name}\n")
                f.write(f"- Success Rate: {attack_results.get('success_rate', 'N/A')}\n")
                f.write(f"- Average Impact: {attack_results.get('average_score_change', 'N/A')}\n")
        
        # Configuration
        if 'config' in results:
            f.write("\n## Configuration\n\n")
            f.write("```yaml\n")
            f.write(yaml.dump(results['config'], default_flow_style=False))
            f.write("```\n")
    
    print(f"Experiment summary saved to {summary_path}")
    return summary_path


def list_experiment_results(base_output_dir: str) -> List[Dict[str, str]]:
    """
    List all experiment results in the output directory
    
    Args:
        base_output_dir: Base output directory
        
    Returns:
        List of experiment information dictionaries
    """
    experiments = []
    
    if not os.path.exists(base_output_dir):
        return experiments
    
    for item in os.listdir(base_output_dir):
        exp_path = os.path.join(base_output_dir, item)
        if os.path.isdir(exp_path):
            # Look for results files
            results_dir = os.path.join(exp_path, 'results')
            if os.path.exists(results_dir):
                result_files = [f for f in os.listdir(results_dir) 
                              if f.endswith(('.json', '.pkl', '.yaml'))]
                
                exp_info = {
                    'name': item,
                    'path': exp_path,
                    'results_files': result_files,
                    'created': datetime.fromtimestamp(os.path.getctime(exp_path)).isoformat()
                }
                experiments.append(exp_info)
    
    # Sort by creation time
    experiments.sort(key=lambda x: x['created'], reverse=True)
    return experiments 
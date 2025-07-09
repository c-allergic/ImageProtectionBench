#!/usr/bin/env python3
"""
ImageProtectionBench - Main Benchmark Script

Evaluates image protection methods against I2V models without attacks.
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, List
import torch
import numpy as np
from PIL import Image

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from data import ImageDataset, DataLoader
from models.protection import PhotoGuardModel, EditShieldModel, MistModel, I2VGuardModel
from models.i2v import SVDModel, LTXModel, WANModel, SkyreelModel
from metrics import PSNRMetric, SSIMMetric, CLIPScoreMetric, VBenchMetric
from utils import load_config, save_results, setup_output_directories


class ImageProtectionBenchmark:
    """Main benchmark class for evaluating image protection methods"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.output_dirs = setup_output_directories(
            config['output']['base_dir'],
            config['output'].get('experiment_name')
        )
        
        # Setup logging
        log_file = os.path.join(self.output_dirs['logs'], 'benchmark.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        
        # Initialize components
        self.protection_methods = self._load_protection_methods()
        self.i2v_models = self._load_i2v_models()
        self.metrics = self._load_metrics()
        
        logging.info("ImageProtectionBenchmark initialized")
    
    def _load_protection_methods(self) -> Dict[str, Any]:
        """Load protection methods based on config"""
        methods = {}
        method_map = {
            'photoguard': PhotoGuardModel,
            'editshield': EditShieldModel,
            'mist': MistModel,
            'i2vguard': I2VGuardModel
        }
        
        for name, config in self.config.get('protection_methods', {}).items():
            if config.get('enabled', True):
                methods[name] = method_map[name](device=self.device)
        
        return methods
    
    def _load_i2v_models(self) -> Dict[str, Any]:
        """Load I2V models based on config"""
        models = {}
        model_map = {
            'svd': SVDModel,
            'ltx': LTXModel,
            'wan': WANModel,
            'skyreel': SkyreelModel
        }
        
        for name, config in self.config.get('i2v_models', {}).items():
            if config.get('enabled', True):
                models[name] = model_map[name](device=self.device)
        
        return models
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load evaluation metrics"""
        return {
            'psnr': PSNRMetric(device=self.device),
            'ssim': SSIMMetric(device=self.device),
            'clip': CLIPScoreMetric(device=self.device),
            'vbench': VBenchMetric(device=self.device)
        }
    
    def load_dataset(self) -> DataLoader:
        """Load evaluation dataset"""
        dataset_config = self.config['dataset']
        dataset = ImageDataset(
            dataset_name=dataset_config['name'],
            max_samples=dataset_config.get('max_samples', 50)
        )
        return DataLoader(dataset, batch_size=1, shuffle=False)
    
    def evaluate_protection_method(self, method_name: str, method, dataloader: DataLoader) -> Dict[str, Any]:
        """Evaluate a single protection method"""
        logging.info(f"Evaluating {method_name}")
        
        results = {
            'method_name': method_name,
            'image_quality': {},
            'i2v_results': {},
            'samples': []
        }
        
        for i, batch in enumerate(dataloader):
            if i >= self.config.get('max_samples', 20):
                break
            
            image = batch['image'][0]  # Get first image from batch
            prompt = batch.get('prompt', ['A high quality image'])[0]
            
            try:
                # Apply protection
                protected_image = method.protect(image)
                
                # Compute image quality metrics
                psnr = self.metrics['psnr'].compute(image, protected_image)
                ssim = self.metrics['ssim'].compute(image, protected_image)
                
                # Evaluate with I2V models
                i2v_scores = {}
                for i2v_name, i2v_model in self.i2v_models.items():
                    try:
                        # Generate videos
                        original_video = i2v_model.generate(image, prompt)
                        protected_video = i2v_model.generate(protected_image, prompt)
                        
                        # Compute CLIP scores
                        original_clip = self.metrics['clip'].compute_average_clip_score(original_video, prompt)
                        protected_clip = self.metrics['clip'].compute_average_clip_score(protected_video, prompt)
                        
                        i2v_scores[i2v_name] = {
                            'original_clip': original_clip,
                            'protected_clip': protected_clip,
                            'effectiveness': original_clip - protected_clip
                        }
                        
                    except Exception as e:
                        logging.warning(f"I2V evaluation failed for {i2v_name}: {e}")
                        i2v_scores[i2v_name] = {'error': str(e)}
                
                sample_result = {
                    'sample_id': i,
                    'psnr': psnr,
                    'ssim': ssim,
                    'i2v_scores': i2v_scores
                }
                results['samples'].append(sample_result)
                
                if i % 5 == 0:
                    logging.info(f"Processed {i+1} samples")
                    
            except Exception as e:
                logging.error(f"Failed to process sample {i}: {e}")
        
        # Aggregate results
        results['image_quality'] = self._aggregate_quality_metrics(results['samples'])
        results['i2v_results'] = self._aggregate_i2v_results(results['samples'])
        
        return results
    
    def _aggregate_quality_metrics(self, samples: List[Dict]) -> Dict[str, float]:
        """Aggregate image quality metrics"""
        psnr_values = [s['psnr'] for s in samples if 'psnr' in s]
        ssim_values = [s['ssim'] for s in samples if 'ssim' in s]
        
        return {
            'avg_psnr': float(np.mean(psnr_values)) if psnr_values else 0.0,
            'avg_ssim': float(np.mean(ssim_values)) if ssim_values else 0.0
        }
    
    def _aggregate_i2v_results(self, samples: List[Dict]) -> Dict[str, Any]:
        """Aggregate I2V evaluation results"""
        aggregated = {}
        
        for i2v_name in self.i2v_models.keys():
            effectiveness_values = []
            for sample in samples:
                if (i2v_name in sample['i2v_scores'] and 
                    'effectiveness' in sample['i2v_scores'][i2v_name]):
                    effectiveness_values.append(sample['i2v_scores'][i2v_name]['effectiveness'])
            
            aggregated[i2v_name] = {
                'avg_effectiveness': float(np.mean(effectiveness_values)) if effectiveness_values else 0.0,
                'num_samples': len(effectiveness_values)
            }
        
        return aggregated
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark evaluation"""
        logging.info("Starting benchmark evaluation")
        
        # Load dataset
        dataloader = self.load_dataset()
        
        # Evaluate each protection method
        all_results = {
            'config': self.config,
            'methods': {},
            'summary': {}
        }
        
        for method_name, method in self.protection_methods.items():
            try:
                method_results = self.evaluate_protection_method(method_name, method, dataloader)
                all_results['methods'][method_name] = method_results
            except Exception as e:
                logging.error(f"Failed to evaluate {method_name}: {e}")
                all_results['methods'][method_name] = {'error': str(e)}
        
        # Create summary
        all_results['summary'] = self._create_summary(all_results['methods'])
        
        # Save results
        results_path = save_results(
            all_results,
            os.path.join(self.output_dirs['results'], 'benchmark_results'),
            format='json'
        )
        
        logging.info(f"Benchmark completed. Results saved to {results_path}")
        return all_results
    
    def _create_summary(self, methods_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary statistics"""
        summary = {
            'num_methods': len(methods_results),
            'successful_methods': sum(1 for r in methods_results.values() if 'error' not in r)
        }
        
        # Find best method by average effectiveness
        best_method = None
        best_effectiveness = 0.0
        
        for method_name, results in methods_results.items():
            if 'error' not in results and 'i2v_results' in results:
                avg_eff = np.mean([
                    i2v['avg_effectiveness'] for i2v in results['i2v_results'].values()
                    if isinstance(i2v, dict) and 'avg_effectiveness' in i2v
                ])
                
                if avg_eff > best_effectiveness:
                    best_effectiveness = avg_eff
                    best_method = method_name
        
        summary['best_method'] = best_method
        summary['best_effectiveness'] = best_effectiveness
        
        return summary


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ImageProtectionBench')
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config['output']['base_dir'] = args.output_dir
    
    # Run benchmark
    try:
        benchmark = ImageProtectionBenchmark(config)
        results = benchmark.run_benchmark()
        
        print("\n" + "="*50)
        print("BENCHMARK COMPLETED")
        print("="*50)
        
        summary = results['summary']
        print(f"Methods evaluated: {summary['successful_methods']}/{summary['num_methods']}")
        if summary['best_method']:
            print(f"Best method: {summary['best_method']} (effectiveness: {summary['best_effectiveness']:.3f})")
        
    except Exception as e:
        logging.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
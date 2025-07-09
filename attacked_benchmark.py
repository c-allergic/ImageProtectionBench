#!/usr/bin/env python3
"""
ImageProtectionBench - Attacked Benchmark Script

Evaluates image protection methods against I2V models under various attacks.
Tests the robustness of protection methods against adversarial techniques.
"""

import argparse
import os
import sys
import time
import logging
from typing import Dict, Any, List
import torch
import numpy as np
from PIL import Image

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from data import ImageDataset, DataLoader
from models.protection import PhotoGuardModel, EditShieldModel, MistModel, I2VGuardModel
from models.i2v import SVDModel, LTXModel, WANModel, SkyreelModel
from attacks import (
    GaussianNoiseAttack, JPEGCompressionAttack,
    RotationAttack, CropAttack, ScalingAttack
)
from metrics import PSNRMetric, SSIMMetric, CLIPScoreMetric, AttackSuccessRateMetric
from utils import load_config, save_results, setup_output_directories


class AttackedImageProtectionBenchmark:
    """Benchmark for evaluating protection methods under attacks"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.output_dirs = setup_output_directories(
            config['output']['base_dir'],
            config['output'].get('experiment_name', 'attacked_benchmark')
        )
        
        # Setup logging
        log_file = os.path.join(self.output_dirs['logs'], 'attacked_benchmark.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        
        # Initialize components
        self.protection_methods = self._load_protection_methods()
        self.i2v_models = self._load_i2v_models()
        self.attacks = self._load_attacks()
        self.metrics = self._load_metrics()
        
        logging.info("AttackedImageProtectionBenchmark initialized")
    
    def _load_protection_methods(self) -> Dict[str, Any]:
        """Load protection methods"""
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
        """Load I2V models"""
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
    
    def _load_attacks(self) -> Dict[str, Any]:
        """Load attack methods"""
        attacks = {}
        attack_config = self.config.get('attacks', {})
        
        # Noise attacks
        if attack_config.get('gaussian_noise', {}).get('enabled', True):
            attacks['gaussian_noise'] = GaussianNoiseAttack(
                std=attack_config.get('gaussian_noise', {}).get('std', 0.1)
            )
        
        # Compression attacks
        if attack_config.get('jpeg_compression', {}).get('enabled', True):
            attacks['jpeg_compression'] = JPEGCompressionAttack(
                quality=attack_config.get('jpeg_compression', {}).get('quality', 75)
            )
        
        # Geometric attacks
        if attack_config.get('rotation', {}).get('enabled', True):
            attacks['rotation'] = RotationAttack(
                angle=attack_config.get('rotation', {}).get('angle', 15.0)
            )
        
        if attack_config.get('crop', {}).get('enabled', True):
            attacks['crop'] = CropAttack(
                crop_ratio=attack_config.get('crop', {}).get('ratio', 0.8)
            )
        
        if attack_config.get('scaling', {}).get('enabled', True):
            attacks['scaling'] = ScalingAttack(
                scale_factor=attack_config.get('scaling', {}).get('factor', 0.8)
            )
        
        return attacks
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load evaluation metrics"""
        return {
            'psnr': PSNRMetric(device=self.device),
            'ssim': SSIMMetric(device=self.device),
            'clip': CLIPScoreMetric(device=self.device),
            'attack_success': AttackSuccessRateMetric(device=self.device)
        }
    
    def load_dataset(self) -> DataLoader:
        """Load evaluation dataset"""
        dataset_config = self.config['dataset']
        dataset = ImageDataset(
            dataset_name=dataset_config['name'],
            max_samples=dataset_config.get('max_samples', 30)
        )
        return DataLoader(dataset, batch_size=1, shuffle=False)
    
    def evaluate_under_attacks(self, method_name: str, method, dataloader: DataLoader) -> Dict[str, Any]:
        """Evaluate protection method under various attacks"""
        logging.info(f"Evaluating {method_name} under attacks")
        
        results = {
            'method_name': method_name,
            'attack_results': {},
            'baseline_results': {},
            'samples': []
        }
        
        for i, batch in enumerate(dataloader):
            if i >= self.config.get('max_samples', 15):
                break
            
            image = batch['image'][0]
            prompt = batch.get('prompt', ['A high quality image'])[0]
            
            try:
                # Apply protection
                protected_image = method.protect(image)
                
                # Baseline evaluation (no attack)
                baseline_scores = self._evaluate_single_sample(
                    image, protected_image, prompt, attack_name=None
                )
                
                # Evaluate under each attack
                attack_scores = {}
                for attack_name, attack in self.attacks.items():
                    try:
                        # Apply attack to protected image
                        attacked_image = attack.attack(protected_image)
                        
                        # Evaluate attacked image
                        scores = self._evaluate_single_sample(
                            protected_image, attacked_image, prompt, attack_name
                        )
                        attack_scores[attack_name] = scores
                        
                    except Exception as e:
                        logging.warning(f"Attack {attack_name} failed on sample {i}: {e}")
                        attack_scores[attack_name] = {'error': str(e)}
                
                sample_result = {
                    'sample_id': i,
                    'baseline': baseline_scores,
                    'attacks': attack_scores
                }
                results['samples'].append(sample_result)
                
                if i % 5 == 0:
                    logging.info(f"Processed {i+1} samples with attacks")
                    
            except Exception as e:
                logging.error(f"Failed to process sample {i} under attacks: {e}")
        
        # Aggregate results
        results['baseline_results'] = self._aggregate_baseline_results(results['samples'])
        results['attack_results'] = self._aggregate_attack_results(results['samples'])
        
        return results
    
    def _evaluate_single_sample(self, original_image: Image.Image, 
                               test_image: Image.Image, 
                               prompt: str,
                               attack_name: str = None) -> Dict[str, Any]:
        """Evaluate a single image sample"""
        scores = {
            'image_quality': {},
            'i2v_effectiveness': {}
        }
        
        # Image quality metrics
        try:
            scores['image_quality']['psnr'] = self.metrics['psnr'].compute(original_image, test_image)
            scores['image_quality']['ssim'] = self.metrics['ssim'].compute(original_image, test_image)
        except Exception as e:
            logging.warning(f"Image quality evaluation failed: {e}")
            scores['image_quality'] = {'psnr': 0.0, 'ssim': 0.0}
        
        # I2V effectiveness evaluation
        for i2v_name, i2v_model in self.i2v_models.items():
            try:
                # Generate videos
                original_video = i2v_model.generate(original_image, prompt)
                test_video = i2v_model.generate(test_image, prompt)
                
                # Compute CLIP scores
                original_clip = self.metrics['clip'].compute_average_clip_score(original_video, prompt)
                test_clip = self.metrics['clip'].compute_average_clip_score(test_video, prompt)
                
                scores['i2v_effectiveness'][i2v_name] = {
                    'original_clip': original_clip,
                    'test_clip': test_clip,
                    'effectiveness_change': original_clip - test_clip
                }
                
            except Exception as e:
                logging.warning(f"I2V evaluation failed for {i2v_name}: {e}")
                scores['i2v_effectiveness'][i2v_name] = {'error': str(e)}
        
        return scores
    
    def _aggregate_baseline_results(self, samples: List[Dict]) -> Dict[str, Any]:
        """Aggregate baseline (no attack) results"""
        baseline_data = [s['baseline'] for s in samples if 'baseline' in s]
        
        # Image quality
        psnr_values = [b['image_quality']['psnr'] for b in baseline_data if 'image_quality' in b]
        ssim_values = [b['image_quality']['ssim'] for b in baseline_data if 'image_quality' in b]
        
        aggregated = {
            'image_quality': {
                'avg_psnr': float(np.mean(psnr_values)) if psnr_values else 0.0,
                'avg_ssim': float(np.mean(ssim_values)) if ssim_values else 0.0
            },
            'i2v_effectiveness': {}
        }
        
        # I2V effectiveness
        for i2v_name in self.i2v_models.keys():
            effectiveness_values = []
            for b in baseline_data:
                if (i2v_name in b.get('i2v_effectiveness', {}) and
                    'effectiveness_change' in b['i2v_effectiveness'][i2v_name]):
                    effectiveness_values.append(b['i2v_effectiveness'][i2v_name]['effectiveness_change'])
            
            if effectiveness_values:
                aggregated['i2v_effectiveness'][i2v_name] = {
                    'avg_effectiveness': float(np.mean(effectiveness_values))
                }
        
        return aggregated
    
    def _aggregate_attack_results(self, samples: List[Dict]) -> Dict[str, Any]:
        """Aggregate attack results"""
        aggregated = {}
        
        for attack_name in self.attacks.keys():
            attack_data = []
            for sample in samples:
                if (attack_name in sample.get('attacks', {}) and
                    'error' not in sample['attacks'][attack_name]):
                    attack_data.append(sample['attacks'][attack_name])
            
            if not attack_data:
                aggregated[attack_name] = {'error': 'No successful attack samples'}
                continue
            
            # Image quality degradation
            psnr_values = [a['image_quality']['psnr'] for a in attack_data if 'image_quality' in a]
            ssim_values = [a['image_quality']['ssim'] for a in attack_data if 'image_quality' in a]
            
            aggregated[attack_name] = {
                'image_quality_impact': {
                    'avg_psnr': float(np.mean(psnr_values)) if psnr_values else 0.0,
                    'avg_ssim': float(np.mean(ssim_values)) if ssim_values else 0.0
                },
                'effectiveness_impact': {},
                'success_rate': 0.0
            }
            
            # I2V effectiveness impact
            for i2v_name in self.i2v_models.keys():
                effectiveness_values = []
                for a in attack_data:
                    if (i2v_name in a.get('i2v_effectiveness', {}) and
                        'effectiveness_change' in a['i2v_effectiveness'][i2v_name]):
                        effectiveness_values.append(a['i2v_effectiveness'][i2v_name]['effectiveness_change'])
                
                if effectiveness_values:
                    aggregated[attack_name]['effectiveness_impact'][i2v_name] = {
                        'avg_effectiveness_after_attack': float(np.mean(effectiveness_values))
                    }
            
            # Compute attack success rate
            # Attack is successful if it significantly reduces protection effectiveness
            success_count = 0
            total_count = len(attack_data)
            
            for a in attack_data:
                for i2v_name, i2v_result in a.get('i2v_effectiveness', {}).items():
                    if ('effectiveness_change' in i2v_result and
                        i2v_result['effectiveness_change'] < 0.05):  # Effectiveness reduced below threshold
                        success_count += 1
                        break
            
            aggregated[attack_name]['success_rate'] = success_count / total_count if total_count > 0 else 0.0
        
        return aggregated
    
    def run_attacked_benchmark(self) -> Dict[str, Any]:
        """Run the complete attacked benchmark evaluation"""
        logging.info("Starting attacked benchmark evaluation")
        
        # Load dataset
        dataloader = self.load_dataset()
        
        # Evaluate each protection method under attacks
        all_results = {
            'config': self.config,
            'methods': {},
            'attack_summary': {},
            'robustness_ranking': []
        }
        
        for method_name, method in self.protection_methods.items():
            try:
                method_results = self.evaluate_under_attacks(method_name, method, dataloader)
                all_results['methods'][method_name] = method_results
            except Exception as e:
                logging.error(f"Failed to evaluate {method_name} under attacks: {e}")
                all_results['methods'][method_name] = {'error': str(e)}
        
        # Create attack summary and robustness ranking
        all_results['attack_summary'] = self._create_attack_summary(all_results['methods'])
        all_results['robustness_ranking'] = self._create_robustness_ranking(all_results['methods'])
        
        # Save results
        results_path = save_results(
            all_results,
            os.path.join(self.output_dirs['results'], 'attacked_benchmark_results'),
            format='json'
        )
        
        logging.info(f"Attacked benchmark completed. Results saved to {results_path}")
        return all_results
    
    def _create_attack_summary(self, methods_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of attack effectiveness across all methods"""
        attack_summary = {}
        
        for attack_name in self.attacks.keys():
            success_rates = []
            for method_name, results in methods_results.items():
                if ('error' not in results and 
                    'attack_results' in results and
                    attack_name in results['attack_results']):
                    success_rate = results['attack_results'][attack_name].get('success_rate', 0.0)
                    success_rates.append(success_rate)
            
            if success_rates:
                attack_summary[attack_name] = {
                    'avg_success_rate': float(np.mean(success_rates)),
                    'max_success_rate': float(np.max(success_rates)),
                    'min_success_rate': float(np.min(success_rates)),
                    'methods_tested': len(success_rates)
                }
        
        return attack_summary
    
    def _create_robustness_ranking(self, methods_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create ranking of methods by robustness (lower attack success = higher robustness)"""
        method_robustness = []
        
        for method_name, results in methods_results.items():
            if 'error' not in results and 'attack_results' in results:
                # Calculate average attack success rate
                success_rates = []
                for attack_name, attack_result in results['attack_results'].items():
                    if isinstance(attack_result, dict) and 'success_rate' in attack_result:
                        success_rates.append(attack_result['success_rate'])
                
                if success_rates:
                    avg_success_rate = np.mean(success_rates)
                    robustness_score = 1.0 - avg_success_rate  # Higher robustness = lower success rate
                    
                    method_robustness.append({
                        'method': method_name,
                        'robustness_score': float(robustness_score),
                        'avg_attack_success_rate': float(avg_success_rate)
                    })
        
        # Sort by robustness score (descending)
        method_robustness.sort(key=lambda x: x['robustness_score'], reverse=True)
        return method_robustness


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ImageProtectionBench - Attacked Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config['output']['base_dir'] = args.output_dir
    
    # Run attacked benchmark
    try:
        benchmark = AttackedImageProtectionBenchmark(config)
        results = benchmark.run_attacked_benchmark()
        
        print("\n" + "="*60)
        print("ATTACKED BENCHMARK COMPLETED")
        print("="*60)
        
        # Print robustness ranking
        ranking = results['robustness_ranking']
        if ranking:
            print("\nRobustness Ranking:")
            for i, method in enumerate(ranking, 1):
                print(f"{i}. {method['method']}: {method['robustness_score']:.3f}")
        
        # Print attack summary
        attack_summary = results['attack_summary']
        if attack_summary:
            print("\nAttack Effectiveness Summary:")
            for attack_name, summary in attack_summary.items():
                print(f"{attack_name}: {summary['avg_success_rate']:.3f} avg success rate")
        
    except Exception as e:
        logging.error(f"Attacked benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
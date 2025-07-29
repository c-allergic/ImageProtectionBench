"""
Metrics Module for ImageProtectionBench

This module provides various metrics for evaluating image protection methods:
- Image quality metrics (PSNR, SSIM, LPIPS)
- Video quality metrics (VBench, FVD, Temporal Consistency)
- Attack effectiveness metrics (CLIP Score, Attack Success Rate)
- Timing metrics (TimeMetric, BatchTimingMetric, timeit decorator)
"""

from .base import BaseMetric, ImageQualityMetric, VideoQualityMetric, EffectivenessMetric, TimeMetric
from .image_quality import PSNRMetric, SSIMMetric, LPIPSMetric
from .video_quality import VBenchMetric # , FVDMetric, TemporalConsistencyMetric
from .attack_effectiveness import CLIPScoreMetric # , AttackSuccessRateMetric, ProtectionRobustnessMetric


__all__ = [
    # Base classes
    'BaseMetric',
    'ImageQualityMetric',
    'VideoQualityMetric', 
    'EffectivenessMetric',
    'TimeMetric',
    # Image quality metrics
    'PSNRMetric',
    'SSIMMetric', 
    'LPIPSMetric',
    # Video quality metrics
    'VBenchMetric',
    # 'FVDMetric',
    # 'TemporalConsistencyMetric',
    # Attack effectiveness metrics
    'CLIPScoreMetric',
    # 'AttackSuccessRateMetric',
    # 'ProtectionRobustnessMetric',
] 
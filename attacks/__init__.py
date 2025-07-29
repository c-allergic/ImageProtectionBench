"""
Attack Methods Module for ImageProtectionBench

This module contains various attack methods to test the robustness 
of image protection algorithms against adversarial techniques.
"""

from .noise import GaussianNoiseAttack, SaltPepperAttack
from .jpeg_compression import JPEGCompressionAttack
from .geometric_attack import RotationAttack, CropAttack, ScalingAttack, WaveAttack

__all__ = [
    # Noise attacks
    'GaussianNoiseAttack',
    'SaltPepperAttack', 
    # Compression attacks
    'JPEGCompressionAttack',
    # Geometric attacks
    'RotationAttack',
    'CropAttack', 
    'ScalingAttack',
    'WaveAttack'
] 
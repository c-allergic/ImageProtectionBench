"""
Attack Methods Module for ImageProtectionBench

This module contains various attack methods to test the robustness 
of image protection algorithms against adversarial techniques.
"""

from .base import BaseAttack
from .noise import GaussianNoiseAttack, SaltPepperAttack
from .jpeg_compression import JPEGCompressionAttack
from .geometric_attack import GeometricAttack, RotationAttack, CropAttack, ScalingAttack

__all__ = [
    'BaseAttack',
    # Noise attacks
    'GaussianNoiseAttack',
    'SaltPepperAttack', 
    # Compression attacks
    'JPEGCompressionAttack',
    # Geometric attacks
    'GeometricAttack',
    'RotationAttack',
    'CropAttack', 
    'ScalingAttack'
] 
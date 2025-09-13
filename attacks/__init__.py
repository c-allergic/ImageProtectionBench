"""
Attack Methods Module for ImageProtectionBench

Now it only contains WAVES distortion attacks.
"""

from .distortions import (
    RotationAttack,
    ResizedCropAttack, 
    ErasingAttack,
    BrightnessAttack,
    ContrastAttack,
    BlurringAttack,
    NoiseAttack,
    SaltPepperAttack,
    CompressionAttack
)

__all__ = [
    # WAVES distortion attacks
    'RotationAttack',
    'ResizedCropAttack',
    'ErasingAttack', 
    'BrightnessAttack',
    'ContrastAttack',
    'BlurringAttack',
    'NoiseAttack',
    'SaltPepperAttack',
    'CompressionAttack'
]
"""
ExpGuard Protection Module - Refactored

This module implements the ExpGuard protection algorithm for defending against
adversarial image-to-video generation attacks. It uses attention-based loss
functions and DCT frequency domain perturbations.

Key Features:
- Target-based optimization strategy (steering toward meaningless outputs)
- DCT mid-frequency domain perturbation for JPEG robustness
- YCbCr color space manipulation (only Cb/Cr channels)
- Modular architecture with clean separation of concerns
- Type-hinted interfaces and comprehensive documentation
"""

import torch
import torch.nn as nn
import kornia.color as K_color
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import copy
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List
from PIL import Image

# Import refactored modules
from .dct_ops import DifferentiableDCT, create_mid_freq_mask
from .wan_wrapper import WanModelWrapper
from .losses import AttackLoss, AttentionHook, HookRegistrar
from .base import ProtectionBase

logging.getLogger().setLevel(logging.ERROR)


@dataclass
class AttackConfig:
    """
    Configuration for the ExpGuard attack.
    
    Attributes:
        num_steps: Number of optimization iterations
        learning_rate: Initial learning rate for Adam optimizer
        epsilon: Maximum perturbation magnitude (in [0, 1] range)
        mid_freq_ratio_low: Lower bound of mid-frequency band
        mid_freq_ratio_high: Upper bound of mid-frequency band
        weight_target: Weight for target (black image) loss
        weight_baseline: Weight for baseline (normal prompt) loss
        weight_constraint: Weight for constraint loss
        timestep_min: Minimum timestep for random sampling (0-1000)
        timestep_max: Maximum timestep for random sampling (0-1000)
        use_random_timestep: Whether to use random timestep sampling
        resample_timestep_per_step: Whether to resample timestep at each optimization step
        y_channel_weight: Weight for Y channel perturbation (relative to CbCr)
    """
    num_steps: int = 50
    learning_rate: float = 1e-2
    epsilon: float = 16.0 / 255.0
    mid_freq_ratio_low: float = 0.1
    mid_freq_ratio_high: float = 0.5
    weight_target: float = 1.0
    weight_baseline: float = 1.0
    weight_constraint: float = 10.0
    timestep_min: int = 200
    timestep_max: int = 800
    use_random_timestep: bool = True
    resample_timestep_per_step: bool = False
    y_channel_weight: float = 0.3


class ExpGuard(ProtectionBase):
    """
    ExpGuard protection algorithm - Target-based attention attack.
    
    This class implements a sophisticated defense mechanism against adversarial
    image-to-video generation. Instead of simply maximizing distance from the
    original output, it steers the perturbed image toward a meaningless target
    (outputs from a black image).
    
    The attack operates in the DCT frequency domain on YCbCr color channels,
    specifically targeting mid-frequency components in Cb/Cr channels. This
    approach provides robustness to JPEG compression while maintaining visual
    imperceptibility.
    
    Key Features:
    - Random timestep sampling for improved robustness across diffusion steps
    - Gradient checkpointing for memory efficiency
    - DCT mid-frequency perturbation for JPEG robustness
    - Optional Y (luminance) channel perturbation for stronger attacks
    
    Architecture:
    1. DifferentiableDCT: Handles DCT/IDCT transformations
    2. WanModelWrapper: Manages model loading and preprocessing
    3. AttackLoss: Computes optimization objectives
    4. AttentionHook: Captures intermediate activations
    
    Example:
        >>> config = AttackConfig(
        ...     num_steps=100,
        ...     epsilon=20/255,
        ...     use_random_timestep=True,
        ...     timestep_min=200,
        ...     timestep_max=800,
        ...     y_channel_weight=0.3
        ... )
        >>> expguard = ExpGuard(device='cuda', config=config)
        >>> protected = expguard.protect(pil_image)
    
    Attributes:
        device: Device for computation ('cuda' or 'cpu')
        config: Attack configuration parameters
        dct_op: Differentiable DCT operator
        model_wrapper: Wan model wrapper
        loss_calculator: Loss function calculator
        hook_manager: Attention hook manager
    """
    
    def __init__(
        self,
        device: str = "cuda",
        config: Optional[AttackConfig] = None,
        **kwargs
    ):
        """
        Initialize ExpGuard protection.
        
        Args:
            device: Device for computation
            config: Attack configuration (uses defaults if None)
            **kwargs: Additional arguments passed to ProtectionBase
        """
        self.device = device
        
        # Store config early (before super().__init__) so _setup_model can access it
        # We need to use a temporary variable because parent will overwrite self.config
        self._attack_config = config if config is not None else AttackConfig()
        
        # Initialize components
        self.dct_op = DifferentiableDCT()
        self.model_wrapper: Optional[WanModelWrapper] = None
        self.loss_calculator = AttackLoss()
        self.hook_manager = AttentionHook()
        
        # Prompts for different targets
        self.explosion_prompt = (
            "massive explosion with intense fireballs and shockwaves, "
            "violent blast with debris and fragments flying in all directions, "
            "dramatic orange and red flames engulfing the scene, "
            "thick black smoke billowing upwards, "
            "destruction and chaos with shattered objects, "
            "high contrast lighting from the explosion, "
            "cinematic action scene, photorealistic, highly detailed"
        )
        
        self.normal_prompt = (
            "peaceful and serene natural scene with soft gentle movement, "
            "calm atmosphere with warm natural lighting, "
            "tranquil environment with subtle changes, "
            "harmonious composition with balanced colors, "
            "quiet and stable setting"
        )
        
        # Call parent __init__ (parent sets self.config = kwargs as dict)
        super().__init__(device=device, **kwargs)
        
        # Override self.config with dataclass AFTER parent init
        # This prevents parent's dict assignment from overwriting our dataclass
        self.config = self._attack_config
        print(f"ExpGuard config: {self.config}")
    
    def _setup_model(self):
        """
        Initialize the Wan model wrapper and register hooks.
        
        This method is called by the parent ProtectionBase class during initialization.
        Note: We use self._attack_config here because self.config is still a dict at this point.
        """
        print("Loading WAN22Model...")
        self.model_wrapper = WanModelWrapper(
            device=self.device,
        )
        
        if self.model_wrapper.is_loaded():
            # Set hook manager in wrapper
            self.model_wrapper.set_hook_manager(self.hook_manager)
            
            # Register attention hooks
            HookRegistrar.register_attention_hooks(
                self.model_wrapper.wan_model,
                self.hook_manager
            )
            print("Attention hooks registered successfully")
        else:
            raise RuntimeError("Failed to load WAN22Model")
    
    def protect_multiple(
        self,
        images: List[Image.Image],
        **kwargs
    ) -> torch.Tensor:
        """
        Protect multiple images in batch.
        
        Args:
            images: List of PIL Image objects
            **kwargs: Additional arguments passed to protect()
            
        Returns:
            protected_images: [B, C, H, W] tensor of protected images
        """
        if not isinstance(images, list):
            raise ValueError(f"ExpGuard.protect_multiple() expects list of PIL Images, got {type(images)}")
        
        if len(images) == 0:
            raise ValueError("Image list cannot be empty")
        
        if not all(isinstance(img, Image.Image) for img in images):
            raise ValueError("All elements must be PIL Image objects")
        
        protected_images = []
        for img_pil in images:
            protected_single = self.protect(img_pil, **kwargs)
            protected_images.append(protected_single)
            
            # Memory cleanup after each image
            torch.cuda.empty_cache()
        
        return torch.stack(protected_images)
    
    @torch.enable_grad()
    def protect(self, image: Image.Image) -> torch.Tensor:
        """
        Protect a single image using target-based attention attack.
        
        This method implements the core protection algorithm:
        1. Preprocess image (resize, normalize)
        2. Generate target activations (from black image)
        3. Generate baseline activations (from original image)
        4. Optimize DCT-domain noise to minimize target loss
        5. Apply final perturbation and return protected image
        
        Args:
            image: PIL Image object (will be preprocessed internally)
            
        Returns:
            protected_image: [C, H, W] tensor in range [0, 1]
        """
        if not isinstance(image, Image.Image):
            raise ValueError(f"ExpGuard.protect() expects PIL Image, got {type(image)}")
        
        print("Preprocessing image...")
        I0_preprocessed = self.model_wrapper.preprocess_image(image)
        
        # Convert to RGB format for DCT processing
        # [C, 1, H, W] -> [1, C, H, W], range [-1, 1] -> [0, 1]
        I0_rgb = I0_preprocessed.squeeze(1).unsqueeze(0)
        I0_rgb = (I0_rgb + 1.0) / 2.0
        I0_rgb = torch.clamp(I0_rgb, 0, 1)
        
        _, C, H, W = I0_rgb.shape
        
        # Step 1: Convert to YCbCr and extract channels
        I0_ycbcr = K_color.rgb_to_ycbcr(I0_rgb)
        y0 = I0_ycbcr[:, 0:1]
        cb0 = I0_ycbcr[:, 1:2]
        cr0 = I0_ycbcr[:, 2:3]
        
        # Step 2: DCT transformation
        y0_dct = self.dct_op.forward(y0)
        cb0_dct = self.dct_op.forward(cb0)
        cr0_dct = self.dct_op.forward(cr0)
        
        # Step 3: Initialize learnable DCT-domain noise
        scale = 5.0  
        scale_y = scale * self.config.y_channel_weight  # Y通道使用较小的权重
        
        # 准备优化参数列表
        params_to_optimize = []
        
        delta_dct_cb = nn.Parameter(torch.randn_like(cb0_dct) * scale)
        delta_dct_cr = nn.Parameter(torch.randn_like(cr0_dct) * scale)
        delta_dct_y = nn.Parameter(torch.randn_like(y0_dct) * scale_y)
        params_to_optimize.extend([delta_dct_cb, delta_dct_cr, delta_dct_y])
        
        # Step 4: Create mid-frequency mask
        mid_freq_mask = create_mid_freq_mask(
            height=cb0_dct.shape[2],
            width=cb0_dct.shape[3],
            freq_ratio_low=self.config.mid_freq_ratio_low,
            freq_ratio_high=self.config.mid_freq_ratio_high,
            device=self.device
        )
        
        # Step 5: Compute sequence length and prepare timesteps
        F = 4
        seq_len = self.model_wrapper.compute_seq_len(F, H, W)
        
        # Random timestep sampling for robustness
        if self.config.use_random_timestep:
            timestep = torch.randint(
                self.config.timestep_min,
                self.config.timestep_max + 1,
                (1,),
                device=self.device
            )
            print(f"Using random timestep: {timestep.item()} (range: [{self.config.timestep_min}, {self.config.timestep_max}])")
        else:
            # Fallback to middle timestep
            timestep = torch.tensor([(self.config.timestep_min + self.config.timestep_max) // 2], device=self.device)
            print(f"Using fixed timestep: {timestep.item()}")
        
        t_expanded = timestep.expand(1, seq_len)
        
        # Step 6: Encode prompts
        print(f"Encoding prompts...")
        context_explosion = self.model_wrapper.encode_prompt(self.explosion_prompt)
        context_normal = self.model_wrapper.encode_prompt(self.normal_prompt)
        context_blank = self.model_wrapper.encode_prompt("")
        
        # Prepare context arguments
        arg_context_explosion = {'context': [context_explosion[0]], 'seq_len': seq_len}
        arg_context_normal = {'context': [context_normal[0]], 'seq_len': seq_len}
        arg_context_blank = {'context': [context_blank[0]], 'seq_len': seq_len}
        
        # Step 7: Generate video latent with noise (for temporal dimension)
        print("Generating video latent with temporal noise...")
        F_latent = (F - 1) // self.model_wrapper.vae_stride[0] + 1
        
        noise = torch.randn(
            self.model_wrapper.vae.model.z_dim,
            F_latent,
            H // self.model_wrapper.vae_stride[1],
            W // self.model_wrapper.vae_stride[2],
            dtype=torch.float32,
            device=self.device
        )
        
        # Import masks_like utility from Wan
        from wan.utils.utils import masks_like
        _, mask2 = masks_like([noise], zero=True)
        
        # Cache noise and mask for reuse
        self._noise = noise
        self._mask2 = mask2
        
        # Step 8: Generate baseline activations (normal prompt)
        print("Generating baseline activations (original + normal prompt)...")
        with torch.no_grad():
            latent_list = self.model_wrapper.vae.encode([I0_preprocessed])
            latent_0 = latent_list[0]
            latent_video = (1.0 - mask2[0]) * latent_0 + mask2[0] * noise
            
            self.hook_manager.clear_activations()
            latent_video = latent_video.to(torch.bfloat16)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                normal_output = self.model_wrapper.wan_model(
                    [latent_video],
                    t=t_expanded,
                    **arg_context_normal
                )
            
            baseline_activations = copy.deepcopy(self.hook_manager.activations)
            print("Baseline activations generated")
            
            # Memory cleanup
            del latent_list, latent_0, latent_video, normal_output
            torch.cuda.empty_cache()
        
        # Step 9: Generate target activations (black image)
        print("Generating target activations (black image)...")
        with torch.no_grad():
            black_normalized = (torch.zeros_like(I0_rgb) * 2.0 - 1.0).squeeze(0).unsqueeze(1)
            latent_black_list = self.model_wrapper.vae.encode([black_normalized])
            latent_black_single = latent_black_list[0]
            latent_black_video = (1.0 - mask2[0]) * latent_black_single + mask2[0] * noise
            
            self.hook_manager.clear_activations()
            latent_black_video = latent_black_video.to(torch.bfloat16)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                black_output = self.model_wrapper.wan_model(
                    [latent_black_video],
                    t=t_expanded,
                    **arg_context_blank
                )
            
            target_activations = copy.deepcopy(self.hook_manager.activations)
            print("Target activations generated")
            
            # Memory cleanup
            del latent_black_list, latent_black_single, latent_black_video
            del black_output, black_normalized
            torch.cuda.empty_cache()
        
        # Step 10: Optimization loop
        print(f"Starting optimization ({self.config.num_steps} steps)...")
        loss_history = {
            'total': [],
            'target': [],
            # 'baseline': [],
            'constraint': []
        }
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.num_steps,
            eta_min=self.config.learning_rate * 0.01
        )
        
        # Progress bar
        pbar = tqdm(range(self.config.num_steps), desc="Optimizing perturbation", ncols=100)
        
        for step in pbar:
            optimizer.zero_grad()
            
            # Resample timestep for each optimization step if enabled
            if self.config.use_random_timestep and self.config.resample_timestep_per_step:
                timestep = torch.randint(
                    self.config.timestep_min,
                    self.config.timestep_max + 1,
                    (1,),
                    device=self.device
                )
                t_expanded = timestep.expand(1, seq_len)
            
            # Apply mid-frequency noise in DCT domain
            cb_dct_perturbed = cb0_dct + delta_dct_cb * mid_freq_mask
            cr_dct_perturbed = cr0_dct + delta_dct_cr * mid_freq_mask
            
            # IDCT back to spatial domain
            cb_perturbed = self.dct_op.inverse(cb_dct_perturbed)
            cr_perturbed = self.dct_op.inverse(cr_dct_perturbed)
            
            y_dct_perturbed = y0_dct + delta_dct_y * mid_freq_mask
            y_perturbed = self.dct_op.inverse(y_dct_perturbed)
            
            # Reconstruct YCbCr and convert to RGB
            ycbcr_perturbed = torch.cat([y_perturbed, cb_perturbed, cr_perturbed], dim=1)
            I_perturbed_rgb = K_color.ycbcr_to_rgb(ycbcr_perturbed)
            I_perturbed_rgb = torch.clamp(I_perturbed_rgb, 0, 1)
            
            # Convert to Wan format and encode
            I_perturbed_normalized = (I_perturbed_rgb * 2.0 - 1.0).squeeze(0).unsqueeze(1)
            latent_perturbed_list = self.model_wrapper.vae.encode([I_perturbed_normalized])
            latent_perturbed_single = latent_perturbed_list[0]
            latent_perturbed_video = (1.0 - mask2[0]) * latent_perturbed_single + mask2[0] * noise
            
            # Forward pass through Wan model
            self.hook_manager.clear_activations()
            latent_perturbed_video = latent_perturbed_video.to(torch.bfloat16)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output = self.model_wrapper.wan_model(
                    [latent_perturbed_video],
                    t=t_expanded,
                    **arg_context_explosion
                )
            
            current_activations = self.hook_manager.activations
            
            # Compute losses
            delta_rgb = I_perturbed_rgb - I0_rgb
            weights = {
                'target': self.config.weight_target,
                # 'baseline': self.config.weight_baseline,
                # 'constraint': self.config.weight_constraint
            }
            
            losses, log_dict = self.loss_calculator.compute(
                target_activations,
                current_activations,
                baseline_activations,
                delta_rgb,
                self.config.epsilon,
                weights
            )
            
            # Backpropagation
            losses['total'].backward()
            optimizer.step()
            lr_scheduler.step()
            
            # Record losses
            for key in ['total', 'target']:
                loss_history[key].append(log_dict[key])
            
            # Memory cleanup
            del output, current_activations, latent_perturbed_list
            del latent_perturbed_single, latent_perturbed_video
            del I_perturbed_normalized, I_perturbed_rgb, ycbcr_perturbed
            del cb_dct_perturbed, cr_dct_perturbed, cb_perturbed, cr_perturbed
            del y_dct_perturbed, y_perturbed
            del losses, delta_rgb
            torch.cuda.empty_cache()
            
            # Update progress bar
            pbar.set_postfix({
                'Total': f'{log_dict["total"]:.4f}',
                'Target': f'{log_dict["target"]:.4f}',
                # 'Base': f'{log_dict["baseline"]:.4f}',
                # 'Cons': f'{log_dict["constraint"]:.4f}'
            })
        
        # Step 11: Generate final protected image
        print("Generating final protected image...")
        with torch.no_grad():
            cb_dct_perturbed = cb0_dct + delta_dct_cb * mid_freq_mask
            cr_dct_perturbed = cr0_dct + delta_dct_cr * mid_freq_mask
            
            cb_perturbed = self.dct_op.inverse(cb_dct_perturbed)
            cr_perturbed = self.dct_op.inverse(cr_dct_perturbed)
            
            y_dct_perturbed = y0_dct + delta_dct_y * mid_freq_mask
            y_perturbed = self.dct_op.inverse(y_dct_perturbed)
            
            ycbcr_perturbed = torch.cat([y_perturbed, cb_perturbed, cr_perturbed], dim=1)
            protected_rgb = K_color.ycbcr_to_rgb(ycbcr_perturbed)
            delta_final = protected_rgb - I0_rgb
            
            # Hard clipping to epsilon bounds
            delta_clipped = torch.clamp(delta_final, -self.config.epsilon, self.config.epsilon)
            protected_image = torch.clamp(I0_rgb + delta_clipped, 0, 1)
            
            print(f"Final perturbation range: [{delta_clipped.min():.4f}, {delta_clipped.max():.4f}]")
        
        # Cleanup
        del self._noise, self._mask2
        del baseline_activations, target_activations
        self.hook_manager.clear_activations()
        torch.cuda.empty_cache()
        
        # Plot loss curves
        self._plot_loss_curves(loss_history)
        
        return protected_image.squeeze(0).detach()
    
    def _plot_loss_curves(self, loss_history: Dict[str, List[float]]):
        """
        Plot and save loss curves for analysis.
        
        Args:
            loss_history: Dictionary containing loss values for each iteration
        """
        plt.figure(figsize=(18, 6))
        
        # Subplot 1: Total Loss
        plt.subplot(1, 4, 1)
        plt.plot(loss_history['total'], linewidth=2, color='#2E86AB')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Total Loss', fontsize=12)
        plt.title('Total Loss', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Target Loss
        plt.subplot(1, 4, 2)
        plt.plot(loss_history['target'], linewidth=2, color='#A23B72')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Target Loss', fontsize=12)
        plt.title('Target Loss (Black Image)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Baseline Divergence Loss
        # plt.subplot(1, 4, 3)
        # plt.plot(loss_history['baseline'], linewidth=2, color='#6C5CE7')
        # plt.xlabel('Iteration', fontsize=12)
        # plt.ylabel('Baseline Loss', fontsize=12)
        # plt.title('Baseline Divergence Loss', fontsize=14, fontweight='bold')
        # plt.grid(True, alpha=0.3)
        
        # Subplot 4: Constraint Loss
        # plt.subplot(1, 4, 4)
        # plt.plot(loss_history['constraint'], linewidth=2, color='#E74C3C')
        # plt.xlabel('Iteration', fontsize=12)
        # plt.ylabel('Constraint Loss', fontsize=12)
        # plt.title('Constraint Loss', fontsize=14, fontweight='bold')
        # plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to current working directory
        save_path = os.path.join(os.path.dirname(__file__), 'loss_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nLoss curves saved to: {save_path}")
        
        # Print final statistics
        print("\n=== Loss Statistics ===")
        print(f"Total Loss:      Initial={loss_history['total'][0]:.4f}, Final={loss_history['total'][-1]:.4f}")
        print(f"Target Loss:     Initial={loss_history['target'][0]:.4f}, Final={loss_history['target'][-1]:.4f}")
        # print(f"Baseline Loss:   Initial={loss_history['baseline'][0]:.4f}, Final={loss_history['baseline'][-1]:.4f}")
        # print(f"Constraint Loss: Initial={loss_history['constraint'][0]:.4f}, Final={loss_history['constraint'][-1]:.4f}")


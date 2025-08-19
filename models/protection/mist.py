import os
import numpy as np
from omegaconf import OmegaConf
import PIL
from PIL import Image
from einops import rearrange
import ssl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from typing import Dict, Any, Optional, List, Union
from .base import ProtectionBase

# 禁用SSL验证
ssl._create_default_https_context = ssl._create_unverified_context

try:
    from ldm.util import instantiate_from_config
    LDM_AVAILABLE = True
except ImportError:
    print("Warning: ldm module not found. MIST protection may not work properly.")
    instantiate_from_config = None
    LDM_AVAILABLE = False

class IdentityLoss(nn.Module):
    """
    An identity loss used for input fn for advertorch. To support semantic loss,
    the computation of the loss is implemented in class target_model.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x

class TargetModel(nn.Module):
    """
    A virtual model which computes the semantic and textural loss in forward function.
    """

    def __init__(self, model, condition: str, target_info: torch.Tensor = None, 
                 mode: int = 2, rate: int = 10000, input_size: int = 512):
        super().__init__()
        self.model = model
        self.condition = condition
        self.fn = nn.MSELoss(reduction="sum")
        self.target_info = target_info
        self.mode = mode
        self.rate = rate
        self.target_size = input_size
        # 设置device
        self.device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_components(self, x, no_loss=False):
        """
        Compute the semantic loss and the encoded information of the input.
        :return: encoded info of x, semantic loss
        """
        z = self.model.get_first_stage_encoding(self.model.encode_first_stage(x)).to(self.device)
        c = self.model.get_learned_conditioning(self.condition)
        if no_loss:
            loss = 0
        else:
            loss = self.model(z, c)[0]
        return z, loss

    def pre_process(self, x, target_size):
        processed_x = torch.zeros([x.shape[0], x.shape[1], target_size, target_size]).to(self.device)
        trans = transforms.RandomCrop(target_size)
        for p in range(x.shape[0]):
            processed_x[p] = trans(x[p])
        return processed_x

    def forward(self, x, components=False):
        """
        Compute the loss based on different mode.
        The textural loss shows the distance between the input image and target image in latent space.
        The semantic loss describles the semantic content of the image.
        :return: The loss used for updating gradient in the adversarial attack.
        """
        zx, loss_semantic = self.get_components(x, True)
        zy, _ = self.get_components(self.target_info, True)
        if self.mode != 1:
            _, loss_semantic = self.get_components(self.pre_process(x, self.target_size))
        if components:
            return self.fn(zx, zy), loss_semantic
        if self.mode == 0:
            return - loss_semantic
        elif self.mode == 1:
            return self.fn(zx, zy)
        else:
            return self.fn(zx, zy) - loss_semantic * self.rate

def perturb_iterative(xvar, yvar, predict, nb_iter, eps, eps_iter, loss_fn,
                      delta_init=None, minimize=False, ord=np.inf,
                      clip_min=0.0, clip_max=1.0, mask=None):
    """
    Iteratively maximize the loss over the input. It is a shared method for
    iterative attacks including IterativeGradientSign, LinfPGD, etc.
    """
    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(xvar)

    delta.requires_grad_()
    for ii in range(nb_iter):
        if mask is None:
            outputs = predict(xvar + delta)
        else:
            outputs = predict(xvar + delta * mask)
        loss = loss_fn(outputs, yvar)
        if minimize:
            loss = -loss

        loss.backward()
        if ord == np.inf:
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + eps_iter * grad_sign
            delta.data = torch.clamp(delta.data, -eps, eps)
            delta.data = torch.clamp(xvar.data + delta.data, clip_min, clip_max) - xvar.data
        else:
            raise NotImplementedError("Only ord = inf has been implemented")
        delta.grad.data.zero_()
    
    if mask is None:
        x_adv = torch.clamp(xvar + delta, clip_min, clip_max)
    else:
        x_adv = torch.clamp(xvar + delta * mask, clip_min, clip_max)
    return x_adv

class LinfPGDAttack:
    """
    PGD Attack with order=Linf
    """
    
    def __init__(self, predict, loss_fn=None, eps=0.3, nb_iter=40,
                 eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
                 targeted=False):
        self.predict = predict
        self.loss_fn = loss_fn
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted

    def perturb(self, x, y, mask=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.
        """
        return perturb_iterative(
            xvar=x, yvar=y, predict=self.predict, nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.eps_iter, loss_fn=self.loss_fn,
            minimize=self.targeted, ord=np.inf, clip_min=self.clip_min, 
            clip_max=self.clip_max, mask=mask
        )

class Mist(ProtectionBase):
    def __init__(self, 
                 epsilon: int = 16,
                 steps: int = 100, 
                 alpha: int = 1,
                 input_size: int = 512,
                 object: bool = False,
                 seed: int = 23,
                 mode: int = 2,
                 rate: int = 10000,
                 config_path: Optional[str] = None,
                 ckpt_path: Optional[str] = None,
                 target_image_path: Optional[str] = None,
                 **kwargs):
        """
        Prepare the config and the model used for generating adversarial examples.
        """
        # Set default paths - 使用相对于当前工作目录的路径
        if config_path is None:
            config_path = 'configs/stable-diffusion/v1-inference-attack.yaml'
        if ckpt_path is None:
            ckpt_path = 'models/ldm/stable-diffusion-v1/model.ckpt'
        if target_image_path is None:
            target_image_path = os.path.join(os.path.dirname(__file__), 'MIST.png')
            
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.target_image_path = target_image_path
        
        # Store parameters
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha
        self.input_size = input_size
        self.object = object
        self.seed = seed
        self.mode = mode
        self.rate = rate
        
        super().__init__(**kwargs)
    
    def _setup_model(self):
        """Initialize MIST model following the original implementation"""
        if not LDM_AVAILABLE:
            raise ImportError("ldm module not available. MIST requires Stable Diffusion model.")
            
        # Set seed for reproducibility
        seed_everything(self.seed)
        
        # Load configuration
        config = OmegaConf.load(self.config_path)
        
        # Load Stable Diffusion model
        model = self._load_model_from_config(config, self.ckpt_path)
        
        # Create identity loss function
        fn = IdentityLoss()
        
        # Set prompt templates
        imagenet_templates_small_style = ['a painting']
        imagenet_templates_small_object = ['a photo']
        
        if self.object:
            imagenet_templates_small = imagenet_templates_small_object
        else:
            imagenet_templates_small = imagenet_templates_small_style
        
        # Create target model
        input_prompt = [imagenet_templates_small[0] for i in range(1)]
        net = TargetModel(model, input_prompt, mode=self.mode, rate=self.rate)
        net.eval()
        
        # Store components
        self.model = model
        self.fn = fn
        self.net = net
        
        # Load target image
        self.target_image = self._load_target_image()
    
    def _load_model_from_config(self, config, ckpt_path):
        """
        Load model from the config and the ckpt path.
        """
        print(f"Loading model from {ckpt_path}")

        pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]

        # Support loading weight from NovelAI
        if "state_dict" in sd:
            import copy
            sd_copy = copy.deepcopy(sd)
            for key in sd.keys():
                if key.startswith('cond_stage_model.transformer') and not key.startswith('cond_stage_model.transformer.text_model'):
                    newkey = key.replace('cond_stage_model.transformer', 'cond_stage_model.transformer.text_model', 1)
                    sd_copy[newkey] = sd[key]
                    del sd_copy[key]
            sd = sd_copy

        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)

        if len(m) > 0:
            print("missing keys:")
            print(m)
        if len(u) > 0:
            print("unexpected keys:")
            print(u)

        model.to(self.device)
        model.eval()
        return model
    
    def _load_target_image(self):
        """Load target image for MIST"""
        target_img = Image.open(self.target_image_path).convert('RGB')
        target_img = target_img.resize((self.input_size, self.input_size))
        
        # Convert to tensor
        target_array = np.array(target_img).astype(np.float32) / 127.5 - 1.0
        target_tensor = torch.from_numpy(target_array).permute(2, 0, 1)
        
        return target_tensor.to(self.device)
    
    def protect(self, image: torch.Tensor, 
                target_image: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        Process the input image and generate the misted image.
        """
        # Convert image to tensor and normalize
        if isinstance(image, np.ndarray):
            img = torch.from_numpy(image).float()
        else:
            img = image.float()
        
        # Resize if needed
        if img.shape[-1] != self.input_size or img.shape[-2] != self.input_size:
            transform = transforms.Resize((self.input_size, self.input_size))
            img = transform(img)
        
        # Normalize to [-1, 1]
        img = img * 2.0 - 1.0
        img = img.to(self.device)
        
        # Prepare target image
        if target_image is not None:
            tar_img = target_image.float() * 2.0 - 1.0
            tar_img = tar_img.to(self.device)
        else:
            tar_img = self.target_image
        
        # Create data tensors
        data_source = img.unsqueeze(0)  # [1, C, H, W]
        target_info = tar_img.unsqueeze(0)  # [1, C, H, W]
        
        # Update target model attributes
        self.net.target_info = target_info
        self.net.target_size = self.input_size
        self.net.mode = self.mode
        self.net.rate = self.rate
        
        # Create label
        label = torch.zeros_like(data_source)
        
        # Execute PGD attack
        attack = LinfPGDAttack(
            predict=self.net,
            loss_fn=self.fn,
            eps=self.epsilon/255.0 * (1-(-1)),
            nb_iter=self.steps,
            eps_iter=self.alpha/255.0 * (1-(-1)),
            clip_min=-1.0,
            clip_max=1.0,
            targeted=True
        )
        
        attack_output = attack.perturb(data_source, label)
        
        # Post-process
        output = attack_output[0]
        save_adv = torch.clamp((output + 1.0) / 2.0, min=0.0, max=1.0).detach()
        
        # Resize back to original size if needed
        if save_adv.shape[-1] != image.shape[-1] or save_adv.shape[-2] != image.shape[-2]:
            transform = transforms.Resize((image.shape[-2], image.shape[-1]))
            save_adv = transform(save_adv)
        
        return save_adv
    

    def protect_multiple(
        self, 
        images: Union[torch.Tensor, List[torch.Tensor]], 
        target_image: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Process multiple images using MIST algorithm.
        """
        return super().protect_multiple(
            images, target_image=target_image, **kwargs
        )


 
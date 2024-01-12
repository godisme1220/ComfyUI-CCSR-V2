import os
import torch

import torch
import einops
from torch.nn import functional as F

from omegaconf import OmegaConf

from .model.q_sampler import SpacedSampler
from .model.ccsr_stage1 import ControlLDM
#from .model.cond_fn import MSEGuidance

from .utils.common import instantiate_from_config, load_state_dict

script_directory = os.path.dirname(os.path.abspath(__file__))

class CCSR_Upscale:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {  
            "image": ("IMAGE", ),
            "steps": ("INT", {"default": 45, "min": 1, "max": 4096, "step": 1}),
            "t_max": ("FLOAT", {"default": 0.6667,"min": 0, "max": 1, "step": 0.01}),
            "t_min": ("FLOAT", {"default": 0.3333,"min": 0, "max": 1, "step": 0.01}),
            "tile_size": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
            "tile_stride": ("INT", {"default": 256, "min": 1, "max": 4096, "step": 1}),
            "tiled": ("BOOLEAN", {"default": False}),
            "color_fix_type": (
            [   
                'none',
                'adain',
                'wavelet',
            ], {
               "default": 'adain'
            }),  
            },
            
            }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("upscaled_image",)
    FUNCTION = "process"

    CATEGORY = "CCSR"

    @torch.no_grad()
    def process(self, image, steps, t_max, t_min, tiled,tile_size, tile_stride, color_fix_type):
        checkpoint_path = os.path.join(script_directory, "../../models/checkpoints/real-world_ccsr.ckpt")
        config_path = os.path.join(script_directory, "configs/model/ccsr_stage2.yaml")

        config = OmegaConf.load(config_path)
        model = instantiate_from_config(config)


        load_state_dict(model, torch.load(checkpoint_path, map_location="cpu"), strict=True)
        # reload preprocess model if specified

        model.freeze()
        model.to("cuda")

        sampler = SpacedSampler(model, var_type="fixed_small")

        # Assuming 'image' is a PyTorch tensor with shape [B, H, W, C] and you want to resize it.
        B, H, W, C = image.shape

        # Calculate the new height and width, rounding down to the nearest multiple of 64.
        new_height = H // 64 * 64
        new_width = W // 64 * 64

        # Reorder to [B, C, H, W] before using interpolate.
        image = image.permute(0, 3, 1, 2).contiguous()

        # Resize the image tensor.
        resized_image = F.interpolate(image, size=(new_height, new_width), mode='bicubic', align_corners=False)

        # Move the tensor to the GPU.
        resized_image = resized_image.to("cuda")
        strength = 1.0
        model.control_scales = [strength] * 13
        cond_fn = None
        height, width = resized_image.size(-2), resized_image.size(-1)
        shape = (1, 4, height // 8, width // 8)
        x_T = torch.randn(shape, device=model.device, dtype=torch.float32)

        if not tiled:
            samples = sampler.sample_ccsr(
                steps=steps, t_max=t_max, t_min=t_min, shape=shape, cond_img=resized_image,
                positive_prompt="", negative_prompt="", x_T=x_T,
                cfg_scale=1.0, cond_fn=cond_fn,
                color_fix_type=color_fix_type
            )
        else:
            samples = sampler.sample_with_mixdiff_ccsr(
                tile_size=tile_size, tile_stride=tile_stride,
                steps=steps, t_max=t_max, t_min=t_min, shape=shape, cond_img=resized_image,
                positive_prompt="", negative_prompt="", x_T=x_T,
                cfg_scale=1.0, cond_fn=cond_fn,
                color_fix_type=color_fix_type
            )

        x_samples = samples.clamp(0, 1)
        x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c")).cpu()
        return (x_samples,)

NODE_CLASS_MAPPINGS = {
    "CCSR_Upscale": CCSR_Upscale,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CCSR_Upscale": "CCSR_Upscale",
}
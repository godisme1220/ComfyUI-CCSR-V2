import os
import torch
from torch.nn import functional as F
from contextlib import nullcontext
from omegaconf import OmegaConf

from .model.q_sampler import SpacedSampler
from .model.ccsr_stage1 import ControlLDM

from .utils.common import instantiate_from_config, load_state_dict

import comfy.model_management

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
            "use_fp16": ("BOOLEAN", {"default": True}), 
            },
            
            }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("upscaled_image",)
    FUNCTION = "process"

    CATEGORY = "CCSR"

    @torch.no_grad()
    def process(self, image, steps, t_max, t_min, tiled,tile_size, tile_stride, color_fix_type, use_fp16):
        checkpoint_path = os.path.join(script_directory, "../../models/checkpoints/real-world_ccsr.ckpt")
        config_path = os.path.join(script_directory, "configs/model/ccsr_stage2.yaml")

        config = OmegaConf.load(config_path)
        model = instantiate_from_config(config)
        device = comfy.model_management.get_torch_device()

        load_state_dict(model, torch.load(checkpoint_path, map_location="cpu"), strict=True)
        # reload preprocess model if specified

        model.freeze()
        model.to(device)
        if (use_fp16):
            model.half()
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
        resized_image = resized_image.to(device)
        strength = 1.0
        model.control_scales = [strength] * 13
        cond_fn = None
        height, width = resized_image.size(-2), resized_image.size(-1)
        shape = (1, 4, height // 8, width // 8)
        x_T = torch.randn(shape, device=model.device, dtype=torch.float32)
        with torch.autocast(comfy.model_management.get_autocast_device(device), dtype=model.dtype) if use_fp16 else nullcontext():
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

        # Original dimensions
        original_height, original_width = H, W

        # Compute the aspect ratio
        aspect_ratio = original_width / original_height

        # Your new height after processing
        processed_height = samples.size(2)

        # Calculate the target width using the aspect ratio
        target_width = int(processed_height * aspect_ratio)

        # Resize while keeping aspect ratio
        resized_back_image = F.interpolate(samples, size=(processed_height, target_width), mode='bicubic', align_corners=False)

        # If necessary, rearrange from [B, C, H, W] back to [B, H, W, C] and move to CPU
        resized_back_image = resized_back_image.permute(0, 2, 3, 1).cpu()
        return(resized_back_image,)

NODE_CLASS_MAPPINGS = {
    "CCSR_Upscale": CCSR_Upscale,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CCSR_Upscale": "CCSR_Upscale",
}
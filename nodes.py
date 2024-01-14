import os
import torch
from torch.nn import functional as F
from contextlib import nullcontext
from omegaconf import OmegaConf

from .model.q_sampler import SpacedSampler
from .model.ccsr_stage1 import ControlLDM

from .utils.common import instantiate_from_config, load_state_dict

import comfy.model_management
import folder_paths
from nodes import ImageScaleBy
from nodes import ImageScale


script_directory = os.path.dirname(os.path.abspath(__file__))

class CCSR_Upscale:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ccsr_model": ("CCSRMODEL", ),
            "image": ("IMAGE", ),
            "resize_method": (s.upscale_methods, {"default": "lanczos"}),
            "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 20.0, "step": 0.01}),
            "steps": ("INT", {"default": 45, "min": 2, "max": 4096, "step": 1}),
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
            "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
            
            }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("upscaled_image",)
    FUNCTION = "process"

    CATEGORY = "CCSR"

    @torch.no_grad()
    def process(self, ccsr_model, image, resize_method, scale_by, steps, t_max, t_min, tiled,tile_size, tile_stride, color_fix_type, keep_model_loaded):
        comfy.model_management.unload_all_models()
        device = comfy.model_management.get_torch_device()
        config_path = os.path.join(script_directory, "configs/model/ccsr_stage2.yaml")
        dtype = torch.float16 if comfy.model_management.should_use_fp16() and not comfy.model_management.is_device_mps(device) else torch.float32
        
        if not hasattr(self, "model") or self.model is None:
            config = OmegaConf.load(config_path)
            self.model = instantiate_from_config(config)
            
            load_state_dict(self.model, torch.load(ccsr_model, map_location="cpu"), strict=True)
            # reload preprocess model if specified

            self.model.freeze()
            self.model.to(device, dtype=dtype)
        sampler = SpacedSampler(self.model, var_type="fixed_small")

        image, = ImageScaleBy.upscale(self, image, resize_method, scale_by)

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
        self.model.control_scales = [strength] * 13
        cond_fn = None
        height, width = resized_image.size(-2), resized_image.size(-1)
        shape = (1, 4, height // 8, width // 8)
        x_T = torch.randn(shape, device=self.model.device, dtype=dtype)
        autocast_condition = dtype == torch.float16 and not comfy.model_management.is_device_mps(device)
        with torch.autocast(comfy.model_management.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
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

        original_height, original_width = H, W  
        processed_height = samples.size(2)
        target_width = int(processed_height * (original_width / original_height))

        resized_back_image, = ImageScale.upscale(self, samples.permute(0, 2, 3, 1).cpu(), "lanczos", target_width, processed_height, crop="disabled")

        if not keep_model_loaded:
            self.model = None            
            comfy.model_management.soft_empty_cache()
        return(resized_back_image,)

class CCSR_Model_Select:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),                                             
                             }}
    RETURN_TYPES = ("CCSRMODEL",)
    RETURN_NAMES = ("ccsr_model",)
    FUNCTION = "load_ccsr_checkpoint"

    CATEGORY = "CCSR"

    def load_ccsr_checkpoint(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        
        return (ckpt_path,)
    
NODE_CLASS_MAPPINGS = {
    "CCSR_Upscale": CCSR_Upscale,
    "CCSR_Model_Select": CCSR_Model_Select
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CCSR_Upscale": "CCSR_Upscale",
    "CCSR_Model_Select": "CCSR_Model_Select"
}
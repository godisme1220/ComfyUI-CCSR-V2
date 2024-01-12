import os
import sys
import torch
import numpy as np

import numpy as np
import torch
import einops
from torch.nn import functional as F
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf

from .model.q_sampler import SpacedSampler
from .model.ccsr_stage1 import ControlLDM
from .model.cond_fn import MSEGuidance

from .utils.common import instantiate_from_config, load_state_dict

script_directory = os.path.dirname(os.path.abspath(__file__))
project_directory = os.path.join(script_directory, '..')  # Adjust the path as necessary
sys.path.insert(0, project_directory)

class CCSR_Upscale:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {  
            "image": ("IMAGE", ),
            #"sr_scale": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1}),
            "steps": ("INT", {"default": 10, "min": 1, "max": 4096, "step": 1}),
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
        """
        Apply CCSR model on a list of low-quality images.

        Args:
            model (ControlLDM): Model.
            control_imgs (List[np.ndarray]): A list of low-quality images (HWC, RGB, range in [0, 255]).
            steps (int): Sampling steps.
            t_max (float):
            t_min (float):
            strength (float): Control strength. Set to 1.0 during training.
            color_fix_type (str): Type of color correction for samples.
            cond_fn (Guidance | None): Guidance function that returns gradient to guide the predicted x_0.
            tiled (bool): If specified, a patch-based sampling strategy will be used for sampling.
            tile_size (int): Size of patch.
            tile_stride (int): Stride of sliding patch.

        Returns:
            preds (List[np.ndarray]): Restoration results (HWC, RGB, range in [0, 255]).
        """
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
        print(resized_image.shape)
        print(x_T.shape)
        if not tiled:
            # samples = sampler.sample_ccsr_stage1(
            #     steps=steps, t_max=t_max, shape=shape, cond_img=control,
            #     positive_prompt="", negative_prompt="", x_T=x_T,
            #     cfg_scale=1.0, cond_fn=cond_fn,
            #     color_fix_type=color_fix_type
            # )
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
        #x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
        x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c")).cpu()
        print(x_samples.shape)
        return (x_samples,)


NODE_CLASS_MAPPINGS = {
    "CCSR_Upscale": CCSR_Upscale,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CCSR_Upscale": "CCSR_Upscale",
}
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from torchvision import transforms
from diffusers import (
    AutoencoderKL,
    UniPCMultistepScheduler,
    DPMSolverMultistepScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from .pipelines.pipeline_ccsr import StableDiffusionControlNetPipeline
from .myutils.wavelet_color_fix import wavelet_color_fix, adain_color_fix
from .models.controlnet import ControlNetModel
from folder_paths import models_dir

class CCSRNode:
    def __init__(self):
        self.loaded_models = {}
        self.download_path = os.path.join(models_dir, "ccsr")
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pretrained_model_path": ("STRING", {"default": "stable-diffusion-2-1-base"}),
                "controlnet_model_path": ("STRING", {"default": "controlnet"}),
                "vae_model_path": ("STRING", {"default": "vae"}),
                "sample_method": (["ddpm", "unipcmultistep", "dpmmultistep"], {"default": "ddpm"}),
                "num_inference_steps": ("INT", {"default": 6, "min": 1, "max": 100}),
                "t_max": ("FLOAT", {"default": 0.6667, "min": 0.0, "max": 1.0}),
                "t_min": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "start_point": (["lr", "noise"], {"default": "lr"}),
                "start_steps": ("INT", {"default": 999, "min": 0}),
                "process_size": ("INT", {"default": 512, "min": 64}),
                "guidance_scale": ("FLOAT", {"default": 4.5, "min": 0.0}),
                "upscale": ("INT", {"default": 4, "min": 1}),
                "align_method": (["wavelet", "adain", "nofix"], {"default": "adain"}),
                "added_prompt": ("STRING", {"default": "clean, high-resolution, 8k"}),
                "negative_prompt": ("STRING", {"default": "blurry, dotted, noise, raster lines, unclear, lowres, over-smoothed"}),
            },
            "optional": {
                "tile_diffusion": ("BOOLEAN", {"default": True}),
                "tile_diffusion_size": ("INT", {"default": 512, "min": 64}),
                "tile_diffusion_stride": ("INT", {"default": 256, "min": 32}),
                "tile_vae": ("BOOLEAN", {"default": True}),
                "vae_decoder_tile_size": ("INT", {"default": 224, "min": 64}),
                "vae_encoder_tile_size": ("INT", {"default": 1024, "min": 64}),
                "use_vae_encode_condition": ("BOOLEAN", {"default": True}),
                "skip_smaller_than": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "CCSRV2"

    def load_pipeline(self, args):
        # Create paths using models_dir
        pretrained_path = os.path.join(self.download_path, args["pretrained_model_path"])
        controlnet_path = os.path.join(self.download_path, args["controlnet_model_path"])
        vae_path = os.path.join(self.download_path, args["vae_model_path"]) if args["vae_model_path"] else pretrained_path

        scheduler_mapping = {
            'unipcmultistep': UniPCMultistepScheduler,
            'ddpm': DDPMScheduler,
            'dpmmultistep': DPMSolverMultistepScheduler,
        }

        try:
            scheduler_cls = scheduler_mapping[args["sample_method"]]
        except KeyError:
            raise ValueError(f"Invalid sample_method: {args['sample_method']}")

        scheduler = scheduler_cls.from_pretrained(pretrained_path, subfolder="scheduler")
        text_encoder = CLIPTextModel.from_pretrained(pretrained_path, subfolder="text_encoder")
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_path, subfolder="tokenizer")
        feature_extractor = CLIPImageProcessor.from_pretrained(os.path.join(pretrained_path, "feature_extractor"))
        unet = UNet2DConditionModel.from_pretrained(pretrained_path, subfolder="unet")
        controlnet = ControlNetModel.from_pretrained(controlnet_path, subfolder="controlnet")
        vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae")

        # Freeze models
        for model in [vae, text_encoder, unet, controlnet]:
            model.requires_grad_(False)

        # Initialize pipeline
        pipeline = StableDiffusionControlNetPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=None,
            requires_safety_checker=False,
        )

        if args.get("tile_vae", True):
            pipeline._init_tiled_vae(
                encoder_tile_size=args.get("vae_encoder_tile_size", 1024),
                decoder_tile_size=args.get("vae_decoder_tile_size", 224)
            )

        # Move to GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipeline = pipeline.to(device=device, dtype=dtype)

        return pipeline

    def process(self, image, **kwargs):
        # Convert ComfyUI image format to PIL
        if isinstance(image, torch.Tensor):
            if image.ndim == 4:
                # Image is [B, H, W, C], convert to [B, C, H, W] for processing
                image = image.permute(0, 3, 1, 2).contiguous()
                # Convert to PIL
                image = F.to_pil_image(image[0].cpu())
        
        # Check if image dimensions are smaller than skip_smaller_than
        skip_smaller_than = kwargs.get("skip_smaller_than", 0)
        if skip_smaller_than > 0:
            width, height = image.size
            if width < skip_smaller_than or height < skip_smaller_than:
                # Return the original image without any processing
                return
        
        # Load pipeline if not already loaded or if paths changed
        pipeline_key = f"{kwargs['pretrained_model_path']}_{kwargs['controlnet_model_path']}_{kwargs['vae_model_path']}"
        
        # Include tiling settings in the pipeline key to ensure we reload when these change
        pipeline_key += f"_tile_vae_{kwargs.get('tile_vae', True)}"
        pipeline_key += f"_encoder_{kwargs.get('vae_encoder_tile_size', 1024)}"
        pipeline_key += f"_decoder_{kwargs.get('vae_decoder_tile_size', 224)}"
        
        if pipeline_key not in self.loaded_models:
            self.loaded_models[pipeline_key] = self.load_pipeline(kwargs)
        pipeline = self.loaded_models[pipeline_key]

        # Process image
        width, height = image.size
        validation_image = image.resize((width * kwargs["upscale"], height * kwargs["upscale"]))
        validation_image = validation_image.resize((validation_image.size[0]//8*8, validation_image.size[1]//8*8))

        inference_time, output_image = pipeline(
            kwargs["t_max"],
            kwargs["t_min"],
            kwargs.get("tile_diffusion", True),
            kwargs.get("tile_diffusion_size", 512),
            kwargs.get("tile_diffusion_stride", 256),
            kwargs["added_prompt"],
            validation_image,
            num_inference_steps=kwargs["num_inference_steps"],
            height=validation_image.size[1],
            width=validation_image.size[0],
            guidance_scale=kwargs["guidance_scale"],
            negative_prompt=kwargs["negative_prompt"],
            start_steps=kwargs["start_steps"],
            start_point=kwargs["start_point"],
            use_vae_encode_condition=kwargs.get("use_vae_encode_condition", True),
        )
        output_image = output_image.images[0]

        # Apply color fixing if specified
        if kwargs["align_method"] != "nofix":
            fix_func = wavelet_color_fix if kwargs["align_method"] == "wavelet" else adain_color_fix
            output_image = fix_func(output_image, validation_image)

        # Convert output back to ComfyUI format
        output_tensor = F.to_tensor(output_image)  # This gives us [C, H, W]
        output_tensor = output_tensor.unsqueeze(0)  # Add batch dim -> [1, C, H, W]
        output_tensor = output_tensor.permute(0, 2, 3, 1)  # Convert to [B, H, W, C] for ComfyUI

        return (output_tensor,)

NODE_CLASS_MAPPINGS = {
    "CCSRUpscale": CCSRNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CCSRUpscale": "CCSR Upscale"
} 
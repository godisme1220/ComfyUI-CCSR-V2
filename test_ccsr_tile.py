import os
import glob
import math
import time
import argparse

import numpy as np
from PIL import Image
import safetensors.torch

import torch
from torchvision import transforms
import torchvision.transforms.functional as F

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    UniPCMultistepScheduler,
    DPMSolverMultistepScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
)

from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from pipelines.pipeline_ccsr import StableDiffusionControlNetPipeline
from myutils.wavelet_color_fix import wavelet_color_fix, adain_color_fix
from models.controlnet import ControlNetModel



def load_pipeline(args, accelerator, enable_xformers_memory_efficient_attention):
    
    scheduler_mapping = {
        'unipcmultistep': UniPCMultistepScheduler,
        'ddpm': DDPMScheduler,
        'dpmmultistep': DPMSolverMultistepScheduler,
    }

    try:
        scheduler_cls = scheduler_mapping[args.sample_method]
    except KeyError:
        raise ValueError(f"Invalid sample_method: {args.sample_method}")

    scheduler = scheduler_cls.from_pretrained(args.pretrained_model_path, subfolder="scheduler")

    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    feature_extractor = CLIPImageProcessor.from_pretrained(os.path.join(args.pretrained_model_path, "feature_extractor"))
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_path, subfolder="controlnet")

    vae_path = args.vae_model_path if args.vae_model_path else args.pretrained_model_path
    vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae")

    # Freeze models
    for model in [vae, text_encoder, unet, controlnet]:
        model.requires_grad_(False)

    # Enable xformers if available
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Ensure it is installed correctly.")

    # Initialize pipeline
    validation_pipeline = StableDiffusionControlNetPipeline(
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

    if args.tile_vae:
        validation_pipeline._init_tiled_vae(
            encoder_tile_size=args.vae_encoder_tile_size,
            decoder_tile_size=args.vae_decoder_tile_size
        )

    # Set weight dtype based on mixed precision
    dtype_mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    weight_dtype = dtype_mapping.get(accelerator.mixed_precision, torch.float32)

    # Move models to accelerator device with appropriate dtype
    for model in [text_encoder, vae, unet, controlnet]:
        model.to(accelerator.device, dtype=weight_dtype)

    return validation_pipeline

def main(args, enable_xformers_memory_efficient_attention=True,):
    
    detailed_output_dir = os.path.join(
        args.output_dir,
        f"sr_{args.baseline_name}_{args.sample_method}_{str(args.num_inference_steps).zfill(3)}steps_{args.start_point}{args.start_steps}_size{args.process_size}_cfg{args.guidance_scale}"
    )

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the output folder creation
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        os.makedirs(detailed_output_dir, exist_ok=True)
        accelerator.init_trackers("Controlnet")

    pipeline = load_pipeline(args, accelerator, enable_xformers_memory_efficient_attention)

    if accelerator.is_main_process:
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator.manual_seed(args.seed)

        image_paths = sorted(glob.glob(os.path.join(args.image_path, "*.*"))) if os.path.isdir(args.image_path) else [args.image_path]

        time_records = []
        for image_path in image_paths:
            validation_image = Image.open(image_path).convert("RGB")
            negative_prompt = args.negative_prompt
            validation_prompt = args.added_prompt 

            ori_width, ori_height = validation_image.size
            resize_flag = False
            rscale = args.upscale
            if ori_width < args.process_size//rscale or ori_height < args.process_size//rscale:
                scale = (args.process_size//rscale)/min(ori_width, ori_height)
                tmp_image = validation_image.resize((round(scale*ori_width), round(scale*ori_height)))
                validation_image = tmp_image
                resize_flag = True


            validation_image = validation_image.resize((validation_image.size[0]*rscale, validation_image.size[1]*rscale))
            validation_image = validation_image.resize((validation_image.size[0]//8*8, validation_image.size[1]//8*8))
            width, height = validation_image.size
            resize_flag = True #
      
            for sample_idx in range(args.sample_times):
                os.makedirs(f'{detailed_output_dir}/sample{str(sample_idx).zfill(2)}/', exist_ok=True)
                
            for sample_idx in range(args.sample_times):

                inference_time, image = pipeline(
                    args.t_max,
                    args.t_min,
                    args.tile_diffusion,
                    args.tile_diffusion_size,
                    args.tile_diffusion_stride,
                    args.added_prompt,
                    validation_image,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    height=height,
                    width=width,
                    guidance_scale=args.guidance_scale,
                    negative_prompt=args.negative_prompt,
                    conditioning_scale=args.conditioning_scale,
                    start_steps=args.start_steps,
                    start_point=args.start_point,
                    use_vae_encode_condition=args.use_vae_encode_condition,
                )
                image = image.images[0]

                print(f"Inference time: {inference_time:.4f} seconds")
                time_records.append(inference_time)

                # Apply color fixing if specified
                if args.align_method != 'nofix':
                    fix_func = wavelet_color_fix if args.align_method == 'wavelet' else adain_color_fix
                    image = fix_func(image, validation_image)
                    
                if resize_flag: 
                    image = image.resize((ori_width*rscale, ori_height*rscale))
                
                image_tensor = torch.clamp(F.to_tensor(image), 0, 1)
                final_image = transforms.ToPILImage()(image_tensor)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                save_path = os.path.join(detailed_output_dir, f"sample{str(sample_idx).zfill(2)}", f"{base_name}.png")
                image.save(save_path)
        
        # Calculate the average inference time, excluding the first few for stabilization
        if len(time_records) > 3:
            average_time = np.mean(time_records[3:])
        else:
            average_time = np.mean(time_records)
        if accelerator.is_main_process:
            print(f"Average inference time: {average_time:.4f} seconds")   
                    

    # Save the run settings to a file
    settings_path = os.path.join(detailed_output_dir, "settings.txt")
    with open(settings_path, 'w') as f:
        f.write("------------------ start ------------------\n")
        for key, value in vars(args).items():
            f.write(f"{key} : {value}\n")
        f.write("------------------- end -------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Diffusion ControlNet Pipeline for Super-Resolution")
    parser.add_argument("--controlnet_model_path", type=str, default="", help="Path to ControlNet model")
    parser.add_argument("--pretrained_model_path", type=str, default="", help="Path to pretrained model")
    parser.add_argument("--vae_model_path", type=str, default="", help="Path to VAE model")
    parser.add_argument("--added_prompt", type=str, default="clean, high-resolution, 8k", help="Additional prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="blurry, dotted, noise, raster lines, unclear, lowres, over-smoothed", help="Negative prompt to avoid certain features")
    parser.add_argument("--image_path", type=str, default="", help="Path to input image or directory")
    parser.add_argument("--output_dir", type=str, default="", help="Directory to save outputs")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="fp16", help="Mixed precision mode")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale for generation")
    parser.add_argument("--conditioning_scale", type=float, default=1.0, help="Conditioning scale")
    parser.add_argument("--num_inference_steps", type=int, default=1, help="Number of inference steps(not the final inference time)")
    # final_inference_time = num_inference_steps * (t_max - t_min) + 1
    parser.add_argument("--t_max", type=float, default=0.6666, help="Maximum timestep")
    parser.add_argument("--t_min", type=float, default=0.0, help="Minimum timestep")
    parser.add_argument("--process_size", type=int, default=512, help="Processing size of the image")
    parser.add_argument("--upscale", type=int, default=1, help="Upscaling factor")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--sample_times", type=int, default=5, help="Number of samples to generate per image")
    parser.add_argument("--sample_method", type=str, choices=['unipcmultistep', 'ddpm', 'dpmmultistep'], default='ddpm', help="Sampling method")
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain', help="Alignment method for color fixing")
    parser.add_argument("--start_steps", type=int, default=999, help="Starting steps")
    parser.add_argument("--start_point", type=str, choices=['lr', 'noise'], default='lr', help="Starting point for generation")
    parser.add_argument("--baseline_name", type=str, default='ccsr-v2', help="Baseline name for output naming")
    parser.add_argument("--use_vae_encode_condition", action='store_true', help="Use VAE encoding LQ condition")
    
    # Tiling settings for high-resolution SR
    parser.add_argument("--tile_diffusion", action="store_true", help="Optionally! Enable tile-based diffusion")
    parser.add_argument("--tile_diffusion_size", type=int, default=512, help="Tile size for diffusion")
    parser.add_argument("--tile_diffusion_stride", type=int, default=256, help="Stride size for diffusion tiles")
    parser.add_argument("--tile_vae", action="store_true", help="Optionally! Enable tiling for VAE")
    parser.add_argument("--vae_decoder_tile_size", type=int, default=224, help="Tile size for VAE decoder")
    parser.add_argument("--vae_encoder_tile_size", type=int, default=1024, help="Tile size for VAE encoder")

    args = parser.parse_args()
    main(args)
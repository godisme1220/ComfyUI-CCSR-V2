
python test_ccsr_tile.py \
--pretrained_model_path preset/models/stable-diffusion-2-1-base \
--controlnet_model_path preset/models \
--vae_model_path preset/models \
--baseline_name ccsr-v2 \
--image_path preset/test_datasets \
--output_dir experiments/test \
--sample_method ddpm \
--num_inference_steps 1 \
--t_min 0.0 \
--start_point lr \
--start_steps 999 \
--process_size 512 \
--guidance_scale 1.0 \
--sample_times 1 \
--use_vae_encode_condition \
--upscale 4
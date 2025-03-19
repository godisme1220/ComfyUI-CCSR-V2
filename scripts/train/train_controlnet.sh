
CUDA_VISIBLE_DEVICES="0,1,2,3," accelerate launch train_controlnet.py \
--pretrained_model_name_or_path="preset/models/stable-diffusion-2-1-base" \
--controlnet_model_name_or_path='' \
 --enable_xformers_memory_efficient_attention \
 --output_dir="./experiments/pretrained_controlnet" \
 --mixed_precision="fp16" \
 --resolution=512 \
 --learning_rate=5e-5 \
 --train_batch_size=4 \
 --gradient_accumulation_steps=6 \
 --dataloader_num_workers=0 \
 --checkpointing_steps=5000 \
 --max_train_steps=40000 \
 --dataset_root_folders 'preset/gt_path.txt'
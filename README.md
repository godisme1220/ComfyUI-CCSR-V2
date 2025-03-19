I found NV RTX-Remix project have implemented CCSR-V2 comfyui wrapper, but seems have some problem, so I fix it with my colleague, then forked from kijai/CCSR project.

the main code is from https://huggingface.co/NightRaven109/PBRFusion project zip file.

Installation:

git clone this repo to the ComfyUI/custom_nodes, then create a folder "ccsr" under ComfyUI/models.

then download the corresponding files from CCSRV2 repo indicated. https://huggingface.co/stabilityai/stable-diffusion-2-1-base/tree/main

the models/ccsr folder structure should be as below:

├─Controlnet
│      config.json
│      diffusion_pytorch_model.safetensors
│
├─stable-diffusion-2-1-base
│  │  model_index.json
│  │
│  ├─feature_extractor
│  │      preprocessor_config.json
│  │
│  ├─scheduler
│  │      scheduler_config.json
│  │
│  ├─text_encoder
│  │      config.json
│  │      model.safetensors
│  │
│  ├─tokenizer
│  │      merges.txt
│  │      special_tokens_map.json
│  │      tokenizer_config.json
│  │      vocab.json
│  │
│  └─unet
│          config.json
│          diffusion_pytorch_model.safetensors
│
└─vae
        config.json
        diffusion_pytorch_model.safetensors


After that, restart comfy, should see a CCSRV2 Upscale node.

![image](https://github.com/user-attachments/assets/0b37ccff-8786-4bf9-b52b-604247f4a04d)


Warning! this is very dirty code, and I'm a newbie to github, so maybe some day the kijai will update the original CCSR repo to CCSRV2. :D

# ComfyUI- CCSR upscaler node

This is a simple wrapper node for https://github.com/csslc/CCSR

NOT a proper ComfyUI implementation, so not very efficient and there might be memory issues, tested on 4090 and 4x upscale tiled worked well.

Upscale the input first with another node for the desired end scale.

The model (https://drive.google.com/drive/folders/1jM1mxDryPk9CTuFTvYcraP2XIVzbPiw_?usp=drive_link) goes to ComfyUI/models/checkpoints
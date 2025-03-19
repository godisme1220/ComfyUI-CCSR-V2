import glob
import os
from PIL import Image
import random
import numpy as np

from torch import nn
from torchvision import transforms
from torch.utils import data as data
import torch.nn.functional as F

from .realesrgan import RealESRGAN_degradation

class PairedCaptionDataset(data.Dataset):
    def __init__(
            self,
            root_folders=None,
            tokenizer=None,
            gt_ratio=0, # let lr is gt
    ):
        super(PairedCaptionDataset, self).__init__()

        self.gt_ratio = gt_ratio
        with open(root_folders, 'r') as f:
            self.gt_list = [line.strip() for line in f.readlines()]

        self.img_preproc = transforms.Compose([
            transforms.RandomCrop((512, 512)),
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            ])

        self.degradation = RealESRGAN_degradation('dataloaders/params_ccsr.yml', device='cuda')
        self.tokenizer = tokenizer

    
    def tokenize_caption(self, caption=""):
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return inputs.input_ids

    def __getitem__(self, index):

        gt_path = self.gt_list[index]
        gt_img = Image.open(gt_path).convert('RGB')
        gt_img = self.img_preproc(gt_img)

        gt_img, img_t = self.degradation.degrade_process(np.asarray(gt_img)/255., resize_bak=True)

        if random.random() < self.gt_ratio:
            lq_img = gt_img
        else:
            lq_img = img_t

        # no caption used
        lq_caption = ''

        example = dict()
        example["conditioning_pixel_values"] = lq_img.squeeze(0) # [0, 1]
        example["pixel_values"] = gt_img.squeeze(0) * 2.0 - 1.0 # [-1, 1]
        example["input_caption"] = self.tokenize_caption(caption=lq_caption).squeeze(0)

        lq_img = lq_img.squeeze()

        return example

    def __len__(self):
        return len(self.gt_list)
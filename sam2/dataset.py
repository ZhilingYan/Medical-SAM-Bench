import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import random
      
    
class GeneralData(Dataset):
    def __init__(
        self, 
        args, 
        data_path , 
        transform = None, 
        transform_msk = None, 
        mode = 'Training',
        prompt = 'click', 
        train_file_dir="train.txt",
        val_file_dir="val.txt",
    ):
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.mask_size = args.out_size

        self.transform = transform
        self.transform_msk = transform_msk
        
        self.sample_list = []
        self.bbox_shift = 20

        if self.mode == "Training":
            with open(train_file_dir, "r") as f1:
                # Strip each line, and keep only the non-empty ones
                self.sample_list = [line.strip() for line in f1 if line.strip()]
            self.sample_list.sort()

        elif self.mode == "Test":
            with open(val_file_dir, "r") as f:
                self.sample_list = [line.strip() for line in f if line.strip()]
            self.sample_list.sort()

        print("total {}  {} samples".format(len(self.sample_list), self.mode))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
    
        case = self.sample_list[index]

        # raw image and mask paths
        img_path = os.path.join(self.data_path, case[2:])
        mask_path = img_path.replace("/image/", "/mask/")

        # raw image and mask loading
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # apply transform
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            mask = torch.as_tensor((self.transform(mask) >= 0.5).float(), dtype=torch.float32)
            torch.set_rng_state(state)

        mask_tmp = mask.mean(axis=0)
        mask_tmp = (mask_tmp >= 0.5).float() 
        final_mask = F.interpolate(
            mask_tmp.unsqueeze(0).unsqueeze(1),
            size=(self.mask_size, self.mask_size), 
            mode='bilinear', 
            align_corners=False
        ).mean(dim=0)
        final_mask = (final_mask >= 0.5).float()

        mask_2d = final_mask  

        if mask_2d.dim() == 3:
            mask_2d = mask_2d[0]  # remove the channel dim

        H, W = mask_2d.shape[-2], mask_2d.shape[-1]  # if 2D

        non_zero_indices = torch.nonzero(mask_2d)

        if non_zero_indices.numel() > 0:
            y_min, x_min = torch.min(non_zero_indices, dim=0)[0]
            y_max, x_max = torch.max(non_zero_indices, dim=0)[0]
            
            x_min_new = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max_new = min(W, x_max + random.randint(0, self.bbox_shift))
            y_min_new = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max_new = min(H, y_max + random.randint(0, self.bbox_shift))

            boxes = torch.tensor([x_min_new, y_min_new, x_max_new, y_max_new], dtype=torch.float32)
        else:
            # fallback if mask is empty
            boxes = torch.tensor([0, 0, W, H], dtype=torch.float32)

        return {
            'image': img, 
            'mask': final_mask, 
            'case': case,
            'boxes': boxes,
        }

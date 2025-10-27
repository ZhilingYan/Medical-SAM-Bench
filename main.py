import os
import argparse
import pickle
from tqdm import tqdm
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import sam2.function as function
from sam2.dataset import *
from sam2.util import *
from sam2.build_sam import build_sam2

import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from skimage import transform
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import random
import shutil
import glob
from PIL import Image

def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='sam2', help='net type')
    parser.add_argument('-sam_ckpt', type=str, default=None , help='sam checkpoint address')
    parser.add_argument('-sam_config', type=str, default=None , help='sam checkpoint address')
    parser.add_argument('-exp_name', default='samba_train_test', type=str, help='experiment name')
    parser.add_argument('-image_size', type=int, default=1024, help='image_size')
    parser.add_argument('-data_path', type=str, default='./data/btcv', help='The path of segmentation data')
    parser.add_argument('-train_file_dir', type=str, default=None, help='train path file')
    parser.add_argument('-val_file_dir', type=str, default=None, help='val path file')
    parser.add_argument('-epoch', type=int, default=100, help='epoch number')
    parser.add_argument('-local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser.add_argument('-seed', type=int, default=42, help='random seed')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-out_size', type=int, default=1024, help='output_size')
    parser.add_argument('-b', type=int, default=4, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('-weight_decay', type=float, default=0.01, help='weight decay (default: 0.01)')
    parser.add_argument('-num_workers', type=int, default=0)
    parser.add_argument('-multimask_output', type=int, default=1 , help='the number of masks output for multi-class segmentation')
    parser.add_argument('-memory_bank_size', type=int, default=640, help='sam 2d memory bank size')
    parser.add_argument('-update_memory_bank_during_val', type=bool, default=True, help='whether to update memory bank during validation')
    parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
    parser.add_argument('-resume', type=str, default="", help="Resuming training from checkpoint")

    opt = parser.parse_args()
    return opt


def main():

    args = parse_args()
    device = torch.device('cuda', args.gpu_device)

    if args.net == 'samed2' or args.net == 'medsam2' or args.net == 'sam2':
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        net, _  = build_sam2(args.sam_config, args.sam_ckpt, device="cuda")
    elif args.net == 'medsam' or args.net == 'sam':
        os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
        os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
        os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
        os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
        os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

        sam_model = sam_model_registry["vit_b"](checkpoint=args.sam_ckpt)
        net = function.MedSAM(
            image_encoder=sam_model.image_encoder,
            mask_decoder=sam_model.mask_decoder,
            prompt_encoder=sam_model.prompt_encoder,
        ).to(device)

    '''segmentation data'''

    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    test_dataset = GeneralData(args, args.data_path, transform = transform_test, mode = 'Test', train_file_dir = args.train_file_dir, val_file_dir = args.val_file_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.b, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    
    net.eval()
    n_val = len(test_loader)
    
    # 加载memory bank
    filename = f"memory_bank_list_{args.memory_bank_size}.pkl"
    with open(filename, "rb") as f:
        memory_bank_list = pickle.load(f)
    print(f"Loaded memory_bank_list of length {len(memory_bank_list)} from {filename}")
    
    total_loss = 0
    total_eiou = 0
    total_dice = 0
    
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for step, pack in enumerate(test_loader):
            if args.net == 'samed2':
                loss, eiou, edice, memory_bank_list = function.validation_step(args, pack, net, memory_bank_list, device)
            elif args.net == 'medsam2':
                loss, eiou, edice, memory_bank_list = function.validation_step_medsam2(args, step, pack, net, device)
            elif args.net == 'sam2':
                loss, eiou, edice, memory_bank_list = function.validation_step_sam2(args, pack, net, memory_bank_list, device)
            elif args.net == 'medsam':
                loss, eiou, edice = function.validation_step_sam(args, step, pack, net, device)
            elif args.net == 'sam':
                loss, eiou, edice = function.validation_step_sam(args, step, pack, net, device)
            total_loss += loss
            total_eiou += eiou
            total_dice += edice
            pbar.update()
    
    avg_loss = total_loss / n_val
    avg_eiou = total_eiou / n_val
    avg_dice = total_dice / n_val
    print(f'Total score: {avg_loss}, IOU: {avg_eiou}, DICE: {avg_dice}.')

if __name__ == '__main__':
    main()

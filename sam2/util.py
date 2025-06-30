import math
import os
import pathlib
import random
import warnings
from typing import BinaryIO, List, Optional, Text, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Function
import monai.losses

# Global variables that will be initialized when needed
dicece_loss = monai.losses.DiceCELoss(
    sigmoid=True,
    squared_pred=True,
    reduction='mean'
)

def set_seed(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch (CPU/CUDA).
    Also configures CUDA algorithms for (attempted) reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # If you use GPUs/accelerators
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic (may slow down training).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    #print(f"[INFO] Global seed set to {seed}")

def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Binary focal loss, similar to RetinaNet. 
    Args:
        logits:  [B, 1, H, W] (or [B, H, W]) raw, un-sigmoided predictions.
        targets: [B, 1, H, W] (or [B, H, W]) in {0,1}.
        alpha:   Weight for positive/negative examples.
        gamma:   Focusing parameter.
        reduction: 'mean' or 'sum' or 'none'
    Returns:
        Focal loss
    """
    bce = F.binary_cross_entropy_with_logits(
        logits, targets.float(), reduction='none'
    )
    probas = torch.sigmoid(logits)
    # p_t: probability assigned to the *true* class
    p_t = probas * targets + (1 - probas) * (1 - targets)
    focal_factor = (1.0 - p_t) ** gamma

    # alpha weighting
    if alpha >= 0:
        alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    else:
        alpha_factor = 1.0

    loss = alpha_factor * focal_factor * bce

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
        
def iou_regression_loss(
    logits: torch.Tensor,  # raw segmentation logits
    iou_preds: torch.Tensor,  # [B, 1] or similar shape
    targets: torch.Tensor,    # GT masks
    use_l1_loss: bool = True
) -> torch.Tensor:
    """
    Compares model's predicted IoU (iou_preds) to the actual IoU 
    computed from segmentation logits vs. ground-truth targets.
    """
    # 1) Binarize predictions
    pred_mask = (torch.sigmoid(logits) > 0.5)
    gt_mask = (targets > 0.5)

    # 2) Compute actual IoU
    intersection = (pred_mask & gt_mask).sum(dim=[1,2,3]).float()
    union = (pred_mask | gt_mask).sum(dim=[1,2,3]).float()
    actual_iou = intersection / (union + 1e-7)  # shape: [B]

    # 3) Compare predicted IoU to actual IoU
    # If iou_preds has shape [B, 1], flatten to match [B]
    if iou_preds.dim() > 1:
        iou_preds = iou_preds.view(-1)

    if use_l1_loss:
        loss = F.l1_loss(iou_preds, actual_iou, reduction='mean')
    else:
        loss = F.mse_loss(iou_preds, actual_iou, reduction='mean')

    return loss

def combined_seg_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    iou_preds: torch.Tensor,
    dice_weight: float = 1.0,
    focal_weight: float = 1.0,
    iou_weight: float = 1.0,
    alpha_focal: float = 0.25,
    gamma_focal: float = 2.0,
    use_l1_iou: bool = True
):
    """
    Combine DiceCE + Focal + IoU losses into a single scalar
    """
    # 1) DiceCE from MONAI
    dicece_l = dicece_loss(logits, masks)

    # 2) Focal
    focal_l = focal_loss_with_logits(
        logits, masks,
        alpha=alpha_focal,
        gamma=gamma_focal
    )

    # 3) IoU regression
    iou_l = iou_regression_loss(
        logits, iou_preds, masks,
        use_l1_loss=use_l1_iou
    )

    # Weighted sum
    total_loss = (
          dice_weight * dicece_l
        + focal_weight * focal_l
        + iou_weight * iou_l
    )
    return total_loss


def get_param_groups(model, missing_keys, base_lr):
    new_params = []
    pretrained_params = []

    for name, param in model.named_parameters():
        if any(missing_key.startswith(name) for missing_key in missing_keys):
            new_params.append(param)
        else:
            pretrained_params.append(param)

    return [
        {"params": pretrained_params, "lr": base_lr},     
        {"params": new_params, "lr": base_lr * 10},       
    ]

@torch.no_grad()
def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    **kwargs
) -> torch.Tensor:
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clamp(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[Text, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs
) -> None:
    """
    Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)
    


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))



def iou(outputs: np.array, labels: np.array):
    
    SMOOTH = 1e-5
    intersection = (outputs & labels).sum()
    union = (outputs | labels).sum()

    iou = (intersection + SMOOTH) / (union + SMOOTH)
    dice = (2 * iou) / (iou+1)

    return iou, dice


def eval_seg(pred,true_mask_p,threshold):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    b, c, h, w = pred.size()
    eiou, edice = 0,0

    gt_vmask_p = (true_mask_p > threshold[0]).float()
    vpred = (pred > threshold[0]).float()
    vpred_cpu = vpred.cpu()
    disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')

    disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')

    '''iou for numpy'''
    eiou_tmp, edice_tmp = iou(disc_pred,disc_mask)
    eiou += eiou_tmp
    edice += edice_tmp

    '''dice for torch'''
    #edice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
        
    return eiou / len(threshold), edice / len(threshold)


def random_click(mask, point_label = 1):
    max_label = max(set(mask.flatten()))
    if round(max_label) == 0:
        point_label = round(max_label)
    indices = np.argwhere(mask == max_label) 
    return point_label, indices[np.random.randint(len(indices))]

def agree_click(mask, label = 1):
    # max agreement position
    indices = np.argwhere(mask == label) 
    if len(indices) == 0:
        label = 1 - label
        indices = np.argwhere(mask == label) 
    return label, indices[np.random.randint(len(indices))]


def random_box(multi_rater):
    max_value = torch.max(multi_rater[:,0,:,:], dim=0)[0]
    max_value_position = torch.nonzero(max_value)

    x_coords = max_value_position[:, 0]
    y_coords = max_value_position[:, 1]


    x_min = int(torch.min(x_coords))
    x_max = int(torch.max(x_coords))
    y_min = int(torch.min(y_coords))
    y_max = int(torch.max(y_coords))


    x_min = random.choice(np.arange(x_min-10,x_min+11))
    x_max = random.choice(np.arange(x_max-10,x_max+11))
    y_min = random.choice(np.arange(y_min-10,y_min+11))
    y_max = random.choice(np.arange(y_max-10,y_max+11))

    return x_min, x_max, y_min, y_max



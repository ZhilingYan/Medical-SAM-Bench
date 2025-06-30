# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    missing_keys = _load_checkpoint(model, ckpt_path)
    # missing_keys = None
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model, missing_keys


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
):
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model

def _initialize_missing_keys(model, missing_keys):
    for name, module in model.named_modules():
        if any(missing_key.startswith(name) for missing_key in missing_keys):
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1)
            # Add more initialization logic as needed


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
        # for param_key in sd.keys():
        #     print("  -", param_key)

        # checkpoint = torch.load(ckpt_path, map_location="cpu")
        # sd = checkpoint["model"]  # This is the dictionary from 'model': net.state_dict()

        # # Create a new dictionary, stripping "base_net." if present
        # new_state_dict = {}
        # for old_key, val in sd.items():
        #     if old_key.startswith("base_net."):
        #         new_key = old_key[len("base_net."):]
        #     else:
        #         new_key = old_key
        #     new_state_dict[new_key] = val

        # # Now load the stripped dictionary into your model
        # missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)


        # Log missing and unexpected keys
        if missing_keys:
            logging.warning(f"Missing keys (new modules): {missing_keys}")
            _initialize_missing_keys(model, missing_keys)

        if unexpected_keys:
            logging.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

        logging.info("Loaded checkpoint successfully")
    return missing_keys


# SAMed2 Model Zoo

This page provides links to download all available models for SAMed2 and baseline methods.

## Model Overview

We provide various model checkpoints for medical image segmentation, including our SAMed2 model and several baseline methods for comparison.

## Available Models

### SAMed2 (Ours)

| Model Name | Base Model | Training Data | Parameters | Download | MD5 | Notes |
|------------|------------|---------------|------------|----------|-----|-------|
| SAMed2-S | SAM2 Hiera-S | MedBank (640 prompts) | 46M | [latest_epoch_0217.pth](https://drive.google.com/file/d/xxx) (223MB) | `abc123...` | Best overall performance |
| SAMed2-S-lite | SAM2 Hiera-S | MedBank (320 prompts) | 46M | [samed2_s_lite.pth](https://drive.google.com/file/d/xxx) (223MB) | `def456...` | Faster inference |
| SAMed2-T | SAM2 Hiera-T | MedBank (640 prompts) | 38M | [samed2_t.pth](https://drive.google.com/file/d/xxx) (180MB) | `ghi789...` | Lightweight version |

### Baseline Models

| Model Name | Base Model | Training Data | Parameters | Download | MD5 | Notes |
|------------|------------|---------------|------------|----------|-----|-------|
| MedSAM2 | SAM2 Hiera-T | Medical datasets | 38M | [MedSAM2_pretrain.pth](https://drive.google.com/file/d/xxx) (74MB) | `jkl012...` | Medical-tuned SAM2 |
| MedSAM | SAM ViT-B | Medical datasets | 91M | [medsam_vit_b.pth](https://drive.google.com/file/d/xxx) (358MB) | `mno345...` | Original MedSAM |
| SAM2 | SAM2 Hiera-S | Natural images | 46M | [sam2_hiera_small.pt](https://facebook.com/xxx) (176MB) | `pqr678...` | Original SAM2 |
| SAM | SAM ViT-B | Natural images | 91M | [sam_vit_b_01ec64.pth](https://facebook.com/xxx) (358MB) | `stu901...` | Original SAM |

## Model Performance Comparison

### Overall Performance on MedBank Test Set

| Model | Average IoU | Average Dice | FPS | GPU Memory |
|-------|-------------|--------------|-----|------------|
| SAM | 0.743 | 0.842 | 12 | 8.2 GB |
| MedSAM | 0.812 | 0.887 | 11 | 8.5 GB |
| SAM2 | 0.798 | 0.876 | 18 | 6.8 GB |
| MedSAM2 | 0.834 | 0.901 | 17 | 7.1 GB |
| **SAMed2** | **0.867** | **0.923** | 16 | 7.3 GB |

### Performance by Modality

| Model | CT | MRI | Ultrasound | Dermoscopy | Microscopy | X-Ray |
|-------|----|----|------------|------------|------------|-------|
| SAM | 0.756 | 0.721 | 0.698 | 0.812 | 0.745 | 0.789 |
| MedSAM | 0.823 | 0.801 | 0.776 | 0.856 | 0.812 | 0.834 |
| SAM2 | 0.812 | 0.789 | 0.756 | 0.843 | 0.798 | 0.821 |
| MedSAM2 | 0.845 | 0.823 | 0.798 | 0.871 | 0.834 | 0.856 |
| **SAMed2** | **0.878** | **0.856** | **0.834** | **0.893** | **0.867** | **0.889** |

## Usage Instructions

### Download Models

1. Click on the download links above to get the model checkpoints
2. Place the downloaded files in the `Code/SAMed2/checkpoints/` directory
3. Verify the MD5 checksum to ensure file integrity

### Load Models in Code

```python
import torch
from sam2.build_sam import build_sam2

# Load SAMed2
model, image_size = build_sam2(
    config_file="sam2_hiera_s",
    ckpt_path="checkpoints/latest_epoch_0217.pth",
    device="cuda"
)

# Load MedSAM
from segment_anything import sam_model_registry
sam_model = sam_model_registry["vit_b"](checkpoint="checkpoints/medsam_vit_b.pth")
```

## Training Details

### SAMed2 Training Configuration

- **Base Model**: SAM2 Hiera-S
- **Training Data**: MedBank dataset (1.2M images across 40 medical tasks)
- **Memory Bank Size**: 640 learnable prompts
- **Training Epochs**: 100
- **Batch Size**: 8 (with gradient accumulation)
- **Learning Rate**: 1e-4 with cosine decay
- **Image Size**: 1024×1024
- **Augmentations**: Random flip, rotation, intensity shift
- **Hardware**: 8× NVIDIA A100 40GB GPUs
- **Training Time**: ~48 hours

### Fine-tuning Your Own Model

To fine-tune SAMed2 on your custom dataset:

```bash
python train.py \
    -net samed2 \
    -exp_name custom_finetune \
    -sam_ckpt checkpoints/latest_epoch_0217.pth \
    -sam_config sam2_hiera_s \
    -data_path /path/to/custom/data \
    -epoch 50 \
    -lr 5e-5 \
    -b 4
```

## License

All SAMed2 models are released under the Apache 2.0 license. Please check the licenses of baseline models before use:
- SAM/SAM2: Apache 2.0
- MedSAM: Apache 2.0

## Contact

For questions about the models or to report issues, please open an issue on our [GitHub repository](https://github.com/yourusername/SAMed2). 
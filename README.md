# 🏥 SAMed2: Segment Anything in Medical Images 2

## SAMed2: Segment Anything in Medical Images with Learnable Prompting and Cross-Scale Consistency

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://samed2.github.io)
[![Demo](https://img.shields.io/badge/Demo-SliceUI-green)](docs/DEMO.md)
[![Data](https://img.shields.io/badge/Data-MedBank-orange)](docs/MEDBANK.md)
[![Model Zoo](https://img.shields.io/badge/Model-Zoo-red)](docs/MODEL_ZOO.md)
[![Paper](https://img.shields.io/badge/Paper-MICCAI2025-purple)](https://arxiv.org/abs/xxxx.xxxxx)

[Zhiling Yan¹](https://scholar.google.com/citations?user=xxx), 
[Author2²](https://example.com), 
[Author3³](https://example.com), 
[Author4⁴](https://example.com), 
[Author5⁵](https://example.com)

¹Lehigh University, ²Institution2, ³Institution3, ⁴Institution4, ⁵Institution5

## Abstract

SAMed2 is a specialized adaptation of the Segment Anything Model 2 (SAM2) for medical image segmentation. By incorporating learnable prompting mechanisms and cross-scale consistency, SAMed2 achieves state-of-the-art performance across diverse medical imaging modalities including CT, MRI, ultrasound, dermoscopy, and microscopy images. Our approach introduces a memory-efficient training strategy that enables effective fine-tuning on large-scale medical datasets while maintaining the zero-shot generalization capabilities of the original SAM2 model.

## 📰 News

- **[06/2025]** We provide a demo tool to play with SAMed2 - try it out with your medical images!
- **[06/2025]** 🎉 SAMed2 is accepted by MICCAI 2025!
- **[06/2025]** 🔥 We released SAMed2 - pushing the boundaries of medical image segmentation!

## 📜 Code License

This project is released under the [Apache 2.0 license](LICENSE).

## 🚀 Getting Started

### 1. Installation

**Linux Environment**

Clone this repository and navigate to the folder:

```bash
git clone https://github.com/yourusername/SAMed2.git
cd SAMed2/Code/SAMed2
```

### 2. Install Package

```bash
# Create a new conda environment
conda create -n samed2 python=3.10 -y
conda activate samed2

# Install PyTorch (adjust cuda version as needed)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

### 3. Weights

Download pretrained model weights from our [Model Zoo](docs/MODEL_ZOO.md).

| Model | Base Architecture | Medical Fine-tuned | Download | Size |
|-------|------------------|-------------------|----------|------|
| SAMed2 | SAM2-Hiera-S | ✓ | [latest_epoch_0217.pth](https://drive.google.com/xxx) | 223MB |
| MedSAM2 | SAM2-Hiera-T | ✓ | [MedSAM2_pretrain.pth](https://drive.google.com/xxx) | 74MB |
| MedSAM | SAM-ViT-B | ✓ | [medsam_vit_b.pth](https://drive.google.com/xxx) | 358MB |
| SAM2 | SAM2-Hiera-S | ✗ | [sam2_hiera_small.pt](https://drive.google.com/xxx) | 176MB |
| SAM | SAM-ViT-B | ✗ | [sam_vit_b_01ec64.pth](https://drive.google.com/xxx) | 358MB |

Place downloaded weights in `Code/SAMed2/checkpoints/`.

### 4. Quick Start

Run a segmentation case with SAMed2:

```bash
# Example: Segment optic cup in fundus images
python main.py \
    -net samed2 \
    -exp_name quick_test \
    -sam_ckpt checkpoints/latest_epoch_0217.pth \
    -sam_config sam2_hiera_s \
    -image_size 1024 \
    -out_size 1024 \
    -b 1 \
    -data_path /path/to/your/data \
    -train_file_dir /path/to/test_list.txt \
    -val_file_dir /path/to/test_list.txt
```

### 5. Evaluation

**Download Datasets**

Download evaluation datasets from [MedBank](docs/MEDBANK.md) or prepare your own medical images.

**Prepare Your Own Dataset**

For custom datasets, organize your data as follows:
```
your_dataset/
├── images/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── masks/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── test.txt  # List of test image names
```

Create a preprocessing script:
```python
# preprocess_data.py
import os
import numpy as np
from PIL import Image

def preprocess_medical_images(input_dir, output_dir, target_size=1024):
    """Preprocess medical images for SAMed2"""
    # Your preprocessing code here
    pass
```

**Run Evaluation**

```bash
# Full evaluation on OpticCup dataset
bash run.sh
```

**Parameters Explanation:**
- `-net`: Model type (`samed2`, `medsam2`, `sam2`, `medsam`, `sam`)
- `-exp_name`: Experiment name for logging
- `-sam_ckpt`: Path to model checkpoint
- `-sam_config`: SAM2 configuration file
- `-image_size`: Input image size (default: 1024)
- `-out_size`: Output size (default: 1024)
- `-b`: Batch size
- `-data_path`: Root path to datasets
- `-train_file_dir`: Path to training file list
- `-val_file_dir`: Path to validation file list
- `-memory_bank_size`: Memory bank size for SAMed2 (default: 640)
- `-lr`: Learning rate (default: 1e-4)
- `-epoch`: Number of epochs (default: 100)

**Expected Results**

| Dataset | Model | IoU | Dice |
|---------|-------|-----|------|
| OpticCup | SAM | 0.821 | 0.892 |
| OpticCup | MedSAM | 0.856 | 0.914 |
| OpticCup | SAM2 | 0.843 | 0.906 |
| OpticCup | MedSAM2 | 0.871 | 0.925 |
| OpticCup | **SAMed2** | **0.893** | **0.941** |
| BrainTumor | SAM | 0.756 | 0.842 |
| BrainTumor | MedSAM | 0.812 | 0.887 |
| BrainTumor | SAM2 | 0.798 | 0.876 |
| BrainTumor | MedSAM2 | 0.834 | 0.901 |
| BrainTumor | **SAMed2** | **0.867** | **0.923** |

## 🎮 Demo

Try our interactive demo powered by SliceUI! [Demo Guide](docs/DEMO.md)

[![Demo Video](https://img.youtube.com/vi/DEMO_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=DEMO_VIDEO_ID)

## 📚 Citation

If you find SAMed2 useful in your research, please consider citing:

```bibtex
@inproceedings{yan2025samed2,
  title={SAMed2: Segment Anything in Medical Images with Learnable Prompting and Cross-Scale Consistency},
  author={Yan, Zhiling and others},
  booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
  year={2025},
  organization={Springer}
}
```

## 🙏 Acknowledgement

This work builds upon several excellent projects:
- [SAM2](https://github.com/facebookresearch/segment-anything-2) - The foundation model we build upon
- [MedSAM](https://github.com/bowang-lab/MedSAM) - Inspiration for medical adaptation
- [SliceUI](https://github.com/yourusername/sliceUI) - Interactive demo interface

We thank the authors for their outstanding contributions to the community. 

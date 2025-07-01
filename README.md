# 🏥 SAMed-2 & Medical SAM Benchmark

## SAMed-2: Selective Memory Enhanced Medical Segment Anything Model

[![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://zhilingyan.github.io/Medical-SAM-Bench/)
[![Demo](https://img.shields.io/badge/Demo-SliceUI-green)](docs/DEMO.md)
[![Data](https://img.shields.io/badge/Data-MedBank-orange)](docs/MEDBANK.md)
[![Model Zoo](https://img.shields.io/badge/Model-Zoo-red)](docs/MODEL_ZOO.md)
[![Paper](https://img.shields.io/badge/Paper-Arxiv-purple)](https://arxiv.org/abs/xxxx.xxxxx)

**🔥 This is the official repository of SAMed-2 and a comprehensive benchmark for Medical SAM models!**

We provide unified evaluation and comparison of multiple Medical SAM variants including:
- **SAMed2** (Ours) - [Arxiv]()
- **MedSAM2** - [Arxiv]()
- **MedSAM** - [Arxiv]()
- **SAM2** - [Arxiv]()
- **SAM** - [Arxiv]()

## Abstract

SAMed-2 is a
new foundation model for medical image segmentation built upon the
SAM-2 architecture. Specifically, we introduce a temporal adapter into
the image encoder to capture image correlations and a confidence-driven
memory mechanism to store high-certainty features for later retrieval.
This memory-based strategy counters the pervasive noise in large-scale
medical datasets and mitigates catastrophic forgetting when encountering
new tasks or modalities.

This repository provides:
- 🔧 Unified implementation and evaluation framework for SAMed-2.
- 🏆 Fair comparison across different medical SAM models.
- 🚀 Easy-to-use interface for testing your own medical images.

## 📰 News

- **[07/2025]** We provide a demo tool to play with SAMed-2 - try it out with your medical images!
- **[06/2025]** 🎉 SAMed2 is accepted by MICCAI 2025!
- **[06/2025]** 🔥 We released SAMed2!

## 📜 Code License

This project is released under the [Apache 2.0 license](LICENSE).

## 🚀 Getting Started

### 1. Installation

**Linux Environment**

Clone this repository and navigate to the folder:

```bash
git clone https://github.com/ZhilingYan/Medical-SAM-Bench.git
cd Medical-SAM-Bench
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

| Model | Base Architecture | Medical Fine-tuned | Download |
|-------|------------------|-------------------|----------|
| **SAMed2** ⭐ | SAM2-Hiera-S | ✓ | [latest_epoch_0217.pth](https://drive.google.com/file/d/1JVmZnpWip7AIi8o9J1heog_Kl5uHGHcP/view?usp=sharing) |
| MedSAM2 | SAM2-Hiera-T | ✓ | [MedSAM2_pretrain.pth](https://drive.google.com/file/d/1XQmJ13-SahH-57eH1-UabU1OpGpoTZWT/view?usp=sharing) |
| MedSAM | SAM-ViT-B | ✓ | [medsam_vit_b.pth](https://drive.google.com/file/d/1V81_3KuJ-7q1gzLYcQFPCTAAymfkxh6Y/view?usp=sharing) |
| SAM2 | SAM2-Hiera-S | ✗ | [sam2_hiera_small.pt](https://drive.google.com/file/d/1bNtsqOCRnzDOb_10EN9bAACLPew32yus/view?usp=sharing) |
| SAM | SAM-ViT-B | ✗ | [sam_vit_b_01ec64.pth](https://drive.google.com/file/d/1LgRKsBkCYOeQQRWyF1RnXZgwe-_xfR0_/view?usp=sharing) |

Place downloaded weights and put in `./checkpoints/`.

Memory bank list is saved during pre training of SAMed-2. It could be downloaded [HERE](https://drive.google.com/file/d/1nrq9GRhlCUG7ha-RuuQktODyfK1UKbwL/view?usp=sharing). Put it directly in the main folder of this repo.

### 4. Quick Start

**🚀 Segment a Medical Image in Just a Few Lines:**

```python
from predict import MedicalSegmenter

# Initialize segmenter
segmenter = MedicalSegmenter(model_type='samed2', 
                             checkpoint_path='checkpoints/latest_epoch_0217.pth')

# Segment an image with bounding box
result = segmenter.predict('medical_image.png', box=[100, 100, 900, 900])

# Get results
mask = result['mask']  # Binary segmentation mask

# Save visualization
segmenter.visualize('medical_image.png', mask, 'result.jpg')

```

### 5. Evaluation

**Download Datasets**

We provide two datasets: [Optic Cup](https://drive.google.com/file/d/1jayJ9q627t6kNXNsacfW3b8i-oVPJ0wz/view?usp=sharing) and [Brain Tumor](https://drive.google.com/file/d/1WuJ8fD2stAqUKxYzsws2mMgS3M6JtFXK/view?usp=sharing) or prepare your own medical images.

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
TODO

Create a preprocessing script:
```python
# preprocess_data.py
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
- `-sam_config`: Configuration file
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

| Dataset | Model | Dice |
|---------|-------|------|
| OpticCup | SAM | ~0.61 |
| OpticCup | MedSAM | ~0.86 |
| OpticCup | SAM2 | ~0.62 |
| OpticCup | MedSAM2 | ~0.40 |
| OpticCup | **SAMed2** | **~0.90** |
| BrainTumor | SAM | ~0.56 |
| BrainTumor | MedSAM | ~0.60 |
| BrainTumor | SAM2 | ~0.44 |
| BrainTumor | MedSAM2 | ~0.581 |
| BrainTumor | **SAMed2** | **~0.67** |

## 🎮 Demo

Try our interactive demo powered by SliceUI! [Demo Guide](docs/DEMO.md)

[![Demo Video](https://img.youtube.com/vi/DEMO_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=DEMO_VIDEO_ID)

## 📚 Citation

If you find SAMed2 useful in your research, please consider citing:

```bibtex
TODO
```

## 🙏 Acknowledgement

This work builds upon several excellent projects:
- [SAM2](https://github.com/facebookresearch/segment-anything-2) - The foundation model we build upon
- [MedSAM](https://github.com/bowang-lab/MedSAM) - Inspiration for medical adaptation
- [SliceUI](https://github.com/yourusername/sliceUI) - Interactive demo interface

We thank the authors for their outstanding contributions to the community:

[Zhiling Yan¹](https://scholar.google.com/citations?user=xxx),
[Sifan Song²](https://scholar.google.com/citations?user=xxx),
[Dingjie Song¹](https://scholar.google.com/citations?user=xxx),
[Yiwei Li³](https://scholar.google.com/citations?user=xxx),
[Rong Zhou¹](https://scholar.google.com/citations?user=xxx),
[Weixiang Sun⁴](https://scholar.google.com/citations?user=xxx),
[Zhennong Chen²](https://scholar.google.com/citations?user=xxx),
[Sekeun Kim²](https://scholar.google.com/citations?user=xxx),
[Hui Ren²](https://scholar.google.com/citations?user=xxx),
[Tianming Liu³](https://scholar.google.com/citations?user=xxx),
[Quanzheng Li²](https://scholar.google.com/citations?user=xxx),
[Xiang Li²](https://scholar.google.com/citations?user=xxx),
[Lifang He¹](https://scholar.google.com/citations?user=xxx),
[Lichao Sun¹](https://scholar.google.com/citations?user=xxx)

¹Lehigh University
²Massachusetts General Hospital and Harvard Medical School
³University of Georgia, Athens
⁴University of Notre Dame 


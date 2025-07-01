<div align="center">

# 🏥 SAMed-2: Selective Memory Enhanced Medical SAM

### *Medical Image Segmentation with Memory-Enhanced SAM*

[![Project Page](https://img.shields.io/badge/🌐_Project-Website-blue)](https://zhilingyan.github.io/Medical-SAM-Bench/)
[![Paper](https://img.shields.io/badge/📄_Paper-Arxiv-purple)](https://arxiv.org/abs/xxxx.xxxxx)
[![Demo](https://img.shields.io/badge/🎮_Demo-SliceUI-green)](docs/DEMO.md)
[![Data](https://img.shields.io/badge/💾_Data-MedBank-orange)](docs/MEDBANK.md)
[![License](https://img.shields.io/badge/📜_License-Apache_2.0-yellow.svg)](LICENSE)

**The Official Repository of SAMed-2 & Medical SAM Benchmark**

[**Installation**](#-getting-started) • [**Models**](#3-model-zoo) • [**Quick Start**](#4-quick-start) • [**Citation**](#-citation--contributors)

</div>

---

## 🌟 Highlights

<table>
<tr>
<td width="33%" align="center"><b>🧠 Memory-Enhanced SOTA</b><br>Best performance on medical benchmarks</td>
<td width="33%" align="center"><b>🔧 Unified Framework</b><br>Fair comparison of all Medical SAM variants</td>
<td width="33%" align="center"><b>🎮 Interactive Tools</b><br>Demo and local UI for interactive medical image segmentation</td>
</tr>
</table>

## 📋 Abstract

SAMed-2 is a new foundation model for medical image segmentation built upon the SAM-2 architecture. Specifically, we introduce a temporal adapter into the image encoder to capture image correlations and a confidence-driven memory mechanism to store high-certainty features for later retrieval. This memory-based strategy counters the pervasive noise in large-scale medical datasets and mitigates catastrophic forgetting when encountering new tasks or modalities.

## 📰 News

> **[07/2025]** 🎮 Interactive demo tool released - try SAMed-2 on your medical images!  
> **[06/2025]** 🎉 SAMed-2 accepted by MICCAI 2025!  
> **[06/2025]** 🚀 Initial release of SAMed-2!

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

### 3. Model Zoo

<div align="center">

| Model | Architecture | Medical<br>Fine-tuned | Performance | Download |
|:-----:|:------------:|:---------------------:|:-----------:|:--------:|
| **SAMed-2** ⭐ | SAM2-Hiera-S | ✅ | **Best** | [📥 Download](https://drive.google.com/file/d/1JVmZnpWip7AIi8o9J1heog_Kl5uHGHcP/view?usp=sharing) |
| MedSAM2 | SAM2-Hiera-T | ✅ | Good | [📥 Download](https://drive.google.com/file/d/1XQmJ13-SahH-57eH1-UabU1OpGpoTZWT/view?usp=sharing) |
| MedSAM | SAM-ViT-B | ✅ | Good | [📥 Download](https://drive.google.com/file/d/1V81_3KuJ-7q1gzLYcQFPCTAAymfkxh6Y/view?usp=sharing) |
| SAM2 | SAM2-Hiera-S | ❌ | Baseline | [📥 Download](https://drive.google.com/file/d/1bNtsqOCRnzDOb_10EN9bAACLPew32yus/view?usp=sharing) |
| SAM | SAM-ViT-B | ❌ | Baseline | [📥 Download](https://drive.google.com/file/d/1LgRKsBkCYOeQQRWyF1RnXZgwe-_xfR0_/view?usp=sharing) |

</div>

> 📁 Place downloaded weights in `./checkpoints/`

#### 💾 Memory Bank (Required for SAMed-2)
Download the pre-trained memory bank: [**memory_bank_list_640.pkl**](https://drive.google.com/file/d/1nrq9GRhlCUG7ha-RuuQktODyfK1UKbwL/view?usp=sharing)  
Place it in the root directory of this repository.

### 4. Quick Start

**🚀 Simple Python API**

```python
from predict import MedicalSegmenter

# Initialize
segmenter = MedicalSegmenter(
    model_type='samed2',
    checkpoint_path='checkpoints/latest_epoch_0217.pth'
)

# Segment
result = segmenter.predict(
    'medical_image.png', 
    box=[100, 100, 900, 900]
)

# Visualize
segmenter.visualize(
    'medical_image.png', 
    result['mask'], 
    'result.jpg'
)
```

**📊 Benchmark Medical SAM Models**

```bash
# 🏆 SAMed-2 (Ours)
python main.py -net samed2 -sam_ckpt checkpoints/latest_epoch_0217.pth -sam_config sam2_hiera_s

# 🔬 MedSAM2
python main.py -net medsam2 -sam_ckpt checkpoints/MedSAM2_pretrain.pth -sam_config sam2_hiera_t_original

# 🏥 MedSAM
python main.py -net medsam -sam_ckpt checkpoints/medsam_vit_b.pth

# 🎯 SAM2
python main.py -net sam2 -sam_ckpt checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s_original

# 🔷 SAM
python main.py -net sam -sam_ckpt checkpoints/sam_vit_b_01ec64.pth
```

> 💡 **Common**: `-exp_name ${DATASET} -image_size 1024 -data_path /path/to/data -val_file_dir /path/to/test.txt`


### 5. Evaluation

#### 📦 Download Test Datasets

<div align="center">

| Dataset | Modality | Size | Download |
|:-------:|:--------:|:----:|:--------:|
| **Optic Cup** | Fundus | ~100MB | [📥 Download](https://drive.google.com/file/d/1jayJ9q627t6kNXNsacfW3b8i-oVPJ0wz/view?usp=sharing) |
| **Brain Tumor** | MRI | ~200MB | [📥 Download](https://drive.google.com/file/d/1WuJ8fD2stAqUKxYzsws2mMgS3M6JtFXK/view?usp=sharing) |

</div>

**Prepare Your Own Dataset**

For custom datasets, organize your data as follows:
```
your_dataset/
├── image/
│   ├── case_idx_slice_001.png
│   ├── case_idx_slice_002.png
│   └── ...
├── mask/
│   ├── case_idx_slice_001.png
│   ├── case_idx_slice_002.png
│   └── ...
└── test.txt  # List of test image names
```

**Run Evaluation**

```bash
# Full evaluation on the dataset
bash run.sh
```

**Parameters Explanation:**
- `-net`: Model type (`samed2`, `medsam2`, `sam2`, `medsam`, `sam`)
- `-exp_name`: Dataset name for logging
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

**📈 Performance Comparison**

<div align="center">

| Dataset | SAM | MedSAM | SAM2 | MedSAM2 | **SAMed-2** |
|:-------:|:---:|:------:|:----:|:-------:|:-----------:|
| **OpticCup** | 0.61 | 0.86 | 0.62 | 0.40 | **0.90** 🏆 |
| **BrainTumor** | 0.56 | 0.60 | 0.44 | 0.58 | **0.67** 🏆 |

<sub>*Dice scores on test sets. Higher is better.*</sub>

</div>

## 🎮 Demo

Try our interactive demo powered by SliceUI! [Demo Guide](docs/DEMO.md)

[![Demo Video](https://img.youtube.com/vi/DEMO_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=DEMO_VIDEO_ID)

## 📚 Citation & Contributors

If you find SAMed-2 useful in your research, please consider citing:

```bibtex
@article{yan2025samed2,
  title={SAMed-2: Selective Memory Enhanced Medical Segment Anything Model},
  author={Yan, Zhiling and Song, Sifan and Song, Dingjie and Li, Yiwei and Zhou, Rong and Sun, Weixiang and Chen, Zhennong and Kim, Sekeun and Ren, Hui and Liu, Tianming and Li, Quanzheng and Li, Xiang and He, Lifang and Sun, Lichao},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```

**Contributors:**

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

## 🙏 Acknowledgements

We gratefully acknowledge:
- **[SAM2](https://github.com/facebookresearch/segment-anything-2)** - The foundation model we build upon
- **[MedSAM](https://github.com/bowang-lab/MedSAM)** - Inspiration for medical adaptation
- **[SliceUI](https://github.com/yourusername/sliceUI)** - Interactive demo interface 


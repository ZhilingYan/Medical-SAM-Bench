# MedBank: A Comprehensive Medical Image Segmentation Dataset

## Overview

MedBank is a large-scale, diverse medical image segmentation dataset designed to train and evaluate foundation models for medical image analysis. We collected public datasets for pre-training and interval validation, covering various imaging modalities and anatomical structures.

## Dataset Statistics

- **Imaging Modalities**: Fundus, Dermoscopy, X-Ray, CT, MRI, Colonoscopy, Echo
- **Anatomical Coverage**: Lung, Skin, Spleen, Liver, Kidney, Eye, Stomach, and more

## Datasets and Information

| Dataset | Link | License | Reference |
| --- | --- | --- | --- |
| **BUSI** | [Link](https://www.kaggle.com/datasets/agneshyadav/breast-ultrasound-images-dataset-dataset-busi) | Database Contents License (DbCL) v1.0 | Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863. |
| **REFUGE** | [Link](https://refuge.grand-challenge.org/) | Academic use license | Orlando, José Ignacio, et al. "Refuge challenge: A unified framework for evaluating automated methods for glaucoma assessment from fundus photographs." Medical image analysis 59 (2020): 101570. |
| **RIM-ONE_r3** | [Link](https://www.idiap.ch/software/bob/docs/bob/bob.db.rimoner3/stable/index.html) | Academic use license | Fumero, Francisco, and Jose Sigut. "Alayón, Silvia andGonzález-Hernández." M., González de la Rosa, M.: Interactive tool and database for optic disc and cup segmentation of stereo and monocular retinal fundus images (06 2015) (2015). |
| **ISIC2018** | [Link](https://challenge.isic-archive.com/) | CC BY-NC 4.0 | Tschandl, Philipp, Cliff Rosendahl, and Harald Kittler. "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions." Scientific data 5.1 (2018): 1-9. |
| **Kvasir-SEG** | [Link](https://datasets.simula.no/kvasir-seg/) | CC BY 4.0 | Jha, Debesh, et al. "Kvasir-seg: A segmented polyp dataset." MultiMedia modeling: 26th international conference, MMM 2020, Daejeon, South Korea, January 5–8, 2020, proceedings, part II 26. Springer International Publishing, 2020. |
| **CVC-ClinicDB** | [Link](https://polyp.grand-challenge.org/CVCClinicDB/) | Academic use license | Bernal, Jorge, et al. "WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians." Computerized medical imaging and graphics 43 (2015): 99-111. |
| **CVC-ColonDB** | [Link](http://vi.cvc.uab.es/colon-qa/cvccolondb/) | Academic use license | Bernal, J., Sánchez, A., Vilariño, F., & Sappa, A. (2015). CVC-ColonDB: A Dataset for the Evaluation of Polyp Segmentation Methods. In MICCAI Endoscopic Vision Challenge 2015. |
| **Covid-QU-Ex** | [Link](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu) | CC BY-SA 4.0 | Tahir, Anas M., et al. "COVID-19 infection localization and severity grading from chest X-ray images." Computers in biology and medicine 139 (2021): 105002. |
| **Cell Segmentation Challenge** | [Link](https://www.nature.com/articles/s41592-024-02233-6) | Research use license | Ma, Jun, et al. "The multimodality cell segmentation challenge: toward universal solutions." Nature methods 21.6 (2024): 1103-1113. |
| **PlantDoc** | [Link](https://github.com/pratikkayal/PlantDoc-Dataset) | Creative Commons Attribution 4.0 International | Singh, Davinder, et al. "PlantDoc: A dataset for visual plant disease detection." Proceedings of the 7th ACM IKDD CoDS and 25th COMAD. 2020. 249-253. |
| **ACDC** | [Link](https://www.creatis.insa-lyon.fr/Challenge/acdc/) | CC BY-NC-SA 4.0 | Bernard, Olivier, et al. "Deep learning techniques for automatic MRI cardiac multi-structures segmentation and diagnosis: is the problem solved?." IEEE transactions on medical imaging 37.11 (2018): 2514-2525. |
| **CAMUS** | [Link](https://www.creatis.insa-lyon.fr/Challenge/camus/index.html) | CC BY-NC-SA 4.0 | Leclerc, Sarah, et al. "Deep learning for segmentation using an open large-scale dataset in 2D echocardiography." IEEE transactions on medical imaging 38.9 (2019): 2198-2210. |
| **BTCV** | [Link](https://www.synapse.org/Synapse:syn3193805/wiki/89480) | Academic use license | Landman, Bennett, et al. "Miccai multi-atlas labeling beyond the cranial vault–workshop and challenge." Proc. MICCAI multi-atlas labeling beyond cranial vault—workshop challenge. Vol. 5. 2015. |
| **LiTS** | [Link](https://competitions.codalab.org/competitions/17094) | Academic use license | Bilic, Patrick, et al. "The liver tumor segmentation benchmark (lits)." Medical image analysis 84 (2023): 102680. |
| **KiTS** | [Link](https://kits19.grand-challenge.org/) | Academic use license | Heller, Nicholas, et al. "The state of the art in kidney and kidney tumor segmentation in contrast-enhanced CT imaging: Results of the KiTS19 challenge." Medical image analysis 67 (2021): 101821. |
| **AMOS** | [Link](https://amos22.grand-challenge.org/) | Academic use license | Ji, Yuanfeng, et al. "Amos: A large-scale abdominal multi-organ benchmark for versatile medical image segmentation." Advances in neural information processing systems 35 (2022): 36722-36732. |
| **MSD Lung** | [Link](http://medicaldecathlon.com/) | CC-BY-SA 4.0 | Antonelli, Michela, et al. "The medical segmentation decathlon." Nature communications 13.1 (2022): 4128. |

## Dataset Structure

```
MedBank/
├── ObjectofInterest_Modality_ID/
│   ├── image/
│   │   ├── case_idx_slice_001.png
│   │   ├── case_idx_slice_002.png
│   │   └── ...
│   ├── mask/
│   │   ├── case_idx_slice_001.png
│   │   ├── case_idx_slice_002.png
│   │   └── ...
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── ObjectofInterest_Modality_ID/
│   └── ...
└── ...
```

## Data Format

- **Images**: PNG format, various resolutions (512×512 to 2048×2048)
- **Masks**: PNG format, binary segmentation masks
- **Split Files**: Text files containing image names for train/val/test splits


## Important Notes

### License Compliance
When using MedBank, please ensure compliance with the individual licenses of each dataset. Some datasets are restricted to academic use only, while others allow broader usage under specific Creative Commons licenses.

### Dataset Access
- Each dataset must be downloaded from its original source using the provided links
- Some datasets may require registration or agreement to specific terms
- Please cite the original papers when using specific datasets

## Citation

If you use MedBank in your research, please cite our work:

```bibtex
TODO
```

And the relevant original dataset citations from the table above.

## License

MedBank itself does not own any of the included datasets. Each dataset retains its original license as specified in the table above. Users must comply with the individual licenses of each dataset they use.

## Acknowledgments

We thank all the original dataset creators and the medical imaging community for making these valuable resources publicly available.

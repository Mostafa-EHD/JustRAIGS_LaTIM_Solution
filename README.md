# LaTIM's Solutions for JustRAIGS Challenge

This repository contains the implementation details and results of the LaTIM team's solutions for the Justified Referral in AI Glaucoma Screening (JustRAIGS) challenge. The challenge involved binary classification of referable glaucoma and no referable glaucoma (Task 1) and multi-label classification of ten additional features (Task 2) in fundus images.

## Challenge Description

The JustRAIGS challenge aimed to develop AI algorithms for glaucoma screening using color fundus photographs (CFPs). The dataset provided consisted of over 110,000 annotated fundus images, with each image labeled as either "referable glaucoma" or "no referable glaucoma." Additionally, images with referable glaucoma were further annotated with ten additional features related to glaucomatous damage.

### Tasks:
1. **Referral Performance (Task 1):** Binary classification of referable glaucoma and no referable glaucoma.
2. **Justification Performance (Task 2):** Multi-label classification of ten additional features related to glaucoma.

## Approach Overview

### Task 1: Referral Performance
- **Data Pre-processing:** Cropping, resizing, and intensity normalization of images.
- **Data Augmentation:** Random horizontal and vertical flips, rotation, and color jitter.
- **Proposed Model:** Ensemble of ResNet50 models trained independently on subsets of the data.
- **Training Details:** AdamW optimizer with dynamic learning rate adjustment based on epoch.

#### Proposed Model

![image](https://github.com/Mostafa-EHD/JustRAIGS_LaTIM_Solution/blob/main/Task1.png?raw=true)

### Task 2: Justification Performance
- **Dataset Utilization:** Focused on a subset of images labeled as referable glaucoma.
- **Data Pre-processing:** Resizing to specified dimensions for each model and data augmentation using Albumentation.
- **Model Architecture:** Ensemble of Eva Large, Deit3, and ResNet50 models.
- **Training Details:** Binary cross-entropy loss with AdamW optimizer and varied learning rates for different models.

#### Proposed Model

![image](https://github.com/Mostafa-EHD/JustRAIGS_LaTIM_Solution/blob/main/Task2.png?raw=true)

## Results

### Task 1: Referral Performance
The performance metrics reported in the table below for Sensitivity at 95% Specificity are provided on the challenge leaderboard and are derived from the independent test dataset provided by the challenge organizers.

#### Results Table

| Model description | Image size | Sensitivity at 95% specificity |
| ----------------- | ---------- | ------------------------------ |
| RetFound ('vit\_large\_patch16') | 224x224 | 0.662 |
| RetFound ('vit\_large\_patch16') | 600x600 | 0.675 |
| ResNet50 | 800x800 | **0.805** |
| ResNet50 Ensemble | 800x800 | **0.870** |

### Task 2: Justification Performance
The evaluation of Task 2 models based on Hamming loss.

#### Results Table

| Model configuration | Image size | Hamming loss |
| ------------------- | ---------- | ------------ |
| Eva | 336x336 | 0.313 |
| Ensemble (Eva + Deit3) | 336x336 / 384x384 | 0.253 |
| Ensemble (Eva + Deit3 + ResNet50) | 336x336 / 384x384 / 800x800 | **0.239** |

## Checkpoints
The checkpoints can be downloaded here: [Download models weights]([task1_proposed_model.png](https://drive.google.com/drive/folders/1v-YCpaZmgtgkQ3SwJ6Xu2ubuBu0OkfWt))

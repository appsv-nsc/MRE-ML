# MRE ML

Official PyTorch implementation accompanying the manuscript:

> **Deep Learning Estimation of Liver Storage and Loss Modulus Maps from Single-Direction Displacement-encoded MRE**

---

## Overview

This repository contains the implementation and supporting materials for the manuscript **"Deep Learning Estimation of Liver Storage and Loss Modulus Maps from Single-Direction Displacement-Encoded MRE."**

The repository includes code for data preprocessing and augmentation, model training and inference, and quantitative evaluation. Configuration files are provided for the pretext and downstream tasks, as well as for running the complete training pipeline. Additional supporting materials related to the experiments and analyses reported in the manuscript are also provided.

---

## Repository Structure

```text
├── SupplementaryMaterials
├── configs
│   ├── downstream
│   ├── full_pipeline
│   └── pretext
├── datasets
│   ├── __init__.py
│   ├── paths.py
│   ├── downstream
│   └── pretext
├── full_pipeline
│   ├── __init__.py
│   └── full_2d_pipeline.py
├── networks
│   ├── __init__.py
│   └── unet2d_custom.py
├── preprocessing
│   ├── downstream
│   └── pretext
├── testers
│   ├── post_analysis
│   ├── __init__.py
│   ├── base_tester.py
│   └── mre_tester.py
├── trainers
│   ├── __init__.py
│   ├── ae_trainer.py
│   ├── base_trainer.py
│   ├── mapping2d_trainer.py
│   └── rot_trainer.py
├── utils
│   ├── dice_loss.py
│   ├── evaluation.py
│   ├── losses.py
│   ├── metrics.py
│   ├── pipeline_tools.py
│   ├── recorder.py
│   └── tools.py
├── mre_ml_environment.yml
├── requirements.txt
└── README.md
```
---

## Requirements

The code was tested with the following environment:

* Python 3.8
* PyTorch 1.7.1
* CUDA 10.2
* cuDNN 7.6.5

---

## Installation

Clone the repository:

```bash
git clone https://github.com/appsv-nsc/MRE-ML.git
cd repository
```

Create and activate the Conda environment:

```bash
conda env create -f mre_ml_environment.yml
conda activate mre_ml_environment
```

Alternatively, create a Python 3.8 environment and install the required packages using `requirements.txt`:

```bash
conda create -n mre_ml_environment python=3.8
conda activate mre_ml_environment
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
cd path/to/project_folder
pip install -r requirements.txt
```

---

## Dataset and Preprocessing

The dataset used in this study is not publicly available due to institutional data-use and patient privacy restrictions.. However, the preprocessing and augmentation scripts used to prepare the data for the experiments are provided.

### Pretext Data

To preprocess the pretext data, run:

```bash
python preprocessing/pretext/preprocess_pretext_data.py
```

followed by:

```bash
python preprocessing/pretext/conc_pretext_data.py
```

To perform data augmentation:

```bash
python preprocessing/pretext/augment_pretext_data.py
```

### Downstream Data

To preprocess the downstream data, run:

```bash
python preprocessing/downstream/preprocess_downstream_data.py
```

To perform data augmentation:

```bash
python preprocessing/downstream/augment_downstream_data.py
```

---

## Training

Training can be performed in either of two ways:

1. Run pretext training first and then manually initialize downstream training using the pretrained weights.
2. Run the full training pipeline, which automates the pretext-to-downstream training workflow.

### Pretext and Downstream Training

For autoencoder pretraining, modify the corresponding configuration as needed and run:

```bash
python configs/pretext/ae_2d_config.py
```

For rotation-prediction pretraining, run:

```bash
python configs/pretext/rot_2d_config.py
```

After pretext training, specify the path to the pretrained model weights in the downstream configuration file and run:

```bash
python configs/downstream/mapping_2d_config.py
```

### Full Training Pipeline

Alternatively, the complete pretext-to-downstream training pipeline can be run using:

```bash
python configs/full_pipeline/mre_2d_pipeline_configs.py
```

The configuration classes define the parameters used for each experiment. These can be modified as needed. Example parameters include:

```python
  val_freq = 1
  num_workers = 0
  max_queue_size = num_workers * 1
  epochs = 500
  loss = 'target_mask_weighted_mse'
  loss_mul_factor = 5  # used with: loss = 'target_weighted_mse' only

  pretrained_model = 'checkpoints/full_pipeline/20260411/pretext_task_mre_ssl/rot_2d/rot_unet/SSM_ROT.pth'

  transferred_part = 'encoder'
```

Please refer to the configuration files for the complete set of available options.

---

## Inference and Evaluation

To load a trained downstream model and generate predictions, configure the model and data paths as required and run:

```bash
python testers/mre_tester.py
```

The inference pipeline saves the predicted maps as arrays and generates CSV files containing the pixel-wise reference and predicted values.

The generated CSV files can then be used to calculate NMSE and NMAE using:

```bash
python testers/post_analysis/CalculateNMSEandNMAE.py
```

---

## Implementation Notes

Parts of the self-supervised learning implementation in this repository were adapted from [Medical SSL](https://github.com/EndoluminalSurgicalVision-IMR/Medical-SSL), which accompanies the paper **"Dive into Self-Supervised Learning for Medical Image Analysis"** by Chuyan Zhang, Hao Zheng, and Yun Gu.
The original implementation was modified and extended for the MRE experiments presented in this work.

---
## Supplementary Materials

Additional materials accompanying the manuscript are provided in the [`SupplementaryMaterials`](./SupplementaryMaterials/) directory:

- [Demographic and laboratory characteristics of the study cohort](./SupplementaryMaterials/DatasetDemographicAndCharacteristics.pdf).

- [Input visualizations](./SupplementaryMaterials/InputVisualizations.pdf), including phase and magnitude images and the corresponding segmentation masks.

- [Supplementary Target-Weighted loss analysis](./SupplementaryMaterials/SuppCustomLoss.pdf), including error-distribution analysis across target-value bins and qualitative comparisons.

- [Extended qualitative results for the self-supervised learning experiments](./SupplementaryMaterials/ExtendedQualitativeResultsForSSL.pdf).

- [Supplementary Sensitivity analysis](./SupplementaryMaterials/SuppSensitivityAnalysis-I.pdf) to address potential heteroscedasticity of the differences between predicted and reference maps, supplementary log-ratio Bland–Altman (multiplicative LoA) and percentage Bland–Altman (relative LoA) formulations were computed in addition to the standard absolute-kPa formulation. Distribution-free bootstrap (5000 resamples) 95% confidence intervals for CCC are reported alongside the Fisher z-transformation intervals as a complementary inferential check. All sensitivity and robustness analyses are presented [here]

- [High-reference-value agreement analysis](./SupplementaryMaterials/SuppSensitivityAnalysis-II-(HighValued).pdf).
Because the test cohort was right-skewed in reference values, patient-level CCC over the full set is more influenced by the lower-value range. To probe agreement in the clinically critical higher-value regime, we additionally analyzed the five patients with the highest mean reference values for each biomarker (n = 5 of 13 test patients, selected to focus on the clinically elevated end of the reference distribution).

- [Tabulated quantitative agreement results](./SupplementaryMaterials/SuppSensitivityAnalysis-III.pdf) corresponding to Figs. 5–7 of the manuscript.

- [Additional Implementation Details](./SupplementaryMaterials/AdditionalImplementationDetails.pd) such as data augmentation and NMSE calculation details.
---


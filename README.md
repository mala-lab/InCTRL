# InCTRL (CVPR 2024)

Official PyTorch implementation of ["Toward Generalist Anomaly Detection via In-context Residual Learning with Few-shot Sample Prompts"](https://arxiv.org/pdf/2403.06495.pdf).

## Setup
- python >= 3.10.11
- torch >= 1.13.0
- torchvision >= 0.14.0
- scipy >= 1.10.1
- scikit-image >= 0.21.0
- numpy >= 1.24.3
- tqdm >= 4.64.0

## Run
#### Step 1. Download the Anomaly Detection Datasets

Download the Anomaly Detection Dataset and convert it to MVTec AD format. ([The convert script](https://github.com/mala-lab/InCTRL/tree/main/datasets/preprocess).)
The dataset folder structure should look like:
```
DATA_PATH/
    subset_1/
        train/
            good/
        test/
            good/
            defect_class_1/
            defect_class_2/
            defect_class_3/
            ...
    ...
```

#### Step 2. Save Training/Test Json Files of Datasets.([The save script](https://github.com/mala-lab/InCTRL/tree/main/datasets/preprocess).)

#### Step 3. Download the Few-shot Normal Samples for Inference on [Google Drive](https://drive.google.com/drive/folders/1_RvmTqiCc4ZGa-Oq-uF7SOVotE1RW5QZ?usp=drive_link)

#### Step 4. Download the Pre-train Models on [Google Drive](https://drive.google.com/file/d/1zEHsbbuUgBC4yuDu3g23wbUGmWmVyDRQ/view?usp=sharing)

#### Step 5. Quick Start

Change the `TEST.CHECKPOINT_FILE_PATH` in config to the path of pre-train model. and run
```python
python test.py --val_normal_json_path $normal-json-files-for-testing --val_outlier_json_path $abnormal-json-files-for-testing --category $dataset-class-name --dataset_dir $dataset-root --few_shot_dir $path-to-few-shot-samples
```
For example, if run on the category `SDD` with `k=2`:
```python
python test.py --val_normal_json_path /AD_json/SDD_val_normal.json --val_outlier_json_path /AD_json/SDD_val_outlier.json --category SDD --dataset_dir /Dataset/SDD_anomaly_detection --few_shot_dir /Few_shot/SDD/2/
```

## Training

```python
python main.py --normal_json_path $normal-json-files-for-training --outlier_json_path $abnormal-json-files-for-training --val_normal_json_path $normal-json-files-for-testing --val_outlier_json_path $abnormal-json-files-for-testing
```

## Citation

```bibtex
@article{zhu2024toward,
  title={Toward Generalist Anomaly Detection via In-context Residual Learning with Few-shot Sample Prompts},
  author={Zhu, Jiawen and Pang, Guansong},
  journal={arXiv preprint arXiv:2403.06495},
  year={2024}
}
```

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
#### Step 1. Download the Anomaly Detection Datasets and Save Training/Test Json Files

Download the Anomaly Detection Dataset and convert it to MVTec AD format. (For datasets we used in the paper, we provided [the convert and save script](https://github.com/mala-lab/InCTRL/tree/main/datasets/preprocess).)
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

#### Step 2. Download the Few-shot Normal Samples for Inference on [Google Drive](https://drive.google.com/drive/folders/1_RvmTqiCc4ZGa-Oq-uF7SOVotE1RW5QZ?usp=drive_link)

#### Step 3. Download the Pre-train Models on [Google Drive](https://drive.google.com/file/d/1zEHsbbuUgBC4yuDu3g23wbUGmWmVyDRQ/view?usp=sharing)

#### Step 4. Quick Start

```python
python test.py --val_normal_json_path $normal-json-files-for-testing --val_outlier_json_path $abnormal-json-files-for-testing --category $dataset-class-name --dataset_dir $dataset-root
```

## Training

```python
python main.py --normal_json_path $normal-json-files-for-training --outlier_json_path $abnormal-json-files-for-training --val_normal_json_path $normal-json-files-for-testing --val_outlier_json_path $abnormal-json-files-for-testing
```

## Citation

```bibtex
@inproceedings{zhu2024generalist,
      title={Toward Generalist Anomaly Detection via In-context Residual Learning with Few-shot Sample Prompts}, 
      author={Jiawen Zhu and Guansong Pang},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2024},
}
```

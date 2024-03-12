# InCTRL

Official PyTorch implementation of "Toward Generalist Anomaly Detection via In-context Residual Learning with Few-shot Sample Prompts".(accepted by CVPR 2024)

The code will be released soon.

## Setup

## Run
#### Step 1. Download the Anomaly Detection Datasets and Save Training/Test Json Files

Download the Anomaly Detection Dataset and convert it to MVTec AD format. (For datasets we used in the paper, we provided [the convert and save script](https://github.com/mala-lab/InCTRL/tree/main/datasets/preprocess).
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

#### Step 2. Download the Few-shot Normal Samples for Inference on [Google Drive]

#### Step 3. Download the Pre-train Models on [Google Drive]

#### Step 4. Quick Start

```python
python test.py --val_normal_json_path $normal-json-files-for-testing --val_outlier_json_path $abnormal-json-files-for-testing --category $dataset-class-name --dataset_dir $dataset-root
```


## Training

```python
python main.py --normal_json_path $normal-json-files-for-training --outlier_json_path $abnormal-json-files-for-training --val_normal_json_path $normal-json-files-for-testing --val_outlier_json_path $abnormal-json-files-for-testing
```




## Citation


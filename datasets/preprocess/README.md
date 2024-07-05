## Convert Script for Anomaly Detection Dataset

### Usage
#### Running the Convert Script
Running the corresponding convert script with the argument `dataset_root` as the root of the dataset.

e.g:
```bash
python convert_visa.py --dataset_root=./AD_json/visa
```

## Generate Json Files for Anomaly Detection Dataset

### Usage
#### Running the Generate Script

e.g:
```bash
python gen_val_json.py --dataset_dir=./visa_anomaly_detection/visa --dataset_name=candle 
```


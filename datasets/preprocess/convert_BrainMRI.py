import os
from sklearn.model_selection import train_test_split
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, help="dataset root")
args = parser.parse_args()

normal_root = os.path.join(args.dataset_root, 'no')
outlier_root = os.path.join(args.dataset_root, 'yes')

normal_fnames = os.listdir(normal_root)
outlier_fnames = os.listdir(outlier_root)

normal_train, normal_test, _, _ = train_test_split(normal_fnames, normal_fnames, test_size=0.25, random_state=42)

target_root = './BrainMRI_anomaly_detection/brainmri'
train_root = os.path.join(target_root, 'train/good')
if not os.path.exists(train_root):
    os.makedirs(train_root)
for f in normal_train:
    source = os.path.join(normal_root, f)
    shutil.copy(source, train_root)

test_normal_root = os.path.join(target_root, 'test/good')
if not os.path.exists(test_normal_root):
    os.makedirs(test_normal_root)
for f in normal_test:
    source = os.path.join(normal_root, f)
    shutil.copy(source, test_normal_root)

test_outlier_root = os.path.join(target_root, 'test/defect')
if not os.path.exists(test_outlier_root):
    os.makedirs(test_outlier_root)
for f in outlier_fnames:
    source = os.path.join(outlier_root, f)
    shutil.copy(source, test_outlier_root)

print("Done")
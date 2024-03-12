import os
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, help="dataset root")
args = parser.parse_args()

label_file = os.path.join(args.dataset_root, 'labels.csv')

data = np.loadtxt(label_file, dtype=int, delimiter=',', skiprows=1)

fnames = data[:, 0]
label = data[:, 1]

normal_fnames = fnames[label==0]
outlier_fnames = fnames[label==1]

normal_train, normal_test, _, _ = train_test_split(normal_fnames, normal_fnames, test_size=0.25, random_state=42)

target_root = './HeadCT_anomaly_detection/headct'
train_root = os.path.join(target_root, 'train/good')
if not os.path.exists(train_root):
    os.makedirs(train_root)
for f in normal_train:
    source = os.path.join(args.dataset_root, 'head_ct/head_ct/', '{:0>3d}.png'.format(f))
    shutil.copy(source, train_root)

test_normal_root = os.path.join(target_root, 'test/good')
if not os.path.exists(test_normal_root):
    os.makedirs(test_normal_root)
for f in normal_test:
    source = os.path.join(args.dataset_root, 'head_ct/head_ct/', '{:0>3d}.png'.format(f))
    shutil.copy(source, test_normal_root)

test_outlier_root = os.path.join(target_root, 'test/defect')
if not os.path.exists(test_outlier_root):
    os.makedirs(test_outlier_root)
for f in outlier_fnames:
    source = os.path.join(args.dataset_root, 'head_ct/head_ct/', '{:0>3d}.png'.format(f))
    shutil.copy(source, test_outlier_root)

print('Done')
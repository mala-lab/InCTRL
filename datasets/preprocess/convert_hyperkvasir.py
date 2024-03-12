import os
from sklearn.model_selection import train_test_split
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, help="dataset root")
args = parser.parse_args()

normal_root = os.path.join(args.dataset_root, 'upper-gi-tract', 'anatomical-landmarks')
outlier_root = os.path.join(args.dataset_root, 'upper-gi-tract', 'pathological-findings')

normal_dir = os.listdir(normal_root)
outlier_dir = os.listdir(outlier_root)

normal_fnames = list()
outlier_fnames = list()

for i in normal_dir:
    class_root = os.path.join(normal_root, i)
    fname = os.listdir(class_root)
    for f in fname:
        normal_fnames.append(os.path.join(class_root, f))

for i in outlier_dir:
    class_root = os.path.join(outlier_root, i)
    fname = os.listdir(class_root)
    for f in fname:
        outlier_fnames.append(os.path.join(class_root, f))

normal_train, normal_test, _, _ = train_test_split(normal_fnames, normal_fnames, test_size=0.25, random_state=42)

target_root = './hyperkvasir_anomaly_detection/hyperkvasir'
train_root = os.path.join(target_root, 'train/good')
if not os.path.exists(train_root):
    os.makedirs(train_root)
for f in normal_train:
    shutil.copy(f, train_root)

test_normal_root = os.path.join(target_root, 'test/good')
if not os.path.exists(test_normal_root):
    os.makedirs(test_normal_root)
for f in normal_test:
    shutil.copy(f, test_normal_root)

test_outlier_root = os.path.join(target_root, 'test')
if not os.path.exists(test_outlier_root):
    os.makedirs(test_outlier_root)
for f in outlier_fnames:
    class_name = f.split('/')[-2]
    target_root = os.path.join(test_outlier_root,class_name)
    if not os.path.exists(target_root):
        os.makedirs(target_root)
    shutil.copy(f, target_root)

print("Done")
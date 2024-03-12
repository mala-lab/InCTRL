import os
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, help="dataset root")
args = parser.parse_args()

label_file = os.path.join(args.dataset_root, 'labels.csv')

data = np.genfromtxt(label_file, dtype=['|S19', '<f8', '|S4'], names=[
                         'path', 'probability', 'type'])
image_fnames = np.char.decode(data['path'])
probs = data['probability']
types = np.char.decode(data['type'])

normal_fnames = image_fnames[probs==0]
normal_labels = probs[probs==0]
mask_outlier = probs==1
mask_mono = types=='mono'
mask_poly = types=='poly'
for i, (m1, m2) in enumerate(zip(mask_outlier, mask_poly)):
    mask_poly[i] = m1 & m2
for i, (m1, m2) in enumerate(zip(mask_outlier, mask_mono)):
    mask_mono[i] = m1 & m2

outlier_fnames_mono = image_fnames[mask_mono]
outlier_fnames_poly = image_fnames[mask_poly]

normal_train, normal_test, _, _ = train_test_split(normal_fnames, normal_labels, test_size=0.25, random_state=42)

target_root = './elpv_anomaly_detection/elpv'
train_root = os.path.join(target_root, 'train/good')
if not os.path.exists(train_root):
    os.makedirs(train_root)
for f in normal_train:
    source = os.path.join(args.dataset_root, f)
    shutil.copy(source, train_root)

test_normal_root = os.path.join(target_root, 'test/good')
if not os.path.exists(test_normal_root):
    os.makedirs(test_normal_root)
for f in normal_test:
    source = os.path.join(args.dataset_root, f)
    shutil.copy(source, test_normal_root)

test_outlier_root = os.path.join(target_root, 'test/mono')
if not os.path.exists(test_outlier_root):
    os.makedirs(test_outlier_root)
for f in outlier_fnames_mono:
    source = os.path.join(args.dataset_root, f)
    shutil.copy(source, test_outlier_root)

test_outlier_root = os.path.join(target_root, 'test/poly')
if not os.path.exists(test_outlier_root):
    os.makedirs(test_outlier_root)
for f in outlier_fnames_poly:
    source = os.path.join(args.dataset_root, f)
    shutil.copy(source, test_outlier_root)

print("Done")
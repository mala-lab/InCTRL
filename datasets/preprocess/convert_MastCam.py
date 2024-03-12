import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, help="dataset root")
args = parser.parse_args()

normal_train = list()
normal_test = list()
outlier_fnames = list()
normal_root = os.path.join(args.dataset_root, 'train_typical')
for file in os.listdir(normal_root):
    normal_train.append(os.path.join(normal_root, file))

test_normal_root = os.path.join(args.dataset_root, 'test_typical')
for file in os.listdir(test_normal_root):
    normal_test.append(os.path.join(test_normal_root, file))

outlier_root = os.path.join(args.dataset_root, 'test_novel')
for dir in os.listdir(outlier_root):
    class_root = os.path.join(outlier_root, dir)
    for file in os.listdir(class_root):
        outlier_fnames.append(os.path.join(class_root, file))

target_root = './MastCam_anomaly_detection/mastcam'
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

print('Done')
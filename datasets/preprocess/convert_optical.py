import os
from sklearn.model_selection import train_test_split
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, help="dataset root")
args = parser.parse_args()

normal_fnames = list()
outlier_fnames = list([0,0,0,0,0,0,0,0,0,0])
outlier_label = list([0,0,0,0,0,0,0,0,0,0])

for i in range(1,11):
    outlier_fnames[i-1] = list()
    outlier_label[i-1] = list()
    class_root = os.path.join(args.dataset_root, 'Class' + str(i), 'Class' + str(i))
    print(class_root)
    train_label = os.path.join(class_root, 'Train', 'Label', 'Labels.txt')
    test_label = os.path.join(class_root, 'Test', 'Label', 'Labels.txt')

    with open(train_label) as f:
        for line in f.readlines():
            if line == '1\n':
                continue
            data = line.replace('\n', '').split('\t')
            if data[1] == '0':
                normal_fnames.append(os.path.join(class_root, 'Train', data[2]))
            else:
                outlier_fnames[i-1].append(os.path.join(class_root, 'Train', data[2]))
                outlier_label[i-1].append(os.path.join(class_root, 'Train', 'Label',data[4]))

    with open(test_label) as f:
        for line in f.readlines():
            if line == '1\n':
                continue
            data = line.replace('\n', '').split('\t')
            if data[1] == '0':
                normal_fnames.append(os.path.join(class_root, 'Test', data[2]))
            else:
                outlier_fnames[i-1].append(os.path.join(class_root, 'Test', data[2]))
                outlier_label[i-1].append(os.path.join(class_root, 'Test', 'Label',data[4]))

normal_train, normal_test, _, _ = train_test_split(normal_fnames, normal_fnames, test_size=0.25, random_state=42)

target_root = './optical_anomaly_detection/optical_class'
train_root = os.path.join(target_root, 'train/good')
if not os.path.exists(train_root):
    os.makedirs(train_root)
for f in normal_train:
    fname = f.split('/')
    shutil.copy(f, os.path.join(train_root,fname[5] + '_' + fname[-2] + '_' + fname[-1]))

test_normal_root = os.path.join(target_root, 'test/good')
if not os.path.exists(test_normal_root):
    os.makedirs(test_normal_root)
for f in normal_test:
    fname = f.split('/')
    shutil.copy(f, os.path.join(test_normal_root,fname[5] + '_' + fname[-2] + '_' + fname[-1]))

for i, (of, ol) in enumerate(zip(outlier_fnames, outlier_label)):
    test_outlier_root = os.path.join(target_root, 'test/Class' + str(i))
    label_root = os.path.join(target_root, 'ground_truth/Class' + str(i))
    if not os.path.exists(test_outlier_root):
        os.makedirs(test_outlier_root)
    if not os.path.exists(label_root):
        os.makedirs(label_root)
    for image_f, label_f in zip(of, ol):
        image_fname = image_f.split('/')
        label_fname = label_f.split('/')
        shutil.copy(image_f, os.path.join(test_outlier_root,image_fname[5] + '_' + image_fname[-2] + '_' + image_fname[-1]))
        shutil.copy(label_f, os.path.join(label_root,label_fname[5] + '_' + label_fname[-3] + '_' + label_fname[-1]))

print("Done")
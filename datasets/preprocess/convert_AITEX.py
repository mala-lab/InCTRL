import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, help="dataset root")
args = parser.parse_args()

DEFEAT_CLASS = {'002': "Broken_end", '006': "Broken_yarn", '010': "Broken_pick",
                '016': "Weft_curling", '019': "Fuzzyball", '022': "Cut_selvage",
                '023': "Crease", '025': "Warp_ball", '027': "Knots",
                '029': "Contamination", '030': "Nep", '036': "Weft_crack"}

normal_images = list()
normal_fname = list()
outlier_images = list()
outlier_labels = list()
outlier_fname = list()


normal_root = os.path.join(args.dataset_root, 'NODefect_images')
normal_dirs = os.listdir(normal_root)
for dir in normal_dirs:
    files = os.listdir(os.path.join(normal_root, dir))
    for image in files:
        image_name = image.split('.')[0]
        image_data = cv2.imread(os.path.join(normal_root, dir, image))
        for i in range(16):
            normal_images.append(image_data[:, i*256:(i+1)*256 ,:])
            normal_fname.append(dir + '_' + image_name + '_' + str(i))

outlier_root = os.path.join(args.dataset_root, 'Defect_images/Defect_images')
label_root = os.path.join(args.dataset_root, 'Mask_images/Mask_images')
files = os.listdir(os.path.join(outlier_root))
for image in files:
    split_images = list()
    split_labels = list()
    image_name = image.split('.')[0]
    image_data = cv2.imread(os.path.join(outlier_root, image))
    label_data = cv2.imread(os.path.join(label_root, image_name + '_mask.png'))
    if image_data.shape[1] % image_data.shape[0] == 0:
        count = image_data.shape[1]//image_data.shape[0]
    else:
        count = image_data.shape[1] // image_data.shape[0] + 1
    for i in range(count):
        split_images.append(image_data[:, i * 256:(i + 1) * 256, :])
        split_labels.append(label_data[:, i * 256:(i + 1) * 256, :])
    for i, (im, la) in enumerate(zip(split_images, split_labels)):
        if np.max(la) != 0:
            outlier_images.append(im)
            outlier_labels.append(la)
            outlier_fname.append(image_name + '_' + str(i))

normal_train, normal_test, normal_name_train, normal_name_test = train_test_split(normal_images, normal_fname, test_size=0.25, random_state=42)

target_root = './AITEX_anomaly_detection/AITEX'
train_root = os.path.join(target_root, 'train/good')
if not os.path.exists(train_root):
    os.makedirs(train_root)
for image, name in zip(normal_train, normal_name_train):
    cv2.imwrite(os.path.join(train_root, name + '.png'), image)

test_root = os.path.join(target_root, 'test/good')
if not os.path.exists(test_root):
    os.makedirs(test_root)
for image, name in zip(normal_test, normal_name_test):
    cv2.imwrite(os.path.join(test_root, name + '.png'), image)

for image, label, name in zip(outlier_images, outlier_labels, outlier_fname):
    defect_class = DEFEAT_CLASS[name.split('_')[1]]
    defect_root = os.path.join(target_root, 'test', defect_class)
    label_root = os.path.join(target_root, 'ground_truth', defect_class)
    if not os.path.exists(defect_root):
        os.makedirs(defect_root)
    if not os.path.exists(label_root):
        os.makedirs(label_root)
    cv2.imwrite(os.path.join(defect_root, name + '.png'), image)
    cv2.imwrite(os.path.join(label_root, name + '_mask.png'), label)

print("Done")
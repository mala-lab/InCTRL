import csv
import logging
import shutil
from pathlib import Path
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, help="dataset root")
args = parser.parse_args()

def apply_cls1_split(root, new_root) -> None:
    """Apply the 1-class subset splitting using the fixed split in the csv file.

    adapted from https://github.com/amazon-science/spot-diff
    """
    categories = [
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
    ]

    split_file = os.path.join(root, "split_csv", "1cls.csv")

    for category in categories:
        train_folder = os.path.join(new_root, category, "train")
        test_folder = os.path.join(new_root, category, "test")
        mask_folder = os.path.join(new_root, category, "ground_truth")

        train_img_good_folder = os.path.join(train_folder, "good")
        test_img_good_folder = os.path.join(test_folder, "good")
        test_img_bad_folder = os.path.join(test_folder, "defect")
        test_mask_bad_folder = os.path.join(mask_folder, "defect")

        if not os.path.exists(train_img_good_folder):
            os.makedirs(train_img_good_folder)
        if not os.path.exists(test_img_good_folder):
            os.makedirs(test_img_good_folder)
        if not os.path.exists(test_img_bad_folder):
            os.makedirs(test_img_bad_folder)
        if not os.path.exists(test_mask_bad_folder):
            os.makedirs(test_mask_bad_folder)

    with open(split_file, "r", encoding="utf-8") as file:
        csvreader = csv.reader(file)
        next(csvreader)
        for row in csvreader:
            category, split, label, image_path, mask_path = row
            if label == "normal":
                label = "good"
            else:
                label = "defect"
            image_name = image_path.split("/")[-1]
            mask_name = mask_path.split("/")[-1]

            img_src_path = os.path.join(root, image_path)
            msk_src_path = os.path.join(root, mask_path)
            img_dst_path = os.path.join(new_root, category, split, label, image_name)
            msk_dst_path = os.path.join(new_root, category, "ground_truth", label, mask_name)

            shutil.copyfile(img_src_path, img_dst_path)
            if split == "test" and label == "defect":
                mask = cv2.imread(str(msk_src_path))

                # binarize mask
                mask[mask != 0] = 255

                cv2.imwrite(str(msk_dst_path), mask)

root = args.dataset_root
target_root = "../../visa_anomaly_detection/visa"
apply_cls1_split(root, target_root)
print("Done")
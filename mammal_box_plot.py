import argparse
import os
import logging
import sys
import itertools
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, ConcatDataset

from vision.datasets.mammal_dataset import MammalDataset

from tensorboardX import SummaryWriter
import cv2

if __name__ == "__main__":
    dataset_path = "D:/Datasets/iNaturalist2017/"
    dataset = MammalDataset(dataset_path, dataset_path + "/" + "mammal_train_2017_boxes.json")

    train_loader = DataLoader(dataset, 1,
                                num_workers=1,
                                shuffle=False)

    height_lteq_width = 0
    width_lteq_height = 0
    aspect_ratio_acc = 0
    index = 0
    for i, data in enumerate(tqdm(train_loader)):
            images, boxes, labels = data
            for box_batch in boxes:
                for box in box_batch:
                    box_width = box[3] - box[1]
                    box_height = box[2] - box[0]
                    if box_width > box_height:
                        height_lteq_width += 1
                        aspect_ratio_acc += box_width / box_height
                    else:
                        width_lteq_height += 1
                        aspect_ratio_acc += box_height / box_width
                    index += 1

    print("Aspect ratio: {}".format(aspect_ratio_acc / index))
    print("Landscape: {}".format(height_lteq_width))
    print("Portrait: {}".format(width_lteq_height))


                    
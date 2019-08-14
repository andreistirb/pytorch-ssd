import numpy as np
import logging
import pathlib
import json
import cv2
import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset

class MammalDataset(Dataset):
    
    def __init__(self, root_path, annotation_file_path, transform=None, target_transform=None):
        self.annotation_file = annotation_file_path
        self.transform = transform
        self.target_transform = target_transform
        self.categories = {}
        self.images = {}
        self.annotations = defaultdict(list)
        self.imgs_idx = []

        # preprocess json file
        json_file = open(self.annotation_file, "r")
        json_data = json.load(json_file)

        categories = json_data["categories"]
        for category in categories:
            self.categories[category["id"]] = category["name"]

        images = json_data["images"]
        for img in images:
            self.images[img["id"]] = img["file_name"]

        self.imgs_idx = self.images.keys()
        
        annotations = json_data["annotations"]
        for annotation in annotations:
            self.annotations[annotation["image_id"]].append((annotation["bbox"], annotation["category_id"]))

    def __get_item__(self, index):
        
        image_id = self.imgs_idx[index]
        image_path = self.images[image_id]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annotations = self.annotations[image_id]
        boxes, labels = map(list, zip(*annotations))

        boxes = self.convert_box(boxes)
        boxes = np.array(boxes)

        labels = np.array(labels)
        
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        return index

    def __len__(self):
        return 2

    def convert_box(self, boxes):
        boxes_return = []
        for box in boxes:
            m_box = box
            m_box[2] = m_box[0] + m_box[2]
            m_box[3] = m_box[3] + m_box[3]
            boxes_return.append(m_box)
        return boxes_return


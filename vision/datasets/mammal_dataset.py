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

        # preprocess json file
        json_file = open(self.annotation_file, "r")
        json_data = json.load(json_file)

        categories = json_data["categories"]
        for category in categories:
            self.categories[category["id"]] = category["name"]

        images = json_data["images"]
        for img in images:
            self.images[img["id"]] = img["file_name"]
        
        annotations = json_data["annotations"]
        for annotation in annotations:
            self.annotations[annotation["image_id"]].append((annotation["bbox"], annotation["category_id"]))

    def __get_item__(self, index):
        return index

    def get_image(self, index):
        return index

    def __len__(self):
        return 2


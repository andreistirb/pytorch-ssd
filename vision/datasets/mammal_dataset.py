from collections import defaultdict
import cv2
import json
import logging
import os
import numpy as np
import pathlib

import torch
from torch.utils.data import Dataset

from ..utils import box_utils
from ..ssd.config import mobilenetv1_ssd_config as config

class MammalDataset(Dataset):

    def __init__(self, root_path, annotation_file_path, transform=None, target_transform=None):
        self.root_path = root_path
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

        self.category_to_index = {}
        self.index_to_category = {}
        index = 1 # 0 is for background
        categories = json_data["categories"]
        #categories = categories.sort()

        for category in categories:
            self.categories[category["id"]] = category["name"]
            self.category_to_index[category["id"]] = index
            self.index_to_category[index] = category["id"]
            #index += 1

        images = json_data["images"]
        for img in images:
            self.images[img["id"]] = img["file_name"]

        self.imgs_idx = list(self.images.keys())
        
        annotations = json_data["annotations"]
        for annotation in annotations:
            self.annotations[annotation["image_id"]].append((annotation["bbox"], self.category_to_index[annotation["category_id"]]))

    def __getitem__(self, index):
        
        image_id = self.imgs_idx[index]
        image_path = self.root_path + "/" + self.images[image_id]

        image = cv2.imread(str(image_path))
        height, width, channels = image.shape
        cv_image = image.copy()
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        

        annotations = self.annotations[image_id]
        boxes, labels = map(list, zip(*annotations))

        boxes = self.convert_box(boxes)

        # For display purpose
        for box in boxes:
            cv2.rectangle(cv_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        #print(self.root_path + "/box/" + self.images[image_id])
        #cv2.imwrite(self.root_path + "/box/" + self.images[image_id], write_img)

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels)
        
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        plot_boxes = boxes.clone()
        plot_boxes = box_utils.convert_locations_to_boxes(plot_boxes, config.priors, config.center_variance,
                                                          config.size_variance)

        plot_priors = config.priors[labels.nonzero()]
        plot_priors = box_utils.center_form_to_corner_form(plot_priors)
        #plot_priors = plot_boxes[labels.nonzero()]
        plot_priors = np.array(plot_priors.data).squeeze()
        print(len(plot_priors.shape))
        if len(plot_priors.shape) == 1:
            xmin, ymin, xmax, ymax = plot_priors[0], plot_priors[1], plot_priors[2], plot_priors[3]
            cv2.rectangle(cv_image, (int(xmin * width), int(ymin * height)), (int(xmax * width), int(ymax * height)),
                          (0, 0, 255), 2)
        else:
            for box in plot_priors:
                xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
                cv2.rectangle(cv_image, (int(xmin * width), int(ymin * height)), (int(xmax * width), int(ymax * height)), (0, 0, 255), 2)

        # here we should plot the associated priors to the actual object
        #print(labels.nonzero())

        #cv_img = image.numpy()
        #cv_img = np.transpose(cv_img, (1,2,0))
        #cv2.imwrite("images/" + str(index) + ".jpg", cv_img)
        #print(image.shape)

        #cv2.imshow("Imagine", cv_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        cv.imwrite(self.root_path + "/priors/" + self.images[image_id], cv_image)
        return image, boxes, labels

    def get_image(self, index):
        return index

    def __len__(self):
        return len(self.imgs_idx)

    def convert_box(self, boxes):
        boxes_return = []
        for box in boxes:
            m_box = box
            m_box[2] = m_box[0] + m_box[2]
            m_box[3] = m_box[1] + m_box[3]
            boxes_return.append(m_box)
        return boxes_return

    def get_annotation(self, index):
        image_id = self.imgs_idx[index]
        annotation = self.annotations[image_id]
        return image_id, annotation


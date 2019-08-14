import torch
from torch.utils.data import DataLoader
from vision.datasets.mammal_dataset import MammalDataset

dataset_path = "D:/Datasets/iNaturalist2017/"
dataset = MammalDataset(dataset_path, dataset_path + "/" + "mammal_train_2017_boxes.json")
val_dataset = MammalDataset(dataset_path, dataset_path + "/" + "mammal_val_2017_boxes.json")

train_loader = DataLoader(dataset)
val_loader = DataLoader(val_dataset)

#for data in train_loader:
    #print(data)
#    x = 1

print(len(train_loader))
print(len(val_loader))

import matplotlib.pyplot as plt
from pandas.core.common import flatten
import copy
import numpy as np
import random
import torchvision
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset

import cv2

import glob
from tqdm import tqdm
####################################################
#       Create Train, Valid and Test sets
####################################################
train_data_path = 'train_images' 
test_data_path = 'test_images'

train_image_paths = [] #to store image paths in list
classes = [] #to store class values

#1.
# get all the paths from train_data_path and append image paths and class to to respective lists
# eg. train path-> 'images/train/26.Pont_du_Gard/4321ee6695c23c7b.jpg'
# eg. class -> 26.Pont_du_Gard
for data_path in glob.glob(train_data_path + '/*'):
    classes.append(data_path.split('/')[-1]) 
    train_image_paths.append(glob.glob(data_path + '/*'))

    
train_image_paths = list(flatten(train_image_paths))
random.shuffle(train_image_paths)

print('train_image_path example: ', train_image_paths[0])
print('class example: ', classes[0])

#2.
# split train valid from train paths (80,20)
train_image_paths, valid_image_paths = train_image_paths[:int(0.8*len(train_image_paths))], train_image_paths[int(0.8*len(train_image_paths)):] 

#3.
# create the test_image_paths
test_image_paths = []
for data_path in glob.glob(test_data_path + '/*'):
    test_image_paths.append(glob.glob(data_path + '/*'))

test_image_paths = list(flatten(test_image_paths))

print("Train size: {}\nValid size: {}\nTest size: {}".format(len(train_image_paths), len(valid_image_paths), len(test_image_paths)))
train_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                      torchvision.transforms.Normalize(
                                          mean = [0.485, 0.456, 0.406],
                                          std = [0.229, 0.224, 0.225],
                                      )])
test_transforms = transforms.Compose([ transforms.Resize((224, 224)), 
                                      transforms.ToTensor(), torchvision.transforms.Normalize(
                                          mean = [0.485, 0.456, 0.406], 
                                          std = [0.229, 0.224, 0.225],
                                      )
])
idx_to_class = {i:j for i, j in enumerate(classes)}
class_to_idx = {value:key for key,value in idx_to_class.items()}

class  InjuryDataset(Dataset):
    def __init__(self, image_paths, transform=False):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = image_filepath.split('/')[-2]
        label = class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        
        return image, label

train_dataset = InjuryDataset(train_image_paths,train_transforms)
valid_dataset = InjuryDataset(valid_image_paths,test_transforms) #test transforms are applied
test_dataset = InjuryDataset(test_image_paths,test_transforms)



def return_loaders():
    train_loader = DataLoader(
     train_dataset, batch_size=8, shuffle=True
    )

    valid_loader = DataLoader(
        valid_dataset, batch_size=8, shuffle=True
    )


    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False
    )
    return train_loader, valid_loader, test_loader

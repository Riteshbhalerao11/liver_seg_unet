#!/usr/bin/env python
# coding: utf-8

# In[2]:


from glob import glob 
import os
import shutil
from tqdm import tqdm

import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    AddChanneld,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
    Orientationd,
)
from monai.utils import set_determinism
from monai.data import Dataset, DataLoader

import matplotlib.pyplot as plt


# In[3]:


def prepare(data_dir, pixdim=(1.5, 1.5, 1.0), a_min=-200, a_max=200, spatial_size=[128,128,64], cache=False):
    set_determinism(seed=0)
    
    train_images = sorted(glob(os.path.join(data_dir,"train\data","*.nii")))
    train_labels = sorted(glob(os.path.join(data_dir,"train\labels","*.nii")))

    test_images = sorted(glob(os.path.join(data_dir,"test\data","*.nii")))
    test_labels = sorted(glob(os.path.join(data_dir,"test\labels","*.nii")))
    
    train_files = [{'vol':image_name, 'seg' : label_name } for image_name,label_name in zip(train_images,train_labels)]
    test_files = [{'vol':image_name, 'seg' : label_name } for image_name,label_name in zip(test_images,test_labels)]
    
    train_transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"]),
            AddChanneld(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True), 
            CropForegroundd(keys=["vol", "seg"], source_key="vol"),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size),   
            ToTensord(keys=["vol", "seg"]),

        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"]),
            AddChanneld(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max,b_min=0.0, b_max=1.0, clip=True), 
            CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size),   
            ToTensord(keys=["vol", "seg"]),

            
        ]
    )
    
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=1)

    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1)

    return train_loader, test_loader


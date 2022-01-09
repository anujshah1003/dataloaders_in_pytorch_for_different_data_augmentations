# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 11:36:49 2021

@author: Anuj

Script to create a custom dataset function in pytorch that loads the data
"""
import os

from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
from helpers import visualize

class DatasetLoader(Dataset):
    """ Tire Dataset Class Wrapper """

    def __init__(self, root_path, data_dir, annotation_file,data_type='train', \
                 data_transform=None,label_transform=None):
        """
        Args:
            data_dir (string):  directory with images
            annotation_file (string):  csv/txt file which has the 
                                        dataset labels
            data_transform: The trasforms to apply to data
            label_transform: The trasforms to apply to data

        """
        
        self.data_path = os.path.join(root_path,data_dir,data_type)
        self.label_path = os.path.join(root_path,annotation_file)
        if not data_transform:
            self.data_transform=transforms.ToTensor()
        else:
            self.data_transform=data_transform

        self._load_data()

    def _load_data(self):
        '''
        function to load the data in the format of [[img_name_1,label_1],
        [img_name_2,label_2],.....[img_name_n,label_n]]
        '''
        self.labels = pd.read_csv(self.label_path)
        
        self.loaded_data = []
        for i in range(self.labels.shape[0]):
            img_name = os.path.join(self.data_path, self.labels['img_names'][i])
            label = self.labels['labels'][i]
            #img,label=self._read_data(img_name,label)
            self.loaded_data.append((img_name,label))
            #self.loaded_data.append((img,label))

    def __len__(self):
        return len(self.loaded_data)

    def __getitem__(self, idx):

        idx = idx % len(self.loaded_data)
        img_name,label = self.loaded_data[idx]
       
        #img,label = self.loaded_data[idx]
        img,label = self._read_data(img_name,label)
        
        return img,label

    def _read_data(self,img_name,label):
        
        '''
        function to read the data
        '''
        img = Image.open(img_name)
        if self.data_transform:
            img = self.data_transform(img)
        return img, label
 
#%%    
if __name__=='__main__':
    root_path=r'D:\youtube\2021\data_aug_pytorch'
    data_dir='dogs-vs-cats'
    annotation_file='train_labels_mini.csv'
    transform_tr=transforms.Compose([transforms.Resize((128,128)),
                                        transforms.ToTensor()])
    
    training_dataset=DatasetLoader(root_path,data_dir,annotation_file,data_type='train',\
                          data_transform=transform_tr)
        
    #img,label=dataset.loaded_data[0]
    

    img,label=next(iter(training_dataset))
    #plt.imshow(img)
    
    train_dataloader = DataLoader(training_dataset, batch_size=16, shuffle=True)
    
    from itertools import islice
    
    for data,label in islice(train_dataloader,5):
        print(data.shape, label)
    visualize(data.numpy(),label,num_imgs=16)
    
from torch.utils.data.dataloader import default_collate
   
data_val = DataLoader(dataset,batch_size=5,collate_fn=collate_func,shuffle=False)

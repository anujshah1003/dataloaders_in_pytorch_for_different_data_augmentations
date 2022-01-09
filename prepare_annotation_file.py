# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:58:51 2021

@author: uid38717
"""
import os
import pandas as pd
from glob import glob
from tqdm import tqdm

def prepare_annotation_file(data_files,data_split='train',dest_filename='train_labels.csv',\
                            create_mini_dataset=False,create_imbalance=False):
    
    dest_df=pd.DataFrame(columns=columns)

    for img_name in tqdm(data_files):
        class_name=img_name.split('.')[0]
        label=labels_dict[class_name]
        row = pd.Series([img_name,label,class_name,data_split],index=dest_df.columns)
        dest_df=dest_df.append(row,ignore_index=True)
        dest_df.to_csv(dest_filename)
    if create_mini_dataset:
        cats=dest_df[dest_df['class_names']=='cat']
        cats_mini=cats.head(200)
        dogs=dest_df[dest_df['class_names']=='dog']
        dogs_mini=dogs.head(200)
        mini_dataset=pd.concat([cats_mini,dogs_mini],axis=0,ignore_index=True)
        mini_dataset.to_csv(dest_filename)
        
    if create_imbalance:
        cats=dest_df[dest_df['class_names']=='cat']
        cats_mini=cats.head(50)
        dogs=dest_df[dest_df['class_names']=='dog']
        dogs_mini=dogs.head(100)
        mini_dataset=pd.concat([cats_mini,dogs_mini],axis=0,ignore_index=True)
        mini_dataset.to_csv(dest_filename)


root_path=r'D:\youtube\2021\data_aug_pytorch'
data_folder='dogs-vs-cats'
data_path=os.path.join(root_path,data_folder)

dest_file='annotations.csv'

columns=['img_names','labels','class_names','data_split']
labels_dict={'cat':0,'dog':1}

train_files=os.listdir(os.path.join(data_path,'train'))
#test_files=os.listdir(os.path.join(data_path,'test1'))

#dest_df['class_names'].value_counts() 
       
prepare_annotation_file(train_files,data_split='train',dest_filename='train_labels_org.csv')
#prepare_annotation_file(test_files,data_split='test',dest_filename='test_labels_org.csv')

prepare_annotation_file(train_files,data_split='train',dest_filename='train_labels_mini.csv',create_mini_dataset=True)
prepare_annotation_file(train_files,data_split='train',dest_filename='train_labels_imbalanced.csv',create_imbalance=True)


    
    
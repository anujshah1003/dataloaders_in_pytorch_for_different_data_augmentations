
import torch
from torch.utils.data import DataLoader

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data_IDs, labels):
        """
        Initialization of the data in the desire format.
        we can have the image_name, label pair or read the data and have data,
        label pair
         [[img_name_1,label_1],[img_name_2,label_2],.....[img_name_n,label_n]]
         [[data_1,label_1],[data_2,label_2],.....[data_n,label_n]]
        """
        self.labels = labels
        self.data_IDs = data_IDs
        self.datset_pair=[]
        for label,data_ID in zip(self.data_IDs,self.labels):
            self.dataset_pair.append([data_ID,label])
            #self.dataset_pair.append([read_fn(data_ID),label])

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset_pair)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        data,label = self.list_IDs[index]
        
        #if you haven't read the data while iitialization read here
        #data=read_fn(data)
        # apply any transfirmation you want
        if self.transform:
            data=self.transform(data)
            label=self.transform(label)

        return data, label
    
# get the dataset
train_dataset=Dataset(train_labels,train_IDs)
# prepare the dataloader - a data generator function

train_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)

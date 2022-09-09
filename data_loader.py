import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
img_rows=40
img_cols=40
ncolors=1



def torch_expand_array(image):

    image = image.view(-1,3)
    
    expandedimage=torch.zeros((img_rows,img_cols,ncolors))

    expandedimage[image[:,0].long(),image[:,1].long()]=image[:,2].view((-1,ncolors))
    expandedimage=expandedimage.permute((2,0,1))


    return expandedimage




class TopTaggingDataset(Dataset):
    """Top tagging dataset."""

    def __init__(self, data,labels,weights,binnums,masses):
        """
        Args:
            datfile (string): File containing all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data=data
        self.labels=labels
        self.weights=weights
        self.binnums=binnums
        self.masses=masses
        
#        self.images, self.labels = read_images(datfile,Nread)

    def __len__(self):
        self.len=len(self.data)
        return self.len

    def __getitem__(self, idx):
        datum = self.data[idx]
        label = self.labels[idx]
        weight = self.weights[idx]
        binnum = self.binnums[idx]
        mass = self.masses[idx]
        sample = (datum, label,weight,binnum,mass)

        return sample
    

class TopTaggingDataset_Image(Dataset):
    """Top tagging dataset."""

    def __init__(self, data,labels,weights,binnums,masses):
        """
        Args:
            datfile (string): File containing all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data=data
        self.labels=labels
        self.weights=weights
        self.binnums=binnums
        self.masses=masses
        
#        self.images, self.labels = read_images(datfile,Nread)

    def __len__(self):
        self.len=len(self.data)
        return self.len

    def __getitem__(self, idx):
        datum = torch_expand_array(self.data[idx])
        label = self.labels[idx]
        weight = self.weights[idx]
        binnum =  self.binnums[idx]
        mass =  self.masses[idx]
        sample = (datum, label,weight,binnum,mass)

        return sample


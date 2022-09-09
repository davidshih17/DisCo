from __future__ import print_function, division

import sys,os
import argparse

parser = argparse.ArgumentParser(description='Train DNN with decorrelation using tau32 and frec as input.')
#parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                    help='an integer for the accumulator')
parser.add_argument('--decorr_mode', default='none',
                    help='decorrelation mode (none [default], plane, dist, distsq, cdf')
parser.add_argument('--gpunum', default='0',
                    help='gpu number (default 0)')
parser.add_argument('--alphamin', default='0',
                    help='minimum alpha (default 0)')
parser.add_argument('--alphamax', default='1',
                    help='maximum alpha (default 1)')
parser.add_argument('--nalpha', default='20',
                    help='number of alpha (default 20)')
parser.add_argument('--logfile', default='log.csv',
                    help='log file')
parser.add_argument('--label', default='',
                    help='label')

results = parser.parse_args(sys.argv[1:])
print(results)

decorr_mode=results.decorr_mode
gpunum=results.gpunum
alphamin=float(results.alphamin)
alphamax=float(results.alphamax)
nalpha=int(results.nalpha)
logfilename=results.logfile
label=results.label

os.environ["CUDA_VISIBLE_DEVICES"]=gpunum

import torch
import torch.nn as nn
#import torch.multiprocessing
#torch.multiprocessing.set_start_method('spawn')
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
import time

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import json
import h5py

from torch.nn import init

from data_loader import TopTaggingDataset
import model
from model import train,val,train_model
from networks import DNNclassifier

import pandas as pd

import glob

files = np.load("Data/W_FCN_v0.npz")
#test = files["test"]
train = files["train"]
val = files["val"]

varnames=['mass','pt','tau21', 'c2', 'd2','fw', 'pf','ap','zcutdef','ktdr','sqrtd12','label']

train[:,0]=train[:,0]*250+50
val[:,0]=val[:,0]*250+50

train[:,-2]=train[:,-2]/100
val[:,-2]=val[:,-2]/100

Nval=10000

WWpt_train=train[train[:,-1]==0][:,1]
qcdpt_train=train[train[:,-1]==1][:,1]
WWpt_val=val[val[:,-1]==0][:Nval,1]
qcdpt_val=val[val[:,-1]==1][:Nval,1]

Wmass_train=train[train[:,-1]==0][:,0]
qcdmass_train=train[train[:,-1]==1][:,0]
Wmass_val=val[val[:,-1]==0][:Nval,0]
qcdmass_val=val[val[:,-1]==1][:Nval,0]

Wdata_train=train[train[:,-1]==0][:,2:-1]
qcddata_train=train[train[:,-1]==1][:,2:-1]

Wdata_val=val[val[:,-1]==0][:Nval,2:-1]
qcddata_val=val[val[:,-1]==1][:Nval,2:-1]


# pT planing
Wminpt=min(WWpt_train)
Wmaxpt=max(WWpt_train)
qcdminpt=min(qcdpt_train)
qcdmaxpt=max(qcdpt_train)

nptbins=50
Wptbins=[Wminpt+(Wmaxpt-Wminpt)/nptbins*(i) for i in range(nptbins)]
qcdptbins=[qcdminpt+(qcdmaxpt-qcdminpt)/nptbins*(i) for i in range(nptbins)]

Wptbinnum_train=np.digitize(WWpt_train,bins=Wptbins)-1
qcdptbinnum_train=np.digitize(qcdpt_train,bins=qcdptbins)-1
Wptbinnum_val=np.digitize(WWpt_val,bins=Wptbins)-1
qcdptbinnum_val=np.digitize(qcdpt_val,bins=qcdptbins)-1

Wptbinwt_train,_=np.histogram(WWpt_train,bins=Wptbins+[Wmaxpt])
qcdptbinwt_train,_=np.histogram(qcdpt_train,bins=qcdptbins+[qcdmaxpt])
Wptbinwt_train=np.where(Wptbinwt_train==0,0,1/Wptbinwt_train/nptbins*len(WWpt_train))
qcdptbinwt_train=np.where(qcdptbinwt_train==0,0,1/qcdptbinwt_train/nptbins*len(qcdpt_train))

Wptbinwt_val,_=np.histogram(WWpt_val,bins=Wptbins+[Wmaxpt])
qcdptbinwt_val,_=np.histogram(qcdpt_val,bins=qcdptbins+[qcdmaxpt])
Wptbinwt_val=np.where(Wptbinwt_val==0,0,1/Wptbinwt_val/nptbins*len(WWpt_val))
qcdptbinwt_val=np.where(qcdptbinwt_val==0,0,1/qcdptbinwt_val/nptbins*len(qcdpt_val))



##########
traindata=torch.from_numpy(np.concatenate((Wdata_train,qcddata_train)).astype('float32'))
trainlabels=torch.from_numpy(np.concatenate((np.ones(len(Wdata_train)),np.zeros(len(qcddata_train)))).astype('int'))
trainweights=torch.from_numpy(np.concatenate((Wptbinwt_train[Wptbinnum_train]
                             ,qcdptbinwt_train[qcdptbinnum_train])).astype('float32'))
trainbinnums=torch.from_numpy(np.concatenate((Wptbinnum_train
                             ,qcdptbinnum_train)).astype('int'))
trainmass=torch.from_numpy(np.concatenate((Wmass_train,qcdmass_train)).astype('float32'))

##########
valdata=torch.from_numpy(np.concatenate((Wdata_val,qcddata_val)).astype('float32'))
vallabels=torch.from_numpy(np.concatenate((np.ones(len(Wdata_val)),np.zeros(len(qcddata_val)))).astype('int'))
valweights=torch.from_numpy(np.concatenate((Wptbinwt_val[Wptbinnum_val]
                             ,qcdptbinwt_val[qcdptbinnum_val])).astype('float32'))
valbinnums=torch.from_numpy(np.concatenate((Wptbinnum_val
                             ,qcdptbinnum_val)).astype('int'))
valmass=torch.from_numpy(np.concatenate((Wmass_val,qcdmass_val)).astype('float32'))

##########
    
trainset = TopTaggingDataset(traindata,trainlabels,trainweights,trainbinnums,trainmass)
valset = TopTaggingDataset(valdata,vallabels,valweights,valbinnums,valmass)

#, num_workers=4
my_batch_size=2048

train_loader = DataLoader(trainset, batch_size=my_batch_size,
                        shuffle=True,pin_memory=True)
val_loader = DataLoader(valset, batch_size=my_batch_size,
                        shuffle=True,pin_memory=True)

from torch.optim import Optimizer



logfile=open(logfilename, "w")
alphalist=np.linspace(alphamin,alphamax,nalpha,endpoint=False)
print(alphalist)
for alpha in alphalist:
    alpha=float(alpha)
    print('alpha',alpha)
#    lrschedule=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#    lrschedule=[10, 15, 20, 25, 30,35,40,45,50]
    net_c = DNNclassifier(9,2)
    print(net_c)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net_c.to(device)

    optimizer_c = torch.optim.Adam(net_c.parameters())
#    optimizer_c = Nadam(net_c.parameters())
#    print("Using Nadam optimizer!")
    output,labels=train_model(200,1e-4,net_c,optimizer_c,train_loader,val_loader
                              ,decorr_mode=decorr_mode    #,lrschedule=lrschedule
                              ,alpha=alpha,logfile=logfile,label=label)

logfile.close()


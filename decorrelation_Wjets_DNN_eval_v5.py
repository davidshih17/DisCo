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
testdata = files["test"]
valdata = files["val"]

varnames=['mass','pt','tau21', 'c2', 'd2','fw', 'pf','ap','zcutdef','ktdr','sqrtd12','label']

testdata[:,0]=testdata[:,0]*250+50
valdata[:,0]=valdata[:,0]*250+50

testdata[:,-2]=testdata[:,-2]/100
valdata[:,-2]=valdata[:,-2]/100

Wmass_test=testdata[testdata[:,-1]==0][:,0]
qcdmass_test=testdata[testdata[:,-1]==1][:,0]

Wmass_val=valdata[valdata[:,-1]==0][:,0]
qcdmass_val=valdata[valdata[:,-1]==1][:,0]

Wdata_test=testdata[testdata[:,-1]==0][:,2:-1]
qcddata_test=testdata[testdata[:,-1]==1][:,2:-1]

Wdata_val=valdata[valdata[:,-1]==0][:,2:-1]
qcddata_val=valdata[valdata[:,-1]==1][:,2:-1]

print(len(Wmass_test),len(qcdmass_test),len(Wmass_val),len(qcdmass_val))

##########
valdata=torch.from_numpy(np.concatenate((Wdata_val,qcddata_val)).astype('float32'))
vallabels=torch.from_numpy(np.concatenate((np.ones(len(Wdata_val)),np.zeros(len(qcddata_val)))).astype('int'))
valweights=torch.from_numpy(np.concatenate((np.ones(len(Wdata_val)),np.ones(len(qcddata_val)))).astype('float32'))
valbinnums=torch.from_numpy(np.concatenate((np.ones(len(Wdata_val)),np.zeros(len(qcddata_val)))).astype('int'))
valmass=torch.from_numpy(np.concatenate((Wmass_val,qcdmass_val)).astype('float32'))
##########
testdata=torch.from_numpy(np.concatenate((Wdata_test,qcddata_test)).astype('float32'))
testlabels=torch.from_numpy(np.concatenate((np.ones(len(Wdata_test)),np.zeros(len(qcddata_test)))).astype('int'))
testweights=torch.from_numpy(np.concatenate((np.ones(len(Wdata_test)),np.ones(len(qcddata_test)))).astype('float32'))
testbinnums=torch.from_numpy(np.concatenate((np.ones(len(Wdata_test)),np.zeros(len(qcddata_test)))).astype('int'))
testmass=torch.from_numpy(np.concatenate((Wmass_test,qcdmass_test)).astype('float32'))
##########

    
#trainset = TopTaggingDataset(traindata,trainlabels,trainweights,trainbinnums,trainmass)
valset = TopTaggingDataset(valdata,vallabels,valweights,valbinnums,valmass)
testset = TopTaggingDataset(testdata,testlabels,testweights,testbinnums,testmass)

#num_workers=4
my_batch_size=10000

val_loader = DataLoader(valset, batch_size=my_batch_size,
                        shuffle=True,pin_memory=True)
test_loader = DataLoader(testset, batch_size=my_batch_size,
                        shuffle=True,pin_memory=True)

from torch.optim import Optimizer
import glob
from evaluation import JSDvsR
import re, os

net_c = DNNclassifier(9,2)
print(net_c)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net_c.to(device)

alphalist=np.linspace(alphamin,alphamax,nalpha,endpoint=False)
print(alphalist)

print("output format: label,alpha, epoch, cl loss sig, cl loss bg, disco loss, R50, JSD50")

for alpha in alphalist:
    modelfiles=sorted(glob.glob('results/Wjets_DNN_dist_v4_unbiased/decorrelation_model_'+str(label)+"_"+str(alpha)+'_*.dict'),key=os.path.getmtime)
    
    for model in modelfiles:
        print(model)
        iepoch=re.split('_|\.dict',model)[-2]
#        if(int(iepoch)<150): continue

        net_c.load_state_dict(torch.load(model))
        output,labels,weights,_,masses,val_sig_loss1,val_bg_loss1,val_loss2=val(net_c,val_loader,decorr_mode=decorr_mode,alpha=float(alpha))
        Wout=output[labels==1.]
        qcdout=output[labels==0.]
        qcdmass=masses[labels==0.]

        JSDR50=JSDvsR(Wout,qcdout,qcdmass,sigeff=50)

        print("val ",label,alpha,iepoch,val_sig_loss1,val_bg_loss1,val_loss2,JSDR50[0],JSDR50[1])

        output2,labels2,weights2,_,masses2,test_sig_loss1,test_bg_loss1,test_loss2=val(net_c,test_loader,decorr_mode=decorr_mode,alpha=float(alpha))
        Wout=output2[labels2==1.]
        qcdout=output2[labels2==0.]
        qcdmass=masses2[labels2==0.]

        JSDR50=JSDvsR(Wout,qcdout,qcdmass,sigeff=50)

        print("test ", label,alpha,iepoch,test_sig_loss1,test_bg_loss1,test_loss2,JSDR50[0],JSDR50[1])

        print("\n")

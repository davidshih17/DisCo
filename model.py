from __future__ import print_function, division

import torch
import torch.nn as nn
#import torch.multiprocessing
#torch.multiprocessing.set_start_method('spawn')
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib
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
import csv


from sklearn.metrics import roc_curve
from scipy.stats import entropy # This is the KL divergence

from evaluation import JSD, JSDvsR,ABCD_metrics

from disco import distance_corr,distance_corr_unbiased

state={}

def calculate_loss(net,data,target,weight,mass,alpha=0.,decorr_mode='none'):

    output = net(data)
    output = F.softmax(output)[:,1]

    loss1_sig = F.binary_cross_entropy(output[target==0],target[target==0].float(),weight[target==0])
    loss1_bg = F.binary_cross_entropy(output[target==1],target[target==1].float(),weight[target==1])

    loss2=0.
    if(decorr_mode=='avg'):
        meany=output[(target==0)].mean()
        for ii in range(len(massbins)):
            templist=output[(target==0) & (binnum==ii)]
            if(len(templist)>0):
                loss2=0.5*(templist.mean()-meany)**2

    elif(decorr_mode=='corr'):

        yqcd=output[(target==0)]
        mqcd=mass[target==0]
        corr=((yqcd-yqcd.mean())*(mqcd-mqcd.mean())).mean()
        corr=corr/((yqcd-yqcd.mean())**2).mean().sqrt()
        corr=corr/((mqcd-mqcd.mean())**2).mean().sqrt()
        loss2=alpha*corr**2

    elif(decorr_mode=='dist'):

        yqcd=output[(target==0)]
        mqcd=mass[target==0]
#            ytop=output[(target==1)]
#            ycut=np.percentile(ytop.detach().cpu().numpy(),70)
        yqcd2=yqcd #[yqcd>ycut]
        mqcd2=mqcd #[yqcd>ycut]
        if(len(yqcd2)>1):
            wqcd=weight[target==0]
            normedweight=wqcd/(wqcd.sum())*len(wqcd) # weights should sum to n=size of sample
#                normedweight=torch.ones(len(yqcd2)).float().cuda()/len(yqcd2)
            dCorr=distance_corr(yqcd2,mqcd2,normedweight,power=1)
            loss2=alpha*dCorr
    elif(decorr_mode=='dist_unbiased'):
        yqcd=output[(target==0)]
        mqcd=mass[target==0]
        yqcd2=yqcd #[yqcd>ycut]
        mqcd2=mqcd #[yqcd>ycut]
        if(len(yqcd2)>2):
            wqcd=weight[target==0]
            normedweight=wqcd/(wqcd.sum())*len(wqcd) 
            dCorr=distance_corr_unbiased(yqcd2,mqcd2,normedweight,power=1)
            loss2=alpha*dCorr
    elif(decorr_mode=='distcut'):
        ysig=output[(target==1)]
        ycut=np.percentile(ysig.detach().cpu().numpy(),(1-sigeff)*100)
        yqcd=output[(target==0) & (output>ycut)]
        mqcd=mass[(target==0) & (output>ycut)]
        wqcd=weight[(target==0) & (output>ycut)]
#            ytop=output[(target==1)]
#            ycut=np.percentile(ytop.detach().cpu().numpy(),70)
        yqcd2=yqcd #[yqcd>ycut]
        mqcd2=mqcd #[yqcd>ycut]
        print(len(yqcd2))
        if(len(yqcd2)>1):
            normedweight=wqcd/(wqcd.sum())*len(wqcd) # weights should sum to n=size of sample
#                normedweight=torch.ones(len(yqcd2)).float().cuda()/len(yqcd2)
            dCorr=distance_corr(yqcd2,mqcd2,normedweight,power=1)
            loss2=alpha*dCorr
    elif(decorr_mode=='distsq'):
        yqcd=output[(target==0)]
        mqcd=mass[target==0]
#            ytop=output[(target==1)]
#            ycut=np.percentile(ytop.detach().cpu().numpy(),70)
        yqcd2=yqcd #[yqcd>ycut]
        mqcd2=mqcd #[yqcd>ycut]
        if(len(yqcd2)>1):
            wqcd=weight[target==0]
            normedweight=wqcd/(wqcd.sum())*len(wqcd) # weights should sum to n=size of sample
#                normedweight=torch.ones(len(yqcd2)).float().cuda()/len(yqcd2)
            dCorr=distance_corr(yqcd2,mqcd2,normedweight,power=2)
            loss2=alpha*dCorr
    elif(decorr_mode=='cdf'):
        ydist=output[(target==0)]
        std=ydist.std()
        binnum2=binnum[(target==0)]
#            ydist=output
#            maxy=max(ydist)
#            miny=min(ydist)
#            ydist=(ydist-miny)/(maxy-miny)
#            print('std',std)
        maxtensor=calc_maxtensor(ydist)
        for ii in range(len(massbins)):
#                ydistii=ydist[binnum2==ii]
            temptensor=maxtensor[binnum2==ii]
            if(len(temptensor)>0):
#                ybin=output[(target==0) & (binnum==ii)]
#                    print(alpha)
#                    print(qcdmassbinwt[ii])
                loss2+=1/std**2*alpha*(maxtensor.mean()+temptensor[:,binnum2==ii].mean()-2*temptensor.mean())
#                ybin=output[(binnum==ii)]
#             if(len(ybin)>0):
#                    ybin=(ybin-miny)/(maxy-miny)
#                 loss2+=alpha*KSsquare(ybin,ydist)

    return loss1_sig,loss1_bg,loss2,output

# train function (forward, backward, update)
def train(net,optimizer,dataloader,decorr_mode='none',alpha=0.1,sigeff=0.7):
    print('Training')
    t0 = time.time()
    Ntrain=len(dataloader.dataset)
    print(Ntrain)
    net.train()
    loss1_avg = 0.0
    loss2_avg = 0.0
    Ndata=0
    for batch_idx, (data, target,weight,binnum,mass) in enumerate(dataloader):
        Ndata+=len(data)
        if(Ndata>Ntrain):
            break
#        data = (expand_array(data)).astype('float32')
#        target = int(target)

        data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())
        weight=torch.autograd.Variable(weight.cuda())
        binnum=torch.autograd.Variable(binnum.cuda())
        mass=torch.autograd.Variable(mass.cuda())
#        data, target = data.to(device), target.to(device)

        print('minibatch',batch_idx+1,'/',int(len(dataloader)), end='\r')
        # forward
        
        # backward
#        print(data.shape,target.shape,output.shape)

        optimizer.zero_grad()
#        loss = F.cross_entropy(output, target)

        loss1_sig,loss1_bg,loss2,_=calculate_loss(net,data,target,weight,mass,alpha,decorr_mode)
        loss=loss1_sig+loss1_bg+loss2
        
        loss.backward()
        optimizer.step()

        # exponential moving average
        loss1_avg = loss1_avg * 0.2 + float(loss1_sig+loss1_bg) * 0.8
        loss2_avg = loss2_avg * 0.2 + float(loss2) * 0.8

#    print('std',std)
    print('\n')
    
    t1 = time.time()
    print('Training time: {} seconds'.format(t1 - t0))

    return loss1_avg,loss2_avg

    
# val function (forward only)
def val(net,dataloader,decorr_mode='none',alpha=0.1,cut=0.5):
    print('Validating')
    t1 = time.time()
    Nval=len(dataloader.dataset)
    print(Nval)
    net.eval()
    loss1_sig_avg = 0.0
    loss1_bg_avg = 0.0
    loss2_avg = 0.0
    correct = 0
    Nvalcount=0
    Nbatchcount=0
    outputall=torch.empty(0)
    labels=torch.empty(0)
    weights=torch.empty(0)
    binnums=torch.empty(0)
    masses=torch.empty(0)
    loss=0.
    for batch_idx, (data, target,weight,binnum,mass) in enumerate(dataloader):
        if(len(data)*batch_idx>Nval):
            break
#        data = (expand_array(data)).astype('float32')
#        target = int(target)
        data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())
        weight=torch.autograd.Variable(weight.cuda())
        mass=torch.autograd.Variable(mass.cuda())

        print('minibatch',batch_idx+1,'/',int(len(dataloader)), end='\r')

        labels=torch.cat((labels,target.float().data.cpu()))
        weights=torch.cat((weights,weight.float().data.cpu()))
        binnums=torch.cat((binnums,binnum.float().data.cpu()))
        masses=torch.cat((masses,mass.float().data.cpu()))

        with torch.no_grad():
            # forward

            loss1_sig,loss1_bg,loss2,output=calculate_loss(net,data,target,weight,mass,alpha,decorr_mode)
            # accuracy
#            pred = output.data.max(1)[1]
#            correct += float(pred.eq(target.data).sum())

            # val loss average
            loss1_sig_avg += float(loss1_sig)
            loss1_bg_avg += float(loss1_bg)
            loss2_avg += float(loss2)
            Nvalcount += len(data)
            Nbatchcount += 1
            outputall=torch.cat((outputall,output.float().data.cpu()))
    print('\n')
    
    loss1_sig_avg = loss1_sig_avg / Nbatchcount
    loss1_bg_avg = loss1_bg_avg / Nbatchcount
    loss2_avg = loss2_avg / Nbatchcount
#    state['val_accuracy'] = correct / Nvalcount

    t2 = time.time()
    print('Validation time: {} seconds'.format(t2 - t1))
    
    return outputall,labels,weights,binnums,masses,loss1_sig_avg,loss1_bg_avg,loss2_avg




#def JSD(hist1,hist2):
#    output=0.5*(entropy(hist1,0.5*(hist1+hist2))+entropy(hist2,0.5*(hist1+hist2)))
#    return output



def train_model(Nepochs,lr,net,optimizer,train_loader,val_loader,decorr_mode='none',alpha=0.1,lrschedule=[],logfile='log',minmass=50,maxmass=300,label=''):
    
#    val_acc_list=[]
    best_loss = 9999999999999999999.
    patience=0

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    for epoch in range(Nepochs):
        state={}
        if epoch in lrschedule:
            lr *= 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        print('epoch',epoch,'lr',param_group['lr'])

        state['epoch'] = epoch

        train_loss1,train_loss2=train(net,optimizer,train_loader,decorr_mode,alpha=alpha)

        output,labels,weights,_,masses,val_sig_loss1,val_bg_loss1,val_loss2\
            =val(net,val_loader,decorr_mode=decorr_mode,alpha=alpha)
        val_loss1=val_sig_loss1+val_bg_loss1
        
        t2=time.time()
        
        labels=labels.numpy()
        output=output.numpy()
        masses=masses.numpy()
        weights=weights.numpy()
        fpr,tpr,_=roc_curve(labels,output)
        val_acc=np.max(0.5*(tpr+1-fpr))
#        val_acc_list.append(val_acc)
        
        Wout=output[labels==1.]
        qcdout=output[labels==0.]
        qcdmass=masses[labels==0.]
        Wweights=weights[labels==1.]
        qcdweights=weights[labels==0.]

#        print(np.mean(qcdout))

        JSDR50=JSDvsR(Wout,qcdout,qcdmass,Wweights,qcdweights,minmass=minmass,maxmass=maxmass,sigeff=50)
        JSDR30=JSDvsR(Wout,qcdout,qcdmass,Wweights,qcdweights,minmass=minmass,maxmass=maxmass,sigeff=30)
        JSDR10=JSDvsR(Wout,qcdout,qcdmass,Wweights,qcdweights,minmass=minmass,maxmass=maxmass,sigeff=10)
              
        R50=JSDR50[0]
        JSD50=JSDR50[1]

        R30=JSDR30[0]
        JSD30=JSDR30[1]

        R10=JSDR10[0]
        JSD10=JSDR10[1]

    
        t3 = time.time()
        print('Calculating metrics time: {} seconds'.format(t3 - t2))
        

        print("Train loss, val loss, val accuracy, val R50, val R30: ",[train_loss1,train_loss2],[val_loss1,val_loss2],val_acc,R50,R30)
        print("JSD50, JSD30, JSD10: ", JSD50, JSD30, JSD10)
        state['JSD50',alpha,epoch]=JSD50
        state['JSD30',alpha,epoch]=JSD30
        state['JSD10',alpha,epoch]=JSD10
        state['val_accuracy',alpha,epoch]=val_acc
        state['val_loss',alpha,epoch]=val_sig_loss1+val_bg_loss1
        state['val_loss2',alpha,epoch]=val_loss2
        state['R50',alpha,epoch]=R50
        state['R30',alpha,epoch]=R30
        state['R10',alpha,epoch]=R10

        torch.save(net.state_dict(), "decorrelation_model_"+label+'_'+str(alpha)+"_"+str(epoch)+".dict")



        if(logfile=='log'):
            logfile=open('log', "w")

        w = csv.writer(logfile)
        for key, value in state.items():
            w.writerow([key, value])

#        if(patience>5): break


    return output,labels

        

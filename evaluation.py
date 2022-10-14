from scipy.stats import entropy # This is the KL divergence
import numpy as np
import matplotlib.pyplot as plt

def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

def JSD(hist1,hist2):
    output=0.5*(entropy(hist1,0.5*(hist1+hist2),base=2)+entropy(hist2,0.5*(hist1+hist2),base=2))
    return output


def JSDvsR(sigscore,bgscore,bgmass,sigweights=np.empty(0),bgweights=np.empty(0),sigeff=50,minmass=50,maxmass=300,nbins=50):
    if(len(sigweights)==0):
        sigweights=np.ones(len(sigscore))
    if(len(bgweights)==0):
        bgweights=np.ones(len(bgscore))
                         
    cutval=weighted_quantile(sigscore,1-sigeff/100,sample_weight=sigweights)
    bgp=bgmass[bgscore>cutval]
    bgf=bgmass[bgscore<cutval]
    bgweightsp=bgweights[bgscore>cutval]
    bgweightsf=bgweights[bgscore<cutval]
    print('JSDvsR',sigeff,np.sum(sigweights[sigscore>cutval])/np.sum(sigweights))
    countsp,_=np.histogram(bgp,weights=bgweightsp,range=[minmass,maxmass],bins=nbins,normed=True)
    countsf,_=np.histogram(bgf,weights=bgweightsf,range=[minmass,maxmass],bins=nbins,normed=True)

    output=[1/(np.sum(bgweightsp)/np.sum(bgweights)),1/JSD(countsp,countsf)]
    
    return output

def ABCDfunc_bg(bgout1,bgout2,y1list,y2list,weights=None):
    if(weights is None):
        weights=np.ones(len(bgout1))
    
    y1lower=y1list[0]
    y1upper=y1list[1]
    y2lower=y2list[0]
    y2upper=y2list[1]
    
    bg1=bgout1[(bgout1>y1lower) & (bgout2>y2lower)]
    bg2=bgout2[(bgout1>y1lower) & (bgout2>y2lower)]
    
    Aa=weights[(bg1>y1upper) & (bg2>y2upper)] # the SR
    Bb=weights[(bg1>y1upper) & (bg2<y2upper)] # VR
    Cc=weights[(bg1<y1upper) & (bg2>y2upper)] # VR
    Dd=weights[(bg1<y1upper) & (bg2<y2upper)] # the CR

    return np.sum(Aa),np.sum(Bb),np.sum(Cc),np.sum(Dd)

def ABCDfunc_sig(sigout1,sigout2,y1,y2,weights=None):
    
    Aa=weights[(sigout1>y1) & (sigout2>y2)] # the SR
    Bb=weights[(sigout1>y1) & (sigout2<y2)] # VR
    Cc=weights[(sigout1<y1) & (sigout2>y2)] # VR
    Dd=weights[(sigout1<y1) & (sigout2<y2)] # the CR

    return np.sum(Aa),np.sum(Bb),np.sum(Cc),np.sum(Dd)

# create a list of y1,y2 cut values based on signal efficiency eS
def create_y1y2list(sigout1,sigout2,eS,Nval,sigweights=None):
    if(sigweights is None):
        sigweights=np.ones(len(sigout1))
        
    y1y2list=[]
    for eS1 in np.linspace(eS,1,Nval):
        eS2=eS/eS1
        y1cut=weighted_quantile(sigout1,1-eS1,sample_weight=sigweights)
        y2cut=weighted_quantile(sigout2[sigout1>y1cut],1-eS2,sample_weight=sigweights[sigout1>y1cut])
    #    print(len(output1[(output1>y1cut) & (output2>y2cut)])/len(output1))
        y1y2list.append([y1cut,y2cut])
    for eS2 in np.linspace(eS,1,Nval):
        eS1=eS/eS2
        y2cut=weighted_quantile(sigout2,1-eS2,sample_weight=sigweights)
        y1cut=weighted_quantile(sigout1[sigout2>y2cut],1-eS1,sample_weight=sigweights[sigout2>y2cut])
    #    print(len(output1[(output1>y1cut) & (output2>y2cut)])/len(output1))
        y1y2list.append([y1cut,y2cut])
    y1y2list=np.array(y1y2list)
    return y1y2list

# calculate eB and eABCD for list of y1y2 cut values
def calculate_eB_eABCD(sigout1,sigout2,bgout1,bgout2,y1y2list,y1lower,y2lower,sigweights=None,bgweights=None):
    if(sigweights is None):
        sigweights=np.ones(len(sigout1))
    if(bgweights is None):
        bgweights=np.ones(len(bgout1))
        
    eBlist=[]
    eRlist=[]
    eRsiglist=[]
    for y1y2 in y1y2list:
        y1cut=y1y2[0]
        y2cut=y1y2[1]
#        y1lower=0
#        y2lower=0
        Abg,Bbg,Cbg,Dbg=ABCDfunc_bg(bgout1,bgout2,[y1lower,y1cut],[y2lower,y2cut],weights=bgweights)
        Asig,Bsig,Csig,Dsig=ABCDfunc_sig(sigout1,sigout2,y1cut,y2cut,weights=sigweights)
    #    print(Asig/(Asig+Bsig+Csig+Dsig))
#        eStry=Asig/(Asig+Bsig+Csig+Dsig)
        eBlist.append([y1lower,y1cut,y2lower,y2cut,Abg/np.sum(bgweights),np.sqrt(Abg)/np.sum(bgweights)])
        if(Dbg>0 and Abg>0 and Bbg>0 and Cbg>0):
            eRlist.append([y1lower,y1cut,y2lower,y2cut,Bbg*Cbg/(Dbg*Abg),
                             np.sqrt(
                                 (Bbg**2 * Cbg**2)/(Abg**2 * Dbg**3) 
                                 + (Bbg**2 * Cbg)/(Abg**2 * Dbg**2) 
                                 + (Bbg * Cbg**2)/(Abg**2 * Dbg**2) 
                                 + (Bbg**2 * Cbg**2)/(Abg**3 * Dbg**2))])                                 
            eRsiglist.append([y1lower,y1cut,y2lower,y2cut,Asig/Abg,Bsig/Bbg,Csig/Cbg,Dsig/Dbg])                                 
        else:
            eRlist.append([y1lower,y1cut,y2lower,y2cut,-1,-1])
            eRsiglist.append([y1lower,y1cut,y2lower,y2cut,-1,-1,-1,-1])                                 

    return eBlist,eRlist,eRsiglist



def ABCD_metrics(output1,output2,labels,weights=None):

    if(weights is None):
        weights=np.ones(len(labels))
       
    bgout1=output1[labels==0]
    bgout2=output2[labels==0]
    bgweights=weights[labels==0]
    
    sigout1=output1[labels==1]
    sigout2=output2[labels==1]
    sigweights=weights[labels==1]
    
    eBlist=[]
    eRlist=[]
    eRsiglist=[]

    for eStry in [0.1,0.3,0.6]:
        print(eStry)

        y1y2list=create_y1y2list(sigout1,sigout2,eStry,100,sigweights=sigweights)
        eBlist_temp,eRlist_temp,eRsiglist_temp=calculate_eB_eABCD(sigout1,sigout2,bgout1,bgout2,y1y2list\
                                                                  ,-1,-1,sigweights=sigweights,bgweights=bgweights)
        eBlist+=eBlist_temp
        eRlist+=eRlist_temp
        eRsiglist+=eRsiglist_temp

    eBlist=np.array(eBlist).reshape((3,-1,6))
    eRlist=np.array(eRlist).reshape((3,-1,6))
    eRsiglist=np.array(eRsiglist).reshape((3,-1,8))

    plt.figure()
    plt.scatter(1/eBlist[0,:,-2],
                    eRlist[0,:,-2],
                    s=5,label='R10 unbiased DD ')
    plt.scatter(1/eBlist[1,:,-2],
                    eRlist[1,:,-2],
                    s=5,label='R30 unbiased DD ')
    plt.scatter(1/eBlist[2,:,-2],
                    eRlist[2,:,-2],
                    s=5,label='R60 unbiased DD ')
    plt.plot(np.linspace(0,3000,10),np.linspace(1.1,1.1,10),linestyle='dashed')
    plt.plot(np.linspace(0,3000,10),np.linspace(0.9,0.9,10),linestyle='dashed')
    plt.ylim(0.,2)
    plt.xlim(1,3000)
    plt.legend(loc=(1.04,0))
    plt.xlabel('1/bgeff')
    plt.ylabel('ABCD closure')
    plt.title('Top Jet Images')
    plt.xscale('log')
    plt.show()        

    plt.figure()
    for itry in range(3):


        goodlist=np.abs(eRlist[itry,:,-2]-1)<0.1


        plt.scatter(1/eBlist[itry][goodlist][:,-2],eRsiglist[itry][goodlist][:,-3]/eRsiglist[itry][goodlist][:,-4],label='B',color='red')
        plt.scatter(1/eBlist[itry][goodlist][:,-2],eRsiglist[itry][goodlist][:,-2]/eRsiglist[itry][goodlist][:,-4],label='C',color='blue')
        plt.scatter(1/eBlist[itry][goodlist][:,-2],eRsiglist[itry][goodlist][:,-1]/eRsiglist[itry][goodlist][:,-4],label='D',color='green')
        plt.legend()
        plt.xlabel('Rx')
        plt.xlim(1,3000)
        plt.xscale('log')
        plt.ylim(0,0.5)
    plt.show()        

    return eBlist,eRlist,eRsiglist
    
#    state['JSD50',alpha,epoch]=JSD50
#    state['JSD30',alpha,epoch]=JSD30
#    state['JSD10',alpha,epoch]=JSD10
#    state['val_accuracy1',alpha,epoch]=val_acc1
#    state['val_accuracy2',alpha,epoch]=val_acc2
#    state['val_loss1',alpha,epoch]=val_loss1
#    state['val_loss2',alpha,epoch]=val_loss2
#    state['val_loss3',alpha,epoch]=val_loss3
#    state['R50',alpha,epoch]=R50
#    state['R30',alpha,epoch]=R30
#    state['R10',alpha,epoch]=R10

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class DNNclassifier(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self,ninput,numout):
        """ Constructor
        """

        super(DNNclassifier, self).__init__()


        self.dense_1 = nn.Linear(ninput, 64)
        self.bn2 = nn.BatchNorm1d(ninput)
        self.bn64 = nn.BatchNorm1d(64)
        self.dense_2 = nn.Linear(64, 64)
        self.dense_3 = nn.Linear(64, 64)
        self.classifier = nn.Linear(64, numout)
        init.kaiming_normal(self.classifier.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0



    def forward(self, x):
        x = self.dense_1.forward(x)
        x = F.relu(x)
        x = self.bn64(x)
        x = self.dense_2.forward(x)
        x = F.relu(x)
        x = self.dense_3.forward(x)
        x = F.relu(x)
        return self.classifier(x)

    
class CNNclassifier(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self,numout):
        """ Constructor
        """

        super(CNNclassifier, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=64
                                , kernel_size=4, stride=1, padding=1)   
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=32
                                , kernel_size=4, stride=1, padding=1)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=32
                                , kernel_size=4, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(in_channels=32, out_channels=32
                                , kernel_size=4, stride=1, padding=1)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dense_1 = nn.Linear(2048, 32)
        self.dense_2 = nn.Linear(32, 64)
        self.dense_3 = nn.Linear(64, 64)
        self.classifier = nn.Linear(64, numout)
        init.kaiming_normal(self.classifier.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0



    def forward(self, x):
#        print('1',x.shape)
#        x = F.pad(x, [0, 1, 0, 1])
        x = self.conv_1.forward(x)
        x = F.relu(x)
#        print('2',x.shape)
        x = self.conv_2.forward(x)
        x = F.relu(x)
#        print('3',x.shape)
        x = self.maxpool_1.forward(x)
#        print('4',x.shape)
        x = self.conv_3.forward(x)
        x = F.relu(x)
#        print('5',x.shape)
        x = self.conv_4.forward(x)
        x = F.relu(x)
#        print('6',x.shape)
        x = self.maxpool_2.forward(x)
#        print('7',x.shape)
        x = x.view(-1, 2048)
        x = self.dense_1.forward(x)
        x = F.relu(x)
#        print('8',x.shape)
        x = self.dense_2.forward(x)
        x = F.relu(x)
#        print('9',x.shape)
        x = self.dense_3.forward(x)
        x = F.relu(x)
#        print('10',x.shape)
#        print('8',x.shape)
        return self.classifier.forward(x)
    

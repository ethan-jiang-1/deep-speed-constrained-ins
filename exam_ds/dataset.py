# dataset definition
#
# Description:
#   Define dataset and import data.
#
# Copyright (C) 2018 Santiago Cortes
#
# This software is distributed under the GNU General Public 
# Licence (version 2 or later); please refer to the file 
# Licence.txt, included with the software, for details.


import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset  # , DataLoader
#from torchvision import transforms, utils
#from torch.autograd import Variable
#import csv

#dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Dataset class
class OdometryDataset(Dataset):
    def __init__(self, data_folder, datasets, transform=None, labs=None, using_cuda=False):
        self.data_folder = data_folder
        self.datasets = datasets
        self.labs = labs
        self.using_cuda = using_cuda

        """
        Args:
            data_folder (string): Path to the csv file with annotations.
            datasets: list of datasets.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        ind=0
        self.imu=[]
        self.imut=[]
        self.pos=[]
        self.post=[]
        self.limits=[]
        self.limits.append(0)
        # plot=False
        self.cache = {}
        #scroll trough folders and attach data. Since there is not that many sequences, one array is used.
        for dataset in datasets:
            imu_path=data_folder+dataset+"iphone/imu-gyro.csv"
            data = pd.read_csv(imu_path,names=list('tlabcdefghijk'))
            pos=data[data['l']==7]
            imu=data[data['l']==34]

            self.imut.append(imu[list('t')])      # ts for imu
            self.imu.append(imu[list('abcdef')])  # w_x, w_y, w_z, a_x, a_y, a_z
            self.post.append(pos[list('t')])      # ts for pos
            self.pos.append(pos[list('bcd')])     # pos_x, pos_y, pos_z  
            
            self.transform = transform
            self.limits.append(self.limits[ind]+len(self.imu[ind])-300)
             
            ind=ind+1
        print("ODS on {} yiels samples: {} ".format(datasets, len(self)))

    def plot_dataset_internals(self, skip_ratio=2):
        datasets = self.datasets
        ind = 0
        for i, dataset in enumerate(datasets):
            if i % skip_ratio != 0:
                print("skip plot", dataset)
                continue

            print("plot ", dataset)
            plt.figure()
            plt.title(dataset)
            plt.subplot(211)
            plt.plot(self.pos[ind].values)
            #print(np.shape(np.diff(self.pos[ind].values,axis=0,n=1)))
            dt=np.diff(self.post[ind].values,axis=0)
            #print(np.shape(dt))
            plt.plot(np.mean((np.diff(self.pos[ind],axis=0)/dt[:,None]),0))
            #plt.figure()
            plt.subplot(212)
            plt.plot(self.imu[ind].values)
            #plt.show()                
            ind=ind+1        

    # Define the length of the dataset as the number of sequences that can be extracted.  
    def __len__(self):
        return int(np.floor((self.limits[-1]-1)/100))
    
    # read a sample of 100 measurements, avoiding the edges of the diferent captures.
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        if idx > len(self):
            raise ValueError('Index out of range(0)')

        idx = idx*100
        dset = None 
        ndx = None      
        for index in range(0,len(self.limits)):
            # search dset(subset from external file) which contain data for idx
            if idx >= self.limits[index] and idx < self.limits[index+1]:
                dset=index
                off = np.random.randint(low=50, high=100)
                ndx = idx - self.limits[index] + off
                break
        if dset is None:
            raise ValueError("Index out of range(1)")
        
        #IMU (200, 6)
        IMU=self.imu[dset][ndx:ndx+200].values
        #acc=IMU[0:3][1]
        #IMU (200, 6) -> (6,200)
        IMU=IMU.swapaxes(0, 1)

        #t (200, 1)
        t=(self.imut[dset])[ndx:ndx+200].values

        #scalar time: ti:from / te:to in sec
        ti=np.min(t)
        te=np.max(t)

        #filter to help filter out what we need out of large seq
        inde=np.logical_and([self.post[dset]['t'].values<te] , [self.post[dset]['t'].values>ti])
        inde=np.squeeze(inde)

        #posi is selected pos from ti to te in dset
        posi=self.pos[dset][inde].values
        dp=np.diff(posi,axis=0)

        #dt is selected post(ts for pos) from ti to te in dset
        tsi = self.post[dset][inde].values
        dt=np.diff(tsi,axis=0)

        minv = np.min(np.sqrt(np.sum(np.square(dp / dt), axis=1)))
        maxv = np.max(np.sqrt(np.sum(np.square(dp / dt), axis=1)))

        #dT=tsi[-1]-tsi[0]
        #dP=posi[:][-1]-posi[:][0]
        # gt=dP/dT  # should be same as below
        gt=np.mean((dp / dt), axis=0)

        # construct the imu -> gt mapping
        # imu(feature: acce and gyro) (6, 200) -> dt(speed: dp/dt) (3)
        gt=gt.astype('float')
        IMU=IMU.astype('float')

        if self.using_cuda:
            gt = torch.Tensor(gt).cuda()
            IMU = torch.Tensor(IMU).cuda()

        sample={'imu':IMU,'gt':gt,'time':tsi[0],'range':[minv,maxv]}
        if self.transform:
            sample = self.transform(sample)
        
        self.cache[idx] = sample
        return sample

# Trasform sample into tensor structure.
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        imu=sample['imu']
        if isinstance(imu, np.ndarray):
            gt=sample['gt']
            T=sample['time']
            R=sample['range']
            #print(type(R))
            return {'imu': torch.from_numpy(imu),'gt':torch.from_numpy(gt),'time':T,'range':R}
        else:
            return sample


#import torch
#import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
#from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils
#from torch.autograd import Variable
#import subprocess
#import time
import csv
#import traceback
#import math
#from torchsummary import summary

try:
    import os
    #print(os.getcwd())
    os.chdir("exam_ds")
    #print(os.getcwd())
except:
    pass

#Import python functions.
try:
    from exam_ds.dataset import OdometryDataset
    from exam_ds.dataset import ToTensor

except:
    from dataset import OdometryDataset
    from dataset import ToTensor



def get_data_folders_and_labs():
    #add path to used folders
    #Advio
    folders=[]
    #for i in [13,15,16,17,1,2,3,5,6,8,9,10,11,12,18,19,20,21,22]:  
    for i in [1,2,3,5,6,8,9,10,11,12,13,15,16,17,18,19,20,21,22]:  
        path= '/advio-'+str(i).zfill(2)+'/'
        folders.append(path)
    #for i in [4,7,14,17]:
    #    path= '/advio-'+str(i).zfill(2)+'/'
    #    folders.append(path)         
    
    #Extra data
    folders.append("/static/dataset-01/")
    folders.append("/static/dataset-02/")
    folders.append("/static/dataset-03/")
    #folders.append("/static/dataset-04/")
    folders.append("/swing/dataset-01/")
    #folders.append("/swing/dataset-02/")

    #Load saved motion labels
    labs=[]
    with open('labels.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            labs.append([int(row[0]),int(row[1]),int(row[2]),float(row[3]),])
    return folders, labs



def find_and_plot_data_labels(folders, labs):
    #visualize labels in sample vector.
    ind=0
    acc_lab=0
    acc_dat=0
    data_labels=[]
    #if plot:
    #    plt.figure(figsize=(8, 35))
    for idx, folder in enumerate(folders):
        #Load one folder at a time
        data=OdometryDataset("../data_ds",[folder],transform=ToTensor())
        #Skip last label from previous dataset
        while labs[ind][3]==-2:
            ind=ind+1               
        #Find corresponding labels
        stay=True
        dat=[]   
        dat.append([-1,0])    
        while stay:
            tim=labs[ind][3]
            tim=np.round(np.floor(tim)*60+(tim-np.floor(tim))*100)
            data_length=(2+(data[len(data)]['time'])-data[0]['time'])[0]        
            if labs[ind][3]==-1:            
                stay=False
                tim=10000
            lab=labs[ind][2]
            dat.append([tim,lab])
            ind=ind+1
              
        #Make label vector for each sample
        label=[]
        start=data[0]['time']
        for i in range(0,len(data)):
            t=data[i]['time']-start
            for j in range(0,len(dat)-1):
                if t<dat[j+1][0] and t>dat[j][0]:
                    label.append(dat[j+1][1])
        #plot results
        acc_dat=acc_dat+len(data)
        acc_lab=acc_lab+len(label)

        # if plot:
        #     ax = plt.subplot(23,1,idx+1)
        #     ax.set_title(folder)
        #     plt.plot(label)
        #     plt.ylim(-1,5)
        #     frame1 = plt.gca()
        #     frame1.axes.get_xaxis().set_visible(False)
        #     plt.yticks([0,1,2,3,4], ['Standing','Walking','Stairs','Escalator','Elevator'])
        #     plt.grid(b=True,axis='y')   
        data_labels.append(label)
    
    return data_labels

def ex_plot_sub_dataset(T):
    folders = T.datasets
    labs = T.labs 

    #visualize labels in sample vector.
    ind=0
    acc_lab=0
    acc_dat=0


    plt.figure(figsize=(8, 35))
    for idx, folder in enumerate(folders):
        #Load one folder at a time
        data=OdometryDataset("../data_ds",[folder],transform=ToTensor())
        #Skip last label from previous dataset
        while labs[ind][3]==-2:
            ind=ind+1               
        #Find corresponding labels
        stay=True
        dat=[]   
        dat.append([-1,0])    
        while stay:
            tim=labs[ind][3]
            tim=np.round(np.floor(tim)*60+(tim-np.floor(tim))*100)
            data_length=(2+(data[len(data)]['time'])-data[0]['time'])[0]        
            if labs[ind][3]==-1:            
                stay=False
                tim=10000
            lab=labs[ind][2]
            dat.append([tim,lab])
            ind=ind+1
              
        #Make label vector for each sample
        label=[]
        start=data[0]['time']
        for i in range(0,len(data)):
            t=data[i]['time']-start
            for j in range(0,len(dat)-1):
                if t<dat[j+1][0] and t>dat[j][0]:
                    label.append(dat[j+1][1])
        #plot results
        acc_dat=acc_dat+len(data)
        acc_lab=acc_lab+len(label)

        ax = plt.subplot(23,1,idx+1)
        ax.set_title(folder)
        plt.plot(label)
        plt.ylim(-1,5)
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        plt.yticks([0,1,2,3,4], ['Standing','Walking','Stairs','Escalator','Elevator'])
        plt.grid(b=True,axis='y')   


def exam_dataset(T):
    print(T)
    pass




#plot velocity and speed.
def ex_plot_dataset(T):
    velo=[]
    sp=[]
    t=[]
    index=(np.round(np.linspace(0,len(T),1000)))
    for i in index:
        data=T[int(i)]
        velo.append(data['gt'].numpy())
        sp.append((data['gt'].norm()))
        t.append(data['time'])
    
    plt.figure()
    plt.plot(velo)
    plt.title('Velocity Vector (entrie train set, interval 1000)')
    plt.xlabel('sample')
    plt.ylabel('Speed (m/s)')
    plt.legend(['x','z','y'])
    plt.figure()
    plt.title('Speed (entrie train set, interval 1000)')
    plt.xlabel('sample')
    plt.ylabel('Speed (m/s)')
    plt.plot(sp)

def load_dataset():
    folders, labs = get_data_folders_and_labs()

    data_labels = find_and_plot_data_labels(folders, labs)
    
    # Create dataset reader.
    print("Final OdometryDataset")
    T = OdometryDataset("../data_ds", folders, transform=ToTensor(), labs=labs)
    return T,  data_labels

class DataLoaderDs(object):
    @classmethod
    def load_dataset(cls):
        T, data_labels = load_dataset()

        exam_dataset(T)

        return T, data_labels

    @classmethod
    def plot_dataset(cls, T):
        ex_plot_dataset(T)

    @classmethod
    def plot_sub_dataset(cls, T):
        ex_plot_sub_dataset(T)

    @classmethod
    def plot_dataset_internals(cls, T):
        T.plot_dataset_internals()
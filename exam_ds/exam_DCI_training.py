# Example Implementation
#
# Description:
#   train and test DCI network.
#
# Copyright (C) 2018 Santiago Cortes
#
# This software is distributed under the GNU General Public 
# Licence (version 2 or later); please refer to the file 
# Licence.txt, included with the software, for details.


import torch
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
import subprocess
import time
import csv
import traceback
import math
from torchsummary import summary

# dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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
    from exam_ds.exam_model import ExamModelDs as Emdl
except:
    from dataset import OdometryDataset
    from dataset import ToTensor
    from exam_ds.exam_model import ExamModelDs as Emdl

def get_data_folders_and_labs():
    #add path to used folders
    #Advio
    folders=[]
    for i in [13,15,16,17,1,2,3,5,6,8,9,10,11,12,18,19,20,21,22]:  
        path= '/advio-'+str(i).zfill(2)+'/'
        folders.append(path)  
    #Extra data
    folders.append("/static/dataset-01/")
    folders.append("/static/dataset-02/")
    folders.append("/static/dataset-03/")
    folders.append("/swing/dataset-01/")

    #Load saved motion labels
    labs=[]
    with open('labels.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            labs.append([int(row[0]),int(row[1]),int(row[2]),float(row[3]),])
    return folders, labs



def plot_data_labels(folders, labs):
    #visualize labels in sample vector.
    ind=0
    acc_lab=0
    acc_dat=0
    data_labels=[]
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
        plt.subplot(23,1,idx+1)
        plt.plot(label)
        plt.ylim(-1,5)
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        plt.yticks([0,1,2,3,4], ['Standing','Walking','Stairs','Escalator','Elevator'])
        plt.grid(b=True,axis='y')   
        data_labels.append(label)
    return data_labels


def exam_dataset(T):
    print(T)
    pass

#plot velocity and speed.
def plot_dataset(T):
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
    plt.title('Velocity Vector')
    plt.xlabel('sample')
    plt.ylabel('Speed (m/s)')
    plt.legend(['x','z','y'])
    plt.figure()
    plt.title('Speed')
    plt.xlabel('sample')
    plt.ylabel('Speed (m/s)')
    plt.plot(sp)

def get_val_gt(data_gt):
    val = torch.norm(data_gt, 2, 1)
    val = val.type(torch.FloatTensor)
    return val.numpy()

def get_pred_gt_vals(model, data):
    val_pred = Emdl.eval_pred(model, data['imu'])
    val_preds = val_pred.ravel().tolist()

    val_gt = get_val_gt(data['gt'])
    val_gts = val_gt.ravel().tolist()
    
    return val_preds, val_gts

def plot_pred_speed_ordered(model, T):
    ordered_Loader = DataLoader(T, batch_size=1, shuffle=False, num_workers=1)

    # Load corresponding prediction and ground truth
    pred_sp=[]
    gt_sp=[]
    for i_batch, sample_batched in enumerate(ordered_Loader):
        data = sample_batched

        val_preds, val_gts = get_pred_gt_vals(model, data)

        pred_sp += val_preds
        gt_sp += val_gts

    # Plot prediction and ground truth.
    print(np.shape((np.asarray(gt_sp))))
    plt.figure()
    plt.subplot(211)
    plt.plot(pred_sp)
    plt.ylabel('Speed (m/s)')
    plt.title('Prediction speed')
    plt.subplot(212)
    plt.plot(gt_sp)
    plt.ylabel('Ground truth speed')


def plot_bunch_confused(model, T, data_labels):
    ordered_Loader = DataLoader(T, batch_size=1, shuffle=False, num_workers=1)

    dat_lab=[]
    for label in data_labels:
        dat_lab=dat_lab+label


    #Plot scatter of prediction and ground truth with labels.
    pred_sp=[]
    gt_sp=[]
    R=[]
    for i_batch, sample_batched in enumerate(ordered_Loader):
        data=sample_batched

        val_preds, val_gts = get_pred_gt_vals(model, data)

        pred_sp += val_preds
        gt_sp += val_gts

        R.append(np.array(data['range']))
    print(len(dat_lab))
    print(len(gt_sp))
    pred=np.asarray(pred_sp)
    sp=np.asarray(gt_sp)    
    stat=[]
    stair=[]
    walk=[]
    esc=[]
    ele=[]

    Rstat=[]
    Rstair=[]
    Rwalk=[]
    Resc=[]
    Rele=[]

    #Separte by label
    for i in range(0,len(dat_lab)):
        if dat_lab[i]==0:
            stat.append([sp[i],pred[i]])
            Rstat.append(R[i])
        elif dat_lab[i]==1:
            walk.append([sp[i],pred[i]])
            Rwalk.append(R[i])
        elif dat_lab[i]==2:
            stair.append([sp[i],pred[i]])
            Rstair.append(R[i])
        elif dat_lab[i]==3:
            esc.append([sp[i],pred[i]])
            Resc.append(R[i])
        else:
            ele.append([sp[i],pred[i]])
            Rele.append(R[i])
    msize=3
    plt.figure(figsize=(8,8))
    #Scatter plot.
    test=np.array(stat)
    plt.plot(test[:,0],test[:,1],'r.',label='static',markersize=msize)
    test=np.array(stair)
    plt.plot(test[:,0],test[:,1],'g.',label='stair',markersize=msize)
    test=np.array(walk)
    plt.plot(test[:,0],test[:,1],'b.',label='walk',markersize=msize)
    test=np.array(esc)
    plt.plot(test[:,0],test[:,1],'k.',label='escalator',markersize=msize)
    test=np.array(ele)
    plt.plot(test[:,0],test[:,1],'y.',label='elevator',markersize=msize)

    plt.plot([0,1.5],[0,1.5],'k')
    plt.xlabel('gt (m/s)')
    plt.ylabel('prediction (m/s)')

    #plot histograms by label
    axes=plt.gca()
    axes.set_xlim((0.0,1.5))
    axes.set_ylim([0.0,1.5])
    axes.legend()


    #axes.grid(b=True, which='major', color='k', linestyle='--')
    bins=np.linspace(0.0,2.0,20)
    f=0
    plt.figure()
    plt.subplot(511)
    plt.title('minimum')
    plt.ylabel('static')
    test=np.array(Rstat)
    plt.hist(test[:,f],bins=bins)
    plt.subplot(512)
    plt.ylabel('stairs')
    test=np.array(Rstair)
    plt.hist(test[:,f],bins=bins)
    plt.subplot(513)
    plt.ylabel('walk')
    test=np.array(Rwalk)
    plt.hist(test[:,f],bins=bins)
    plt.subplot(514)
    plt.ylabel('escalator')
    test=np.array(Resc)
    plt.hist(test[:,f],bins=bins)
    plt.subplot(515)
    plt.ylabel('elevator')
    test=np.array(Rele)
    plt.hist(test[:,f],bins=bins)

    f=1
    plt.figure()
    plt.subplot(511)
    plt.title('maximum')
    plt.ylabel('static')
    test=np.array(Rstat)
    plt.hist(test[:,f],bins=bins)
    plt.subplot(512)
    plt.ylabel('stairs')
    test=np.array(Rstair)
    plt.hist(test[:,f],bins=bins)
    plt.subplot(513)
    plt.ylabel('walk')
    test=np.array(Rwalk)
    plt.hist(test[:,f],bins=bins)
    plt.subplot(514)
    plt.ylabel('escalator')
    test=np.array(Resc)
    plt.hist(test[:,f],bins=bins)
    plt.subplot(515)
    plt.ylabel('elevator')
    test=np.array(Rele)
    plt.hist(test[:,f],bins=bins)



def plot_model_on_test_dataset(model):
    plt.figure()
    # Evaluate in unknown data to the network.
    nfolders=[]
    nfolders.append("/static/dataset-04/")
    Test = OdometryDataset("./../data_ds/",nfolders,transform=ToTensor())
    test_Loader = DataLoader(Test, batch_size=1,shuffle=False, num_workers=1)

    pred_sp=[]
    gt_sp=[]
    t=[]
    for i_batch, sample_batched in enumerate(test_Loader):
        data=sample_batched

        val_preds, val_gts = get_pred_gt_vals(model, data)

        pred_sp += val_preds
        gt_sp += val_gts


        t.append(data['time'])
    plt.subplot(211)
    plt.plot(pred_sp)
    plt.ylabel('Predicted speed')
    plt.subplot(212)
    plt.plot(gt_sp)
    plt.ylabel('ground truth speed')

    fig = plt.figure(figsize=(6,6))
    plt.plot(np.asarray(gt_sp),np.asarray(pred_sp),'.', label='test data')
    plt.plot([0,2],[0,2],'k')
    plt.xlabel('gt (m/s)')
    plt.ylabel('prediction (m/s)')

    axes=plt.gca()

    axes.set_xlim((0.0,2))
    axes.set_ylim([0.0,2])
    axes.legend()


def plot_traning(tls, vls):
    # Plot loss
    plt.figure()
    plt.plot(np.log(np.array(tls)),label = 'Training loss')
    plt.plot(np.log(np.array(vls)),label = 'Validation loss')

def load_dataset():

    folders, labs = get_data_folders_and_labs()

    data_labels = plot_data_labels(folders, labs)
    
    # Create dataset reader.
    T = OdometryDataset("../data_ds", folders, transform=ToTensor())
    return T,  data_labels



def plot_model_and_pred_on_train_dataset(model, T, data_labels):
    # Load corresponding prediction and ground truth
    plot_pred_speed_ordered(model, T)

    plot_bunch_confused(model, T, data_labels)


def run_main(load_model=False):


    T, data_labels = load_dataset()

    exam_dataset(T)
    plot_dataset(T)

    model = None
    try:    
        #load pretrained model or create new one.
        if load_model:
            model = Emdl.get_model_from_trained_model()
        else:
            model, tls, vls = Emdl.get_model_from_new_training(T, epochs_num=1)
            plot_traning(tls, vls)
    
    except Exception as ex:
        print("Exception occured: ", ex)
        print(traceback.format_exc())

    Emdl.exam_model(model)

    plot_model_and_pred_on_train_dataset(model, T, data_labels)
    plot_model_on_test_dataset(model)


    plt.show()


if __name__ == "__main__":

    run_main(load_model=False)





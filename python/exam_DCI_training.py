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

import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    import os
    #print(os.getcwd())
    os.chdir("python")
    #print(os.getcwd())
except:
    pass

#Import python functions.
try:
    from python.dataset import OdometryDataset
    from python.dataset import ToTensor
except:
    from dataset import OdometryDataset
    from dataset import ToTensor


# Model
class vel_regressor(torch.nn.Module):
    def __init__(self, Nin=6, Nout=1, Nlinear=112*60):
        super(vel_regressor, self).__init__()
        # Convolutional layers
        self.model1 = torch.nn.Sequential(
        torch.nn.Conv1d(Nin,60,kernel_size=3,stride=1,groups=Nin),
        torch.nn.ReLU(),
        torch.nn.Conv1d(60,120,kernel_size=3,stride=1,groups=Nin),
        torch.nn.ReLU(),
        torch.nn.Conv1d(120,240,kernel_size=3,stride=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool1d(10, stride=6),
        )
        # Fully connected layers
        self.model2=model2=torch.nn.Sequential(
        torch.nn.Linear(Nlinear, 10*40),
        torch.nn.ReLU(),
        torch.nn.Linear(10*40, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, Nout)
        )
        
    # Forward pass
    def forward(self, x):
        x = self.model1(x)
        x = x.view(x.size(0), -1)
        x = self.model2(x)
        return x

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
        data=OdometryDataset("../data",[folder],transform=ToTensor())
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


def plot_pred_speed_ordered(model, ordered_Loader):
    # Load corresponding prediction and ground truth
    pred=[]
    sp=[]
    for i_batch, sample_batched in enumerate(ordered_Loader):
        data=sample_batched
        pred.append(model(Variable(data['imu'].float())).data[0].numpy())
        #vec=data['gt']
        y=torch.norm(data['gt'],2,1).type(torch.FloatTensor)
        sp.append(y.type(torch.FloatTensor).numpy())

    # Plot prediction and ground truth.
    print(np.shape((np.asarray(sp))))
    plt.figure()
    plt.subplot(211)
    plt.plot(np.asarray(pred)[:,:])
    plt.ylabel('Speed (m/s)')
    plt.title('Prediction')
    plt.subplot(212)
    plt.plot(np.asarray(sp)[:,0])
    plt.ylabel('ground truth speed')


def plot_bunch_confused(model, data_labels, ordered_Loader):
    dat_lab=[]
    for label in data_labels:
        dat_lab=dat_lab+label


    #Plot scatter of prediction and ground truth with labels.
    pred=[]
    sp=[]
    R=[]
    for i_batch, sample_batched in enumerate(ordered_Loader):
        data=sample_batched
        pred.append(model(Variable(data['imu'].float())).data[0].numpy())
        vec=data['gt']
        y=torch.norm(data['gt'],2,1).type(torch.FloatTensor)
        sp.append(y.type(torch.FloatTensor).numpy())
        R.append(np.array(data['range']))
    print(len(dat_lab))
    print(len(sp))
    pred=np.asarray(pred)
    sp=np.asarray(sp)    
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
            stat.append([sp[i,0],pred[i]])
            Rstat.append(R[i])
        elif dat_lab[i]==1:
            walk.append([sp[i,0],pred[i]])
            Rwalk.append(R[i])
        elif dat_lab[i]==2:
            stair.append([sp[i,0],pred[i]])
            Rstair.append(R[i])
        elif dat_lab[i]==3:
            esc.append([sp[i,0],pred[i]])
            Resc.append(R[i])
        else:
            ele.append([sp[i,0],pred[i]])
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



def plot_pred_speed_test(model):
    plt.figure()
    # Evaluate in unknown data to the network.
    nfolders=[]
    nfolders.append("/static/dataset-04/")
    Test=OdometryDataset("./../data/",nfolders,transform=ToTensor())
    test_Loader = DataLoader(Test, batch_size=1,shuffle=False, num_workers=1)

    pred=[]
    sp=[]
    t=[]
    for i_batch, sample_batched in enumerate(test_Loader):
        data=sample_batched
        pred.append(model(Variable(data['imu'].float())).data[0].cpu().numpy())
        vec=data['gt']
        #vertical=torch.norm(vec[:,[1]],2,1) 
        #vertical=vec[:,1]
        #horizontal=torch.norm(vec[:,[0,2]],2,1)  
        #y=torch.stack((vertical,horizontal),1)
        y=torch.norm(data['gt'],2,1).type(torch.FloatTensor)
        sp.append(y.type(torch.FloatTensor).numpy())
        t.append(data['time'])
    plt.subplot(211)
    plt.plot(np.asarray(pred))
    plt.ylabel('Predicted speed')
    plt.subplot(212)
    plt.plot(np.asarray(sp)[:,0])
    plt.ylabel('ground truth speed')

    fig = plt.figure(figsize=(6,6))
    plt.plot(np.asarray(sp)[:,0],np.asarray(pred)[:],'.', label='test data')
    plt.plot([0,2],[0,2],'k')
    plt.xlabel('gt (m/s)')
    plt.ylabel('prediction (m/s)')

    axes=plt.gca()

    axes.set_xlim((0.0,2))
    axes.set_ylim([0.0,2])
    axes.legend()


def train_model(model, T, epochs_num=10):

    #Configure data loaders and optimizer
    learning_rate = 1e-6
    loss_fn = torch.nn.MSELoss(size_average=False)
    
    index=np.arange(len(T))
    np.random.shuffle(index)
    train = index[1:int(np.floor(len(T)/10*9))]
    test = index[int(np.floor(len(T)/10*9)):-1]
    
    #Split training and validation.
    training_loader = DataLoader(T, batch_size=10, shuffle=False, num_workers=4, sampler=torch.utils.data.sampler.SubsetRandomSampler(list(train)))
    validation_loader = DataLoader(T, batch_size=10, shuffle=False, num_workers=4, sampler=torch.utils.data.sampler.SubsetRandomSampler(list(test)))
    #Create secondary loaders
    #single_train_Loader = DataLoader(T, batch_size=1,shuffle=False, num_workers=1, sampler=torch.utils.data.sampler.SubsetRandomSampler(list(train)))
    #single_validation_Loader = DataLoader(T, batch_size=1,shuffle=False, num_workers=1, sampler=torch.utils.data.sampler.SubsetRandomSampler(list(test)))

    #define optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    tls=[]
    vls=[]

    # Epochs
    for t in range(epochs_num):
        print("epochs {}".format(t))
        ti = time.time()
        acc_loss=0
        val_loss=0
        # Train
        for i_batch, sample_batched in enumerate(training_loader):
            # Sample data.
            data = sample_batched
            # Forward pass.
            y_pred = model(Variable(data['imu'].float()))

            # Sample corresponding ground truth.
            y_gt = torch.norm(data['gt'], 2, 1).type(torch.FloatTensor)
            y_gt_val =  Variable(y_gt)
            y_pred_val = y_pred.view(-1)
            # Compute and print loss.
            loss = loss_fn(y_pred_val, y_gt_val)

            # Save loss.
            # acc_loss +=np.sum(loss.data[0])
            acc_loss += loss.data
            # Zero the gradients before running the backward pass.
            model.zero_grad()
            # Backward pass.
            loss.backward()
            # Take optimizer step.
            optimizer.step()

        # Validation
        for i_batch, sample_batched in enumerate(validation_loader):
            # Sample data.
            data=sample_batched
            # Forward pass.   
            y_pred =model(Variable(data['imu'].float()))    

            y_gt =torch.norm(data['gt'],2,1).type(torch.FloatTensor)
            y = Variable(y_gt)
            loss = loss_fn(y_pred.view(-1), y)
            
            #val_loss +=np.sum(loss.data[0])
            val_loss += loss.data
        # Save loss and print status.
        tls.append(acc_loss/(len(T)*9/10))
        vls.append(val_loss/(len(T)/10))
        print("epoch\t", t)
        print("loss_train\t", tls[-1])
        print("loss_val\t", vls[-1])
        elapsed = time.time() - ti
        print("elapsed (sec)\t", elapsed)
        
    return tls, vls

def plot_traning(tls, vls):
    # Plot loss
    plt.figure()
    plt.plot(np.log(np.array(tls)),label = 'Training loss')
    plt.plot(np.log(np.array(vls)),label = 'Validation loss')

def load_dataset():

    folders, labs = get_data_folders_and_labs()

    data_labels = plot_data_labels(folders, labs)
    
    # Create dataset reader.
    T = OdometryDataset("../data", folders, transform=ToTensor())
    return T,  data_labels

def get_model_from_new_training(T, epochs_num=10, save_model=False):
    model = None
    tls, vls = None, None
    try:    
        model=vel_regressor(Nout=1, Nlinear=7440)
        if torch.cuda.is_available():
            model = model.to('cuda')
        if train_model:
            tls, vls = train_model(model, T, epochs_num=epochs_num)
        else:
            raise ValueError("What?")
    except Exception as ex:
        print("Exception occured: ", ex)
        print(traceback.format_exc())

    #save model
    if save_model:
        torch.save(model,'./full_new.pt')
    return model, tls, vls

def get_model_from_trained_model():
    model= torch.load('./full_new.pt', map_location=lambda storage, loc: storage)
    return model

def exam_model(model):
    pass

def plot_model_and_pred(model, T, data_labels):
    ordered_Loader = DataLoader(T, batch_size=1, shuffle=False, num_workers=1)

    # Load corresponding prediction and ground truth
    plot_pred_speed_ordered(model, ordered_Loader)

    plot_bunch_confused(model, data_labels, ordered_Loader)

    plot_pred_speed_test(model)


def run_main(load_model=False):

    T, data_labels = load_dataset()

    exam_dataset(T)
    plot_dataset(T)

    model = None
    try:    
        #load pretrained model or create new one.
        if load_model:
            model = get_model_from_trained_model()
        else:
            model, tls, vls = get_model_from_new_training(T, epochs_num=1)
            plot_traning(tls, vls)
    
    except Exception as ex:
        print("Exception occured: ", ex)
        print(traceback.format_exc())

    exam_model(model)
    plot_model_and_pred(model, T, data_labels)

    plt.show()


if __name__ == "__main__":

    run_main(load_model=False)





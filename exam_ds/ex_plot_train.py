import torch
# import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def _get_val_gt(data_gt):
    val = torch.norm(data_gt, 2, 1)
    val = val.type(torch.FloatTensor)
    return val.numpy()

def get_pred_gt_vals(model, data, using_cuda):
    # val_pred = Emdl.eval_pred(model, data['imu'])

    imu_data = data['imu']

    val_pred = model.eval_pred(imu_data, using_cuda)

    val_preds = val_pred.ravel().tolist()

    val_gt = _get_val_gt(data['gt'])
    val_gts = val_gt.ravel().tolist()
    
    return val_preds, val_gts


def plot_pred_speed_ordered(model, T, using_cuda=False, batch_size=1):
    ordered_Loader = DataLoader(T, batch_size=batch_size, shuffle=False, num_workers=0)

    # Load corresponding prediction and ground truth
    pred_sp=[]
    gt_sp=[]
    for i_batch, sample_batched in enumerate(ordered_Loader):
        data = sample_batched

        val_preds, val_gts = get_pred_gt_vals(model, data, using_cuda=using_cuda)

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


def plot_bunch_confused(model, T, data_labels, using_cuda=False, batch_size=1):
    ordered_Loader = DataLoader(T, batch_size=batch_size, shuffle=False, num_workers=0)

    dat_lab=[]
    for label in data_labels:
        dat_lab=dat_lab+label

    #Plot scatter of prediction and ground truth with labels.
    pred_sp=[]
    gt_sp=[]
    R=[]
    for i_batch, sample_batched in enumerate(ordered_Loader):
        data=sample_batched

        val_preds, val_gts = get_pred_gt_vals(model, data, using_cuda=using_cuda)

        pred_sp += val_preds
        gt_sp += val_gts

        if batch_size == 1:
            R.append(np.array(data['range']))
        else:
            for ndx in range(len(data['range'][0])):
                R.append(np.array([data['range'][0][ndx], data['range'][1][ndx]]))
    print(len(R))
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


class PlotTrainDs(object):
    @classmethod
    def plot_all(cls, model, T, data_labels, using_cuda=False, batch_size=1):
        print("prepare and plot pred_speed_ordered...")
        plot_pred_speed_ordered(model, T, using_cuda=using_cuda, batch_size=batch_size)
        print("prepare and plot plot_bunch_confused...")
        plot_bunch_confused(model, T, data_labels, using_cuda=using_cuda, batch_size=batch_size)

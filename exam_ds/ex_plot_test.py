import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

try:
    #import os
    #print(os.getcwd())
    os.chdir("exam_ds")
    #print(os.getcwd())
except:
    pass

from exam_ds.ex_dataset_loader import get_labs, find_plot_data_labels


def _get_val_gt_vec_speed(data_gt):
    val = torch.norm(data_gt, 2, 1)
    val = val.type(torch.FloatTensor)
    return val.numpy()

def get_pred_gt_vals(model, data, using_cuda):
    val_pred = model.eval_pred(data['imu'], using_cuda)

    #inputs: imu [bz, 6, 200]
    #outputs: speed [bz]
    # which we only get one speed no matter what direction it is
    val_preds = val_pred.ravel().tolist()

    # we only get one speed out of 3 (i.e. what norm(x, 2, 1) does, to make sqrt(vx**2 + vy**2 + vz**2))
    val_gt = _get_val_gt_vec_speed(data['gt'])
    val_gts = val_gt.ravel().tolist()
    
    return val_preds, val_gts

def get_test_dataset(test_folders=None):
    #Import python functions.
    try:
        from exam_ds.dataset import OdometryDataset
        from exam_ds.dataset import ToTensor
    except:
        from dataset import OdometryDataset
        from dataset import ToTensor
    
    # Evaluate in unknown data to the network.
    nfolders=[]
    if test_folders is None:
        nfolders += ["/static/dataset-04/"]
    else:
        nfolders += test_folders
    Test = OdometryDataset("./../data_ds/", nfolders, transform=ToTensor())   
    return Test 


def plot_model_pred_result(model, test_folders=None, using_cuda=False, batch_size=1, test=True):

    Test = get_test_dataset(test_folders)
    test_Loader = DataLoader(Test, batch_size=batch_size, shuffle=False, num_workers=0)

    pred_sp=[]
    gt_sp=[]
    t=[]
    for i_batch, sample_batched in enumerate(test_Loader):
        data=sample_batched

        val_preds, val_gts = get_pred_gt_vals(model, data, using_cuda)

        pred_sp += val_preds
        gt_sp += val_gts

        if batch_size == 1:
            t.append(data['time'])
        else:
            for ndx in range(len(data['time'])):
                t.append(data['time'][ndx])
        #t.append(data['time'])

    plt.figure()
    plt.subplot(211)
    plt.plot(pred_sp)
    plt.ylabel('Predicted speed')
    plt.subplot(212)
    plt.plot(gt_sp)
    plt.ylabel('ground truth speed')

    label = "test data"
    if not test:
        label = "train data"
    plt.figure(figsize=(6,6))
    if not test:
        plt.plot(np.asarray(gt_sp), np.asarray(pred_sp), '.', label=label)
    else:
        plt.plot(np.asarray(gt_sp), np.asarray(pred_sp), '*', label=label)
    plt.plot([0,2],[0,2],'k')
    plt.xlabel('gt (m/s)')
    plt.ylabel('prediction (m/s)')

    axes=plt.gca()

    axes.set_xlim((0.0,2))
    axes.set_ylim([0.0,2])
    axes.legend()


def plot_model_pred_categorized_result(model, test_folders=None, data_labels=None, using_cuda=False, batch_size=1, test=True):

    labs = get_labs()

    data_labels = find_plot_data_labels(test_folders, labs, using_cuda=using_cuda)

    Test = get_test_dataset(test_folders)
    #test_Loader = DataLoader(Test, batch_size=batch_size,
    #                         shuffle=False, num_workers=0)
    ordered_Loader = DataLoader(
        Test, batch_size=batch_size, shuffle=False, num_workers=0)

    dat_lab = []
    for label in data_labels:
        dat_lab = dat_lab + label

    #Plot scatter of prediction and ground truth with labels.
    pred_sp = []
    gt_sp = []
    R = []
    for i_batch, sample_batched in enumerate(ordered_Loader):
        data = sample_batched

        val_preds, val_gts = get_pred_gt_vals(
            model, data, using_cuda=using_cuda)

        pred_sp += val_preds
        gt_sp += val_gts

        if batch_size == 1:
            R.append(np.array(data['range']))
        else:
            for ndx in range(len(data['range'][0])):
                R.append(
                    np.array([data['range'][0][ndx], data['range'][1][ndx]]))
    print(len(R))
    print(len(dat_lab))
    print(len(gt_sp))
    pred = np.asarray(pred_sp)
    sp = np.asarray(gt_sp)
    stat = []
    stair = []
    walk = []
    esc = []
    ele = []

    Rstat = []
    Rstair = []
    Rwalk = []
    Resc = []
    Rele = []

    #Separte by label
    for i in range(0, len(dat_lab)):
        if dat_lab[i] == 0:
            stat.append([sp[i], pred[i]])
            Rstat.append(R[i])
        elif dat_lab[i] == 1:
            walk.append([sp[i], pred[i]])
            Rwalk.append(R[i])
        elif dat_lab[i] == 2:
            stair.append([sp[i], pred[i]])
            Rstair.append(R[i])
        elif dat_lab[i] == 3:
            esc.append([sp[i], pred[i]])
            Resc.append(R[i])
        else:
            ele.append([sp[i], pred[i]])
            Rele.append(R[i])

    msize = 3
    plt.figure(figsize=(8, 8))
    #Scatter plot.
    test = np.array(stat)
    plt.plot(test[:, 0], test[:, 1], 'r*', label='static', markersize=msize)
    test = np.array(stair)
    plt.plot(test[:, 0], test[:, 1], 'g*', label='stair', markersize=msize)
    test = np.array(walk)
    plt.plot(test[:, 0], test[:, 1], 'b*', label='walk', markersize=msize)
    test = np.array(esc)
    plt.plot(test[:, 0], test[:, 1], 'k*', label='escalator', markersize=msize)
    test = np.array(ele)
    plt.plot(test[:, 0], test[:, 1], 'y*', label='elevator', markersize=msize)

    plt.plot([0, 1.5], [0, 1.5], 'k')
    plt.xlabel('gt (m/s)')
    plt.ylabel('prediction (m/s)')

    #plot histograms by label
    axes = plt.gca()
    axes.set_xlim((0.0, 1.5))
    axes.set_ylim([0.0, 1.5])
    axes.legend()


class PlotTestDs(object):
    @classmethod
    def plot_pred_result(cls, model, test_folders, using_cuda=False, batch_size=1, test=True):
        plot_model_pred_result(model, test_folders=test_folders, using_cuda=using_cuda, batch_size=batch_size, test=test)

    @classmethod
    def plot_pred_result_categrozied(cls, model, test_folders, using_cuda=False, batch_size=1, test=True):
        plot_model_pred_categorized_result(model, test_folders=test_folders, data_labels=None, using_cuda=using_cuda, batch_size=batch_size, test=test)

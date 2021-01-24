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

def _get_val_gt(data_gt):
    val = torch.norm(data_gt, 2, 1)
    val = val.type(torch.FloatTensor)
    return val.numpy()

def get_pred_gt_vals(model, data, using_cuda):
    val_pred = model.eval_pred(data['imu'], using_cuda)
    val_preds = val_pred.ravel().tolist()

    val_gt = _get_val_gt(data['gt'])
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
    plt.plot(np.asarray(gt_sp), np.asarray(pred_sp),'.', label=label)
    plt.plot([0,2],[0,2],'k')
    plt.xlabel('gt (m/s)')
    plt.ylabel('prediction (m/s)')

    axes=plt.gca()

    axes.set_xlim((0.0,2))
    axes.set_ylim([0.0,2])
    axes.legend()


class PlotTrainDs(object):
    @classmethod
    def plot_pred_result(cls, model, test_folders, using_cuda=False, batch_size=1, test=True):
        plot_model_pred_result(model, test_folders=test_folders, using_cuda=using_cuda, batch_size=batch_size, test=test)

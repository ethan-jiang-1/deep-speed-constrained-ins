import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

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
    from exam_ds.ex_model import ExamModelDs as Emdl
except:
    from dataset import OdometryDataset
    from dataset import ToTensor
    from ex_model import ExamModelDs as Emdl



def _get_val_gt(data_gt):
    val = torch.norm(data_gt, 2, 1)
    val = val.type(torch.FloatTensor)
    return val.numpy()

def get_pred_gt_vals(model, data):
    val_pred = Emdl.eval_pred(model, data['imu'])
    val_preds = val_pred.ravel().tolist()

    val_gt = _get_val_gt(data['gt'])
    val_gts = val_gt.ravel().tolist()
    
    return val_preds, val_gts


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


class PlotTrainDs(object):
    @classmethod
    def plot_all(cls, model, data_labels):
        plot_model_on_test_dataset(model)
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


import os
import numpy as np
# import traceback
import matplotlib.pyplot as plt

# dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from exam_ds.ex_dataset_loader import DataLoaderDs as Dsl
from exam_ds.ex_plot_train import PlotTrainDs as Ptn
from exam_ds.ex_plot_test import PlotTestDs as Ptt

def plot_traning(tls, vls):

    # Plot loss
    plt.figure()
    plt.plot(np.log(np.array(tls)),label='Training loss')
    plt.plot(np.log(np.array(vls)),label='Validation loss')


def select_model(model_name):
    #Import python functions.
    print("model selected {}".format(model_name))

    Emdl = None
    if model_name == "conv1d":
        from exam_ds.ex_model_ds_conv1d import ExamModelDs as Emdl
    elif model_name == "lstm":
        from exam_ds.ex_model_ds_lstm import ExamModelDs as Emdl
    elif model_name == "resnet18":
        from exam_ds.ex_model_ds_resnet18 import ExamModelDs as Emdl
    elif model_name == "org":
        from exam_ds.ex_model_ds_org import ExamModelDs as Emdl
    elif model_name == "org_da":
        from exam_ds.ex_model_ds_org_da import ExamModelDs as Emdl
    elif model_name == "org_db":
        from exam_ds.ex_model_ds_org_db import ExamModelDs as Emdl
    elif model_name == "convdp":
        from exam_ds.ex_model_ds_convdp import ExamModelDs as Emdl

    if Emdl is None:
        raise ValueError("no_model_for{}".format(model_name))
    return Emdl
    

def run_model(model_name="conv1d", load_model=False, plt_show=True, early_stop=False):
    T, data_labels = Dsl.load_dataset()

    Dsl.plot_dataset(T)
    Dsl.plot_sub_dataset(T)
    # Dsl.plot_dataset_internals(T)
    if plt_show:
        plt.show()


    Emdl = select_model(model_name)
   
    #load pretrained model or create new one.
    if load_model:
        model = Emdl.get_model_from_trained_model()
    else:
        model = Emdl.get_empty_model()
        Emdl.exam_model(model)
        model, tls, vls = Emdl.keep_train_model(model, T, epochs_num=1, early_stop=early_stop)
        if tls is not None:
            plot_traning(tls, vls)
    

    Emdl.exam_model(model)

    if model is not None and hasattr(model, "eval_pred"):
        # plot on trained testset (tain/val)
        Ptn.plot_all(model, T, data_labels, batch_size=4)

        # test on test (not tranined)
        Ptt.plot_pred_result(model, test_folders=[
                             "/static/dataset-04/"], batch_size=4)

        if plt_show:
            plt.show()


def check_model(model_name="conv1d"):
    Emdl = select_model(model_name)

    model = Emdl.get_empty_model()
    print(model)


if __name__ == "__main__":
    #ok
    #model_name="org"
    #model_name="org_da"
    #model_name="org_db"
    #model_name="conv1d"
    model_name = "convdp"
    
    #failed
    #model_name="lstm" -- no-learning
    #model_name = "resnet18"  -- tensor mis-match

    load_model = False
    inspect_model_only = False

    if inspect_model_only:
        check_model(model_name=model_name)
    else:
        run_model(model_name=model_name, 
                load_model=False)
    





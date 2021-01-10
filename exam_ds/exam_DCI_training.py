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
import traceback
import matplotlib.pyplot as plt

import sys

from torch._C import Value
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
    from exam_ds.ex_dataset_loader import DataLoaderDs as Dsl
    from exam_ds.ex_plot_train import PlotTrainDs as Ptn
    from exam_ds.ex_plot_test import PlotTrainDs as Ptt
except:
    from ex_dataset_loader import DataLoaderDs as Dsl
    from ex_plot_train import PlotTrainDs as Ptn
    from ex_plot_test import PlotTrainDs as Ptt


def plot_traning(tls, vls):

    # Plot loss
    plt.figure()
    plt.plot(np.log(np.array(tls)),label = 'Training loss')
    plt.plot(np.log(np.array(vls)),label = 'Validation loss')


def select_model(model_name):
    #Import python functions.
    if model_name in ["conv1d", "lstm"]:
        print("model selected {}".format(model_name))

        if model_name == "conv1d":
            try:
                from exam_ds.ex_model_ds_conv1d import ExamModelDs as Emdl
            except:
                from ex_model_ds_conv1d import ExamModelDs as Emdl
            print(Emdl)
            return Emdl        
        else:
            try:
                from exam_ds.ex_model_ds_lstm import ExamModelDs as Emdl
            except:
                from ex_model_ds_lstm import ExamModelDs as Emdl
            print(Emdl)
            return Emdl
    raise ValueError("no_model_for{}".format(model_name))
    

def run_main(model_name="lstm", load_model=False):
    T, data_labels = Dsl.load_dataset()

    Emdl = select_model(model_name)

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

    if model is not None and hasattr(model, "eval_pred"):
        # plot on trained testset (tain/val)
        Ptn.plot_all(model, T, data_labels)

        # test on test (not tranined)
        Ptt.plot_all(model, test_folders=["/static/dataset-04/"])

        plt.show()
    else:
        raise ValueError("invalid_model")


if __name__ == "__main__":
    #model_name="lstm"
    model_name="conv1d"

    run_main(model_name=model_name, load_model=False)

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


def select_model():
    #Import python functions.
    try:
        from exam_ds.ex_model_ds_conv1d import ExamModelDs as Emdl
    except:
        from ex_model_ds_conv1d import ExamModelDs as Emdl
    return Emdl
    

def run_main(load_model=False):
    T, data_labels = Dsl.load_dataset()

    Emdl = select_model()

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

    # plot on trained testset (tain/val)
    Ptn.plot_all(model, T, data_labels)

    # test on test (not tranined)
    Ptt.plot_all(model, test_folders=["/static/dataset-04/"])

    plt.show()


if __name__ == "__main__":
    run_main(load_model=False)





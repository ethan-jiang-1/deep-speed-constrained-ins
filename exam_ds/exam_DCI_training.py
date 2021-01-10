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
    from exam_ds.dataset import OdometryDataset
    from exam_ds.dataset import ToTensor
    from exam_ds.ex_model import ExamModelDs as Emdl
    from exam_ds.ex_dataset_loader import DataLoaderDs as Dsl
    from exam_ds.ex_plot_train import PlotTrainDs as Ptn
    from exam_ds.ex_plot_test import PlotTrainDs as Ptt
except:
    from dataset import OdometryDataset
    from dataset import ToTensor
    from ex_model import ExamModelDs as Emdl
    from ex_dataset_loader import DataLoaderDs as Dsl
    from ex_plot_train import PlotTrainDs as Ptn
    from ex_plot_test import PlotTrainDs as Ptt


def plot_traning(tls, vls):

    # Plot loss
    plt.figure()
    plt.plot(np.log(np.array(tls)),label = 'Training loss')
    plt.plot(np.log(np.array(vls)),label = 'Validation loss')


def run_main(load_model=False):
    T, data_labels = Dsl.load_dataset()

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

    Ptn.plot_all(model, T, data_labels)
    Ptt.plot_all(model, data_labels)

    plt.show()


if __name__ == "__main__":
    run_main(load_model=False)





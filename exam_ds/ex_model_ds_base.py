# from numpy.lib.npyio import save
import torch
from torch.autograd import Variable
import traceback
from torchinfo import summary
import os


def _inspect_model(model, batch_size=10, enforced=False):
    if model is not None and not hasattr(model, "model_examed") or enforced:
        try:
            #if not torch.cuda.is_available():
            # 6 channel, 200 samples/channel,  this does not include batch size
            input_size = (6, 200)
            batch_input_size = (batch_size, *input_size)
            print("batch_input_shape", batch_input_size)
            summary(model, batch_input_size, verbose=2, col_names=["input_size",
                                                                   "output_size",
                                                                   "num_params",
                                                                   "kernel_size",
                                                                   "mult_adds"])
        except Exception as ex:
            print("Exception occured ", ex)
            print(model)
        model.model_examed = True


#def get_val_gt_float(data_gt):
#    val = torch.norm(data_gt, 2, 1)
#    val = val.type(torch.FloatTensor)
#    return val

class ExamModelBase(object):
    # subclass need implement followings
    @classmethod
    def ex_get_regressor_klass(cls):
        return None
    
    @classmethod
    def ex_get_train_func(cls):
        return None
    
    @classmethod
    def ex_get_saved_model_pathnames(cls):
        return "full_new.pt", "/content/drive/MyDrive/full_new.pt"

    #
    @classmethod
    def set_saved_model_path_name(cls, saved_model_path_name):
        cls.saved_model_pathname_local = saved_model_path_name
        cls.saved_model_pathname_gdrive = "/content/drive/MyDrive/" + saved_model_path_name

    @classmethod
    def get_saved_model_path_name_local(cls):
        pnl, png = cls.ex_get_saved_model_pathnames()
        return pnl

    @classmethod
    def get_saved_model_path_name_gdrive(cls):
        pnl, png = cls.ex_get_saved_model_pathnames()
        return png

    @classmethod
    def exam_model(cls, model, batch_size=10, enforced=False):
        return _inspect_model(model, batch_size=batch_size, enforced=enforced)

    @classmethod
    def get_empty_model(cls):
        modelKlass = cls.ex_get_regressor_klass()
        model = modelKlass()
        #if torch.cuda.is_available():
        #    model.to('cuda')
        # cls.exam_model(model)
        return model 
    
    @classmethod
    def keep_train_model(cls, model, T, epochs_num=20, save_model=False, batch_size=10, using_cuda=False, early_stop=False):
        global g_using_cuda
        g_using_cuda = using_cuda

        if g_using_cuda:
            model.cuda()

        tls, vls = None, None
        try:
            tain_func = cls.ex_get_train_func()
            if tain_func:
                tls, vls = tain_func(model, T, epochs_num=epochs_num, batch_size=batch_size, early_stop=early_stop)
            else:
                raise ValueError("What?")
        except Exception as ex:
            print("Exception occured: ", ex)
            print(traceback.format_exc())

        #save model
        if model is not None:
            if save_model:
                cls.save_trained_model(model)
            cls.attach_eval_pred(model)
        return model, tls, vls

    @classmethod
    def get_model_from_new_training(cls, T, epochs_num=20, save_model=False, batch_size=10, using_cuda=False):
        global g_using_cuda
        g_using_cuda = using_cuda
        model = cls.get_empty_model()
        return cls.keep_train_model(model, T, epochs_num=epochs_num, save_model=save_model, batch_size=batch_size, using_cuda=using_cuda)

    @classmethod
    def save_trained_model(cls, model):
        try:
            model_path = cls.get_saved_model_path_name_local()
            cls.dettach_eval_pred(model)
            torch.save(model, model_path)
            print("model saved at", model_path)
        except Exception as ex:
            print("exception ", ex)
        try:
            model_path = cls.get_saved_model_path_name_gdrive()
            cls.dettach_eval_pred(model)
            torch.save(model, model_path)
            print("model saved at", model_path)
        except Exception as ex:
            print("exception ", ex)
    
    @classmethod
    def get_model_from_trained_model(cls, model_path=None):
        model_path = cls.get_saved_model_path_name_local()
        if os.path.isfile(model_path):
            model= torch.load(model_path, map_location=lambda storage, loc: storage)
            cls.exam_model(model)
            print("load pre-trained model from", model_path)
            return model
        model_path = cls.get_saved_model_path_name_gdrive()
        if os.path.isfile(model_path):
            model = torch.load(model_path, map_location=lambda storage, loc: storage)
            cls.exam_model(model)
            print("load pre-trained model from", model_path)
            return model

        raise ValueError("no_saved_model_{}".format(model_path))

    @classmethod
    def get_pred_model_from_trained_model(cls, model, force_to_cpu=False, using_cuda=False):
        model_path = "tmp_state"
        
        # save no matter where model is come from cpu or gpu
        cls.dettach_eval_pred(model)
        torch.save(model.state_dict(), model_path)
        
        # Load model in cpu
        if force_to_cpu:
            device = torch.device('cpu')
        else:
            if using_cuda:    
                has_cuda = torch.cuda.is_available()
                device = torch.device('cuda' if has_cuda else 'cpu')
            else:
                device = torch.device('cpu')

        pred_model = cls.get_empty_model()
        pred_model.load_state_dict(torch.load(model_path, map_location=device))
        if force_to_cpu:
            pred_model.cpu()

        cls.attach_eval_pred(pred_model)
        return pred_model

    @classmethod
    def eval_pred(cls, model, features, using_cuda):
        #model(Variable(data['imu'].float())).data[0].numpy() 
        var = Variable(features.float())
        if using_cuda:
            var = var.cuda()

        #cls.attach_eval_pred(model)
        result = model(var)
        #if using_cuda:
        result = result.cpu()

        return result.data.numpy()

    @classmethod
    def attach_eval_pred(cls, model):
        def attached_eval_pred(features, using_cuda):
            return cls.eval_pred(model, features, using_cuda)
        if not hasattr(model, "eval_pred"):
            model.eval_pred = attached_eval_pred

    @classmethod
    def dettach_eval_pred(cls, model):
        if hasattr(model, "eval_pred"):
            del model.eval_pred

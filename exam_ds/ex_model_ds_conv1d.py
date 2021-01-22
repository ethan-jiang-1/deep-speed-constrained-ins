try:
    from exam_ds.ex_model_ds_base import ExamModelBase
    from exam_ds.vel_regressor_conv1d import VelRegressorConv1d as vel_regressor
    from exam_ds.vel_trainner_dp import train_model
except:
    from ex_model_ds_base import ExamModelBase
    from vel_regressor_conv1d import VelRegressorConv1d as vel_regressor
    from vel_trainner_dp import train_model


class ExamModelDs(ExamModelBase):
    # subclass need implement followings
    @classmethod
    def ex_get_regressor_klass(cls):
        return vel_regressor

    @classmethod
    def ex_get_train_func(cls):
        return train_model

    @classmethod
    def ex_get_saved_model_pathnames(cls):
        return "md_conv1d.pt", "/content/drive/MyDrive/md_conv1d.pt"

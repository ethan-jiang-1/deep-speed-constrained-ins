try:
    from exam_ds.ex_model_ds_base import ExamModelBase
    from exam_ds.vel_regressor_lstm import VelRegressorLstm as vel_regressor
    from exam_ds.vel_trainner import train_model
except:
    from ex_model_ds_base import ExamModelBase
    from vel_regressor_lstm import VelRegressorLstm as vel_regressor
    from vel_trainner import train_model


class ExamModelDs(ExamModelBase):
    vel_regressor = vel_regressor
    train_model = train_model

    saved_model_pathname_local = "md_lstm.pt"
    saved_model_pathname_gdrive = "/content/drive/MyDrive/md_lstm.pt"

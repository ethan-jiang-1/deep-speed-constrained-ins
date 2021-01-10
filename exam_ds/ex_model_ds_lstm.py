import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import traceback
from torchsummary import summary

# Model
class vel_regressor_lstm(torch.nn.Module):
    def __init__(self, Nin=200, Nout=1, batch_size=10, device=None,
                 lstm_size=200, lstm_layers=6, dropout=0):
        """
        Simple LSTM network
        Input: torch array [batch x frames x input_size]
        Output: torch array [batch x frames x out_size]

        :param input_size: num. channels in input
        :param out_size: num. channels in output
        :param batch_size:
        :param device: torch device
        :param lstm_size: number of LSTM units per layer
        :param lstm_layers: number of LSTM layers
        :param dropout: dropout probability of LSTM (@ref https://pytorch.org/docs/stable/nn.html#lstm)
        """
        super(vel_regressor_lstm, self).__init__()
        self.input_size = Nin
        self.lstm_size = lstm_size
        self.output_size = Nout
        self.num_layers = lstm_layers
        self.batch_size = batch_size
        self.device = device

        self.lstm = torch.nn.LSTM(self.input_size, self.lstm_size, self.num_layers, batch_first=True, dropout=dropout)
        self.linear1 = torch.nn.Linear(self.lstm_size, self.output_size * 60)
        self.linear2 = torch.nn.Linear(self.output_size * 60, self.output_size)
        #self.hidden = self.init_weights()

    def forward(self, input, hidden=None):
        # input tensor shape (10, 6, 200) in batch_mode(10)
        output, _ = self.lstm(input) #, self.init_weights())
        output = self.linear1(output)
        output = self.linear2(output)
        return output

    def init_weights(self):
        h0 = torch.zeros(self.num_layers, self.batch_size, self.lstm_size)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.lstm_size)
        if self.device is not None:
            h0 = h0.to(self.device)
            c0 = c0.to(self.device)
        return Variable(h0), Variable(c0)


def inspect_model(model, batch_size=10):
    if not hasattr(model, "model_examed"):
        print(model)
        if not torch.cuda.is_available():
            # 6 channel, 200 samples/channel,  this does not include batch size
            input_size = (6, 200)
            batch_input_size = (batch_size, *input_size)
            print("batch_input_shape", batch_input_size)
            summary(model, (6, 200), batch_size=batch_size, device="cpu")  
        model.model_examed = True


model_activation = {}
def get_activation(name):
    def hook(model, input, output):
        global model_activation
        model_activation[name] = output.detach()
    return hook


#def get_val_gt_float(data_gt):
#    val = torch.norm(data_gt, 2, 1)
#    val = val.type(torch.FloatTensor)
#    return val


loss_fn = torch.nn.MSELoss(reduction='sum')
def compute_loss(model, data):
    global model_activation
    #loss_fn = torch.nn.MSELoss(reduction='sum')

    x_features = Variable(data['imu'].float())
    # shape of x_features [10, 6, 200]
    y_pred = model(x_features)
    # shape of y_pred [10, 1]
    y_pred_val = y_pred.view(-1)
    # [10]

    # Sample corresponding ground truth.
    # shape of y_gt [10, 3]  
    y_gt = torch.norm(data['gt'], 2, 1).type(torch.FloatTensor)
    # [10, 1]
    y_gt_val =  Variable(y_gt)
    # [10]

    # Compute and print loss.
    loss = loss_fn(y_pred_val, y_gt_val)
    return loss

def train_model(model, T, epochs_num=20, batch_size=10):
    #model.model3.register_forward_hook(get_activation('model3'))

    #Configure data loaders and optimizer
    learning_rate = 1e-6

    index=np.arange(len(T))
    np.random.shuffle(index)
    train = index[1:int(np.floor(len(T)/10*9))]
    test = index[int(np.floor(len(T)/10*9)):-1]
    
    #Split training and validation.
    training_loader = DataLoader(T, batch_size=batch_size, shuffle=False, num_workers=4, sampler=torch.utils.data.sampler.SubsetRandomSampler(list(train)))
    validation_loader = DataLoader(T, batch_size=batch_size, shuffle=False, num_workers=4, sampler=torch.utils.data.sampler.SubsetRandomSampler(list(test)))
    #Create secondary loaders
    #single_train_Loader = DataLoader(T, batch_size=1,shuffle=False, num_workers=1, sampler=torch.utils.data.sampler.SubsetRandomSampler(list(train)))
    #single_validation_Loader = DataLoader(T, batch_size=1,shuffle=False, num_workers=1, sampler=torch.utils.data.sampler.SubsetRandomSampler(list(test)))

    #define optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    tls=[]
    vls=[]

    # Epochs
    for t in range(epochs_num):
        print("epochs {}...".format(t))
        ti = time.time()
        acc_loss=0
        val_loss=0
        # Train
        for i_batch, sample_batched in enumerate(training_loader):
            # Sample data.
            data = sample_batched
            loss = compute_loss(model, data)
            acc_loss += loss.data

            # Zero the gradients before running the backward pass.
            model.zero_grad()
            # Backward pass.
            loss.backward()
            # Take optimizer step.
            optimizer.step()

        # Validation
        for i_batch, sample_batched in enumerate(validation_loader):
            # Sample data.
            data=sample_batched
            loss = compute_loss(model, data)
            val_loss += loss.data

        # Save loss and print status.
        tls.append(acc_loss/(len(T)*9/10))
        vls.append(val_loss/(len(T)/10))
        elapsed = time.time() - ti        

        print("epochs {} elapsed: {:.2f}(sec)\t\tloss_train: {:.4f}\tloss_val: {:.4f}".format(t, elapsed, tls[-1], vls[-1]))
        
    return tls, vls

class ExamModelDs(object):
    @classmethod
    def exam_model(cls, model):
        inspect_model(model)

    @classmethod
    def get_model_from_new_training(cls, T, epochs_num=20, save_model=False, batch_size=10):
        model = None
        tls, vls = None, None
        try:    
            model = vel_regressor_lstm()
            #if torch.cuda.is_available():
            #    model.to('cuda')
            cls.exam_model(model)

            #model = model.to(dev)
            if train_model:
                tls, vls = train_model(model, T, epochs_num=epochs_num, batch_size=batch_size)
            else:
                raise ValueError("What?")
        except Exception as ex:
            print("Exception occured: ", ex)
            print(traceback.format_exc())

        #save model
        if model is not None:
            if save_model:
                torch.save(model,'./full_new.pt')
            cls.attach_eval_pred(model)
        return model, tls, vls

    @classmethod
    def get_model_from_trained_model(cls):
        model= torch.load('./full_new.pt', map_location=lambda storage, loc: storage)
        cls.exam_model(model)
        cls.attach_eval_pred(model)
        return model

    @classmethod
    def eval_pred(cls, model, features):
        #model(Variable(data['imu'].float())).data[0].numpy() 
        var = Variable(features.float())
        result = model(var)
        return result.data[0].numpy()

    @classmethod
    def attach_eval_pred(cls, model):
        def attached_eval_pred(features):
            return cls.eval_pred(model, features)
        if not hasattr(model, "eval_pred"):
            model.eval_pred = attached_eval_pred
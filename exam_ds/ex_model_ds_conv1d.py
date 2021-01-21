import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import traceback
from torchinfo import summary

# Model
g_using_cuda = False

class vel_regressor_conv1d(torch.nn.Module):
    def __init__(self, Nin=6, Nout=1, Nlinear=5760):
        super(vel_regressor_conv1d, self).__init__()

        # Convolutional layers
        self.model1 = torch.nn.Sequential(
        torch.nn.Conv1d(Nin, 180, kernel_size=1, stride=1, groups=Nin),
        torch.nn.ReLU(),
        torch.nn.Conv1d(180, 180, kernel_size=2, stride=1, groups=Nin),
        torch.nn.ReLU(),
        torch.nn.Conv1d(180, 180, kernel_size=3, stride=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool1d(10, stride=6),
        )
        
        # Fully connected layers
        self.model2 = torch.nn.Sequential(
        torch.nn.Linear(Nlinear, 10*40),
        torch.nn.ReLU(),
        torch.nn.Linear(10*40, 100),
        torch.nn.ReLU())
        
        # Last FC
        self.model3 = torch.nn.Sequential(
        torch.nn.Linear(100, 3)
        )
        
    # Forward pass
    def forward(self, x):
        # x tensor shape (10, 6, 200) in batch_mode(10)
        x = self.model1(x)
        x = x.view(x.size(0), -1)
        x = self.model2(x)
        x = self.model3(x)
        y = torch.norm(x, dim=1)
        return y


def inspect_model(model, batch_size=10):
    if not hasattr(model, "model_examed"):
        print(model)        
        if not torch.cuda.is_available():
            # 6 channel, 200 samples/channel,  this does not include batch size
            input_size = (6, 200)
            batch_input_size = (batch_size, *input_size)
            print("batch_input_shape", batch_input_size)
            summary(model, batch_input_size, verbose=2)
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

    if g_using_cuda:
        x_features = x_features.cuda()

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
    if g_using_cuda:
        y_gt_val = y_gt_val.cuda()

    # Compute and print loss.
    loss = loss_fn(y_pred_val, y_gt_val)
    return loss

def train_model(model, T, epochs_num=10, batch_size=10):
    #model.model3.register_forward_hook(get_activation('model3'))

    #Configure data loaders and optimizer
    learning_rate = 1e-6

    all_ndxs=np.arange(len(T))
    np.random.shuffle(all_ndxs)
    train_ndxs = all_ndxs[1:int(np.floor(len(T)/10*9))]
    test_ndxs = all_ndxs[int(np.floor(len(T)/10*9)):-1]
    
    num_workers = 4
    if g_using_cuda:
        num_workers = 0

    #Split training and validation.
    training_loader = DataLoader(T, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=torch.utils.data.sampler.SubsetRandomSampler(list(train_ndxs)))
    validation_loader = DataLoader(T, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=torch.utils.data.sampler.SubsetRandomSampler(list(test_ndxs)))

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
        return inspect_model(model)

    @classmethod
    def get_empty_model(cls):
        model = vel_regressor_conv1d()
        #if torch.cuda.is_available():
        #    model.to('cuda')
        cls.exam_model(model)
        return model 

    @classmethod
    def get_model_from_new_training(cls, T, epochs_num=20, save_model=False, batch_size=10, using_cuda=False):
        global g_using_cuda
        g_using_cuda = using_cuda
        model = None
        tls, vls = None, None
        try:    
            model = cls.get_empty_model()
            if g_using_cuda:
                model.cuda()

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
        if g_using_cuda:
            var = var.cuda()

        result = model(var)
        if g_using_cuda:
            result = result.cpu()

        return result.data[0].numpy()

    @classmethod
    def attach_eval_pred(cls, model):
        def attached_eval_pred(features):
            return cls.eval_pred(model, features)
        if not hasattr(model, "eval_pred"):
            model.eval_pred = attached_eval_pred

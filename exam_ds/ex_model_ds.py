import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import traceback
from torchsummary import summary

# Model
class vel_regressor(torch.nn.Module):
    def __init__(self, Nin=6, Nout=1, Nlinear=112*30):
        super(vel_regressor, self).__init__()

        # Convolutional layers
        self.model1 = torch.nn.Sequential(
        torch.nn.Conv1d(Nin,180, kernel_size=1, stride=1, groups=Nin),
        torch.nn.ReLU(),
        torch.nn.Conv1d(180, 90, kernel_size=2, stride=1, groups=Nin),
        torch.nn.ReLU(),
        torch.nn.Conv1d(90,  60, kernel_size=3, stride=1),
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
        x = self.model1(x)
        x = x.view(x.size(0), -1)
        x = self.model2(x)
        x = self.model3(x)
        y = torch.norm(x, dim=1)
        return y


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


loss_fn_triplet = torch.nn.TripletMarginWithDistanceLoss(distance_function=torch.nn.PairwiseDistance())
loss_fn_MSELoss = torch.nn.MSELoss(reduction='sum')
def compute_loss(model, data):
    global model_activation
    if "model3" in model_activation:
        del model_activation["model3"]
    
    #loss_fn = torch.nn.MSELoss(reduction='sum')

    x_features = Variable(data['imu'].float())
    # shape of x_features [10, 6, 200]
    y_pred = model(x_features)

    if "model3" in model_activation:
        anchor = model_activation["model3"]
        positive = data['gt']
        negative = data['gt'].clone().detach()
        for i in range(len(negative)):
            for j in range(3):
                negative[i][j] = -negative[i][j]

        loss = loss_fn_triplet(anchor, positive, negative)
        return loss
    else:
        # shape of y_pred [10, 1]
        y_pred_val = y_pred.view(-1)
        # [10]

        # shape of y_gt [10, 3]  
        y_gt = torch.norm(data['gt'], 2, 1).type(torch.FloatTensor)
        # [10, 1]
        y_gt_val =  Variable(y_gt)
        # [10]

        # Compute and print loss.
        loss = loss_fn_MSELoss(y_pred_val, y_gt_val)
        return loss

def train_model(model, T, epochs_num=10):
    model.model3.register_forward_hook(get_activation('model3'))

    #Configure data loaders and optimizer
    learning_rate = 1e-6

    index=np.arange(len(T))
    np.random.shuffle(index)
    train = index[1:int(np.floor(len(T)/10*9))]
    test = index[int(np.floor(len(T)/10*9)):-1]
    
    #Split training and validation.
    training_loader = DataLoader(T, batch_size=10, shuffle=False, num_workers=4, sampler=torch.utils.data.sampler.SubsetRandomSampler(list(train)))
    validation_loader = DataLoader(T, batch_size=10, shuffle=False, num_workers=4, sampler=torch.utils.data.sampler.SubsetRandomSampler(list(test)))

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
        if not hasattr(model, "model_examed"):
            if not torch.cuda.is_available():
                print(model)
                summary(model, (6, 200))
            model.model_examed = True

    @classmethod
    def get_model_from_new_training(cls, T, epochs_num=10, save_model=False):
        model = None
        tls, vls = None, None
        try:    
            model=vel_regressor(Nout=1, Nlinear=1920) # 1860) # Nlinear=7440)
            #if torch.cuda.is_available():
            #    model.to('cuda')
            cls.exam_model(model)

            #model = model.to(dev)
            if train_model:
                tls, vls = train_model(model, T, epochs_num=epochs_num)
            else:
                raise ValueError("What?")
        except Exception as ex:
            print("Exception occured: ", ex)
            print(traceback.format_exc())

        #save model
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
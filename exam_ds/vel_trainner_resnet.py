import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import os

try:
    from exam_ds.ex_early_stopping import EarlyStopping
except:
    from ex_early_stopping import EarlyStopping

g_using_cuda = None
def get_using_cuda():
    global g_using_cuda
    if g_using_cuda is not None:
        return g_using_cuda

    g_using_cuda = False
    if "USING_CUDA" in os.environ:
        if os.os.environ["USING_CUDA"] == "True":
            if torch.cuda.is_available():
                g_using_cuda = True
    else:
        if torch.cuda.is_available():
            g_using_cuda = True
    return g_using_cuda


model_activation = {}


def get_activation(name):
    def hook(model, input, output):
        global model_activation
        model_activation[name] = output.detach()
    return hook


loss_fn = torch.nn.MSELoss(reduction='sum')


def compute_loss(model, data):
    global model_activation
    #loss_fn = torch.nn.MSELoss(reduction='sum')

    x_features = Variable(data['imu'].float())

    if get_using_cuda():
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
    y_gt_val = Variable(y_gt)
    # [10]
    if get_using_cuda():
        y_gt_val = y_gt_val.cuda()

    # Compute and print loss.
    loss = loss_fn(y_pred_val, y_gt_val)
    return loss


def train_model(model, T, epochs_num=10, batch_size=10, early_stop=False):
    #model.model3.register_forward_hook(get_activation('model3'))

    #Configure data loaders and optimizer
    learning_rate = 1e-6

    all_ndxs = np.arange(len(T))
    np.random.shuffle(all_ndxs)
    train_ndxs = all_ndxs[1:int(np.floor(len(T) / 10 * 9))]
    test_ndxs = all_ndxs[int(np.floor(len(T) / 10 * 9)):-1]

    num_workers = 4
    if get_using_cuda():
        num_workers = 0

    #Split training and validation.
    training_loader = DataLoader(T, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 sampler=torch.utils.data.sampler.SubsetRandomSampler(list(train_ndxs)))
    validation_loader = DataLoader(T, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(list(test_ndxs)))

    #define optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    early_stopping = EarlyStopping(patience=20, verbose=True, path=None)

    tls = []
    vls = []

    # Epochs
    for t in range(epochs_num):
        print("epochs {}...".format(t))
        ti = time.time()
        acc_loss = 0
        val_loss = 0
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
            data = sample_batched
            loss = compute_loss(model, data)
            val_loss += loss.data

        # Save loss and print status.
        val = acc_loss / (len(T) * 9 / 10)
        if hasattr(val, "item"):
            val = val.item()
        tls.append(val)
        val = val_loss / (len(T) / 10)
        vls.append(val)
        elapsed = time.time() - ti

        print("epochs {} elapsed: {:.2f}(sec)\t\tloss_train: {:.4f}\tloss_val: {:.4f}".format(
            t, elapsed, tls[-1], vls[-1]))

        if early_stop:
            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        
    return tls, vls

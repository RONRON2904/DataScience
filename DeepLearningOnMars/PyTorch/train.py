import numpy as np
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from unet import Unet
import dataset
from utils import *

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--fp16', action='store_true',
                    help='Run model fp16 mode.')
args = parser.parse_args()

train_dataset = dataset.MarsDataset()
val_dataset  = dataset.MarsDataset(val=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
model = Unet(1, 1)
model.cuda()
lr=0.01
momentum=0.9

if args.fp16:
    assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."
    model = network_to_half(model)
    param_copy = [param.clone().type(torch.cuda.FloatTensor).detach() for param in model.parameters()]
    for param in param_copy:
        param.requires_grad = True
 else:
  param_copy = list(model.parameters())
optimizer = torch.optim.SGD(param_copy, lr,momentum=momentum)

if args.fp16:
    model.zero_grad()
#optimizer = optim.Adagrad(model.parameters(), lr=0.01)

def train(epoch):
    start = time.time()
    if args.fp16:
        loss_fn = nn.MSELoss().cuda().half()
    else:
        loss_fn = nn.MSELoss().cuda()
    
    loss_sum = 0
    for i, (x, y) in enumerate(train_loader):
        x, y_true = Variable(x), Variable(y)
        if args.fp16:
            x = x.cuda().half()
            y_true = y_true.cuda().half()
        else:
            x = x.cuda()
            y_true = y_true.cuda()
        y_pred = model(x)
        loss = loss_fn(y_pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        if args.fp16:
            set_grad(param_copy, list(model.parameters()))
        optimizer.step()
        loss_sum += loss.data[0]
        if args.fp16:
            params = list(model.parameters())
            copy_in_params(model, param_copy)
            torch.cuda.synchronize()
        print(loss_sum)
    end = time.time()
    print('epoch: {}, epoch loss: {}, duration time: {}'.format(epoch,loss.data[0]/len(train_loader), end-start))
    
for epoch in range(1):
    train(epoch)

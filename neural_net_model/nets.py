import os
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.nn.parameter import Parameter
import torch.optim as optim
import torch.cuda

import model_classes


def run_rmse_net(model, variables, X_train, Y_train):
    opt = optim.Adam(model.parameters(), lr=1e-3)
    #scheduler = optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=50, verbose=True)
    loss_train_arr = []
    loss_test_arr = []
    iteration_list = []

    for i in range(1000):
        opt.zero_grad()
        model.train()
        train_loss = nn.MSELoss()(
                model(variables['X_train_']), variables['Y_train_'])
        train_loss.backward()
        opt.step()
        scheduler.step(1.)
        print('Epoch-{0} lr: {1}'.format(i, opt.param_groups[0]['lr']))

        model.eval()
        test_loss = nn.MSELoss()(
                model(variables['X_test_']), variables['Y_test_'])

        print(i, train_loss.item(), test_loss.item())
        iteration_list.append(i)
        loss_train_arr.append(train_loss.item())
        loss_test_arr.append(test_loss.item())

    model.eval()

    return model, iteration_list, loss_train_arr, loss_test_arr


def rmse_loss(mu_pred, Y_actual):
    return ((mu_pred - Y_actual)**2).mean(dim=0).sqrt().data.cpu().numpy()


def eval_net(model, variables):
    
    model.eval()
    mu_pred_train = model(variables['X_train_'])
    mu_pred_test = model(variables['X_test_'])

    # Eval model on rmse
    train_rmse = rmse_loss(mu_pred_train, variables['Y_train_'])
    test_rmse = rmse_loss(mu_pred_test, variables['Y_test_'])

    return train_rmse, test_rmse, mu_pred_train, mu_pred_test

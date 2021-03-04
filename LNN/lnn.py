import torch
import numpy as np
import os, sys, random
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLP
from data import get_dataset
from utils import L1_loss, rk4
from torch.utils.tensorboard import SummaryWriter  
writer = SummaryWriter('log')



class LNN(torch.nn.Module):
    '''learn Lagrangian'''
    def __init__(self, input_dim, differentiable_model ):
        super(LNN, self).__init__()
        self.differentiable_model = differentiable_model

    def forward_new(self, x):  #x is ( 2 * degree of freedom ,)     '2' is position,velocity

        freedom = x.shape[0]
        freedom = freedom / 2
        freedom = int(freedom)

        L = self.differentiable_model(x)
        dx, = torch.autograd.grad(L, x, create_graph=True)
        hess = torch.autograd.functional.hessian(self.differentiable_model, x)
        spl1, spl2 = hess.split(freedom, 0)
        grad_v_q, grad_v_v = spl2.split(freedom, 1)
        grad_v_v_inverse = torch.pinverse(grad_v_v)

        acce = torch.mm(grad_v_v_inverse, dx[:freedom].unsqueeze(1) - torch.mm(grad_v_q, x[ freedom:].unsqueeze(1))).squeeze()  #acceleration
        return acce

def train():
    data = get_dataset()
    x = torch.tensor(data['x'], requires_grad=True, dtype=torch.float32)  # x 实际上是位置和速度
    test_x = torch.tensor( data['test_x'], requires_grad=True, dtype=torch.float32)
    _, acce = torch.Tensor(data['test_dx']).chunk(2, 1)
    _, test_acce = torch.Tensor(data['test_dx']).chunk(2, 1)
    N, freedom = x.shape
    freedom /= 2
    input_dim = int(freedom * 2)
    output_dim = int(freedom)
    model_nn = MLP(input_dim,50, output_dim, 'tanh')
    model = LNN(input_dim, differentiable_model=model_nn)
    optim = torch.optim.Adam(model.parameters(), 5e-3, weight_decay=1e-4)
    # vanilla train loop
    stats={'train_loss': [], 'test_loss': []}
    torch.autograd.set_detect_anomaly(True)
    for step in range(500):  #500 epoch
    # train step
        loss=0
        for i in range(100):
            acce_hat=model.forward_new(x[i])
            loss = loss + L1_loss(acce[i], acce_hat)
        loss.backward()
        loss /= 100
        optim.step()
        optim.zero_grad()
        print("step {}, train_loss {:.4e}, ".format(step, loss))
        writer.add_scalar('LNN/spring_train_loss', loss, step)

train()
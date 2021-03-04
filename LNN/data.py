# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import autograd
import autograd.numpy as np
import torch
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp  #https://vimsky.com/examples/usage/python-scipy.integrate.solve_ivp.html
#这个就是用来解一个微分方程的初值问题

def hamiltonian_fn(coords):
    q, p = np.split(coords,2)
    H = p**2 + q**2 # spring hamiltonian (linear oscillator)
    return H

def dynamics_fn(t, coords):
    dcoords = autograd.grad(hamiltonian_fn)(coords)   #对哈密顿量求梯度，算对时间的微分
    dqdt, dpdt = np.split(dcoords,2)
    S = np.concatenate([dpdt, -dqdt], axis=-1)
    return S

def get_trajectory(t_span=[0,3], timescale=10, radius=None, y0=None, noise_std=0.1, **kwargs):#一条轨迹是30个点
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))
    #这个函数用来造出‘真实’数据
    # get initial state
    if y0 is None:  #一维谐振子对平衡位置的偏移
        y0 = np.random.rand(2)*2-1
    if radius is None:
        radius = np.random.rand()*0.9 + 0.1 # sample a range of radii
    y0 = y0 / np.sqrt((y0**2).sum()) * radius ## set the appropriate radius

    spring_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    q, p = spring_ivp['y'][0], spring_ivp['y'][1]
    dydt = [dynamics_fn(None, y) for y in spring_ivp['y'].T]
    dydt = np.stack(dydt).T
    dqdt, dpdt = np.split(dydt,2)
    # add noise
    q += np.random.randn(*q.shape)*noise_std
    p += np.random.randn(*p.shape) * noise_std
    return q, p, dqdt, dpdt, t_eval

def integrate_model(model, t_span=[0, 3], timescale=10, y0=None, **kwargs):  #这个相当于ground truth数据的get_trajectory
    y0=y0.detach().numpy()
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))
    def fun(t, np_x):
        x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32).view(1,2)
        dx = model.time_derivative(x).data#.numpy().reshape(-1)
        return dx
    # if y0 is None:  #一维谐振子对平衡位置的偏移
    #     y0 = np.random.rand(2)*2-1
    spring_ivp=solve_ivp(fun=fun, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    q, p = spring_ivp['y'][0], spring_ivp['y'][1]
    q = torch.from_numpy(q)
    p = torch.from_numpy(p)
    return torch.stack((q,p), 1)



def get_dataset(seed=0, samples=50, test_split=0.5, **kwargs):  #这里面得到的是算出来的“真实”轨迹
    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []
    for s in range(samples):
        x, y, dx, dy, t = get_trajectory(**kwargs)
        xs.append( np.stack( [x, y]).T )
        dxs.append( np.stack( [dx, dy]).T )
        
    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data

def get_field(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20):
    field = {'meta': locals()}

    # meshgrid to get vector field
    b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    ys = np.stack([b.flatten(), a.flatten()])
    
    # get vector directions
    dydt = [dynamics_fn(None, y) for y in ys.T]
    dydt = np.stack(dydt).T

    field['x'] = ys.T
    field['dx'] = dydt.T
    return field
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import sys

def dirichlet(s, n, m, M):
    w = np.random.exponential(scale=1, size=s*m).reshape(m,s)
    w_mean = np.repeat(np.mean(w, axis=1),s).reshape(m,s)
    w_s = w / w_mean
    w = np.repeat(w_s, M, axis=0)
    if s == n:
        w_m = w_s
    elif n % s == 0:
        w_m = np.repeat(w_s, n/s).reshape(m,n)
    else:
        w_m = np.repeat(w_s, int(n/s)+1)[0:m*n]
        w_m = w_m.reshape(m,n)
    return w, w_m
def input_G(n, m, M, w):
    z = np.random.normal(loc=0, scale=1, size = n*m*M).reshape(m*M,n)
    input_g = np.concatenate((w,z),axis=1)
    return input_g
def trans_cuda(obj):
    if torch.cuda.is_available():
       device = torch.device('cuda')
    else:
       device = torch.device('cpu')
    obj = np.asarray(obj)
    obj = torch.from_numpy(obj)
    obj = obj.to(device, dtype=torch.float)
    return obj


def GB_NPMLE_Sampling(fit, Boot_size):
  
    if torch.cuda.is_available(): device = torch.device('cuda')
    else: device = torch.device('cpu')
    
    tau = fit[1]
    p = fit[2]
    k = fit[3] #k=l
    dist = fit[4]
    s = fit[5]
    n = fit[6]
    zm = fit[7] #zm=q
    NN = fit[0].to(device)
    NN.eval()
    
    size = int(Boot_size) 
    
    Generation_start = time.perf_counter()
    with torch.no_grad(): 
      Theta = torch.zeros(size, p)
      index = torch.tensor(range(k))
      index_num = torch.round(size * trans_cuda(tau))
      if torch.sum(index_num) == (size):
        sample = np.repeat(index, index_num.cpu().detach())
      else:
        index_num[torch.argmax(index_num)] = index_num[torch.argmax(index_num)] + torch.tensor(size) - torch.sum(index_num)
        sample = np.repeat(index, index_num.cpu().detach())
      Sample = np.repeat(sample, p)
      w, w_m = dirichlet(s, n, m=size, M=1)
      input_g = trans_cuda(input_G(n=zm, m=size, M=1, w=w))
      output = NN(input_g).reshape(size, p*k)
      output = torch.index_select(output,
                                  1,
                                  torch.arange(0,k*p).reshape(k,p).transpose(1,0).reshape(-1).to(device))
    
      Theta = torch.gather(output,1,Sample.reshape(size*p,1).to(device)).reshape(size,p)
      Theta = pd.DataFrame(Theta.cpu().detach().numpy())
    
    if dist == "Gaussian location": Theta_dist = Theta
    if dist == "Gaussian scale": Theta_dist = np.exp(Theta)
    if dist == "Poisson": Theta_dist = np.exp(Theta)
    if dist == "Gamma rate": Theta_dist = np.exp(Theta)
    if dist == "Gamma shape": Theta_dist = np.exp(Theta)
    if dist == "Binomial": Theta_dist = np.exp(Theta)/(1+np.exp(Theta))
    if dist == "Uniform": Theta_dist = np.exp(Theta)+param
    if dist == "Weibull scale": Theta_dist = np.exp(Theta)
    
    Generation_time = time.perf_counter() - Generation_start
    return Theta_dist, Generation_time, p

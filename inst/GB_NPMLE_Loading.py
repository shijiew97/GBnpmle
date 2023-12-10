import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import sys

def GB_NPMLE_Loading(Option, Save_file, Gpu_ind, N1, M, Num_it, Verb, Tol, LrDecay, Boot_size,
Tau0, P, Q, Dist, S, n1, Zm, Hidden_size, Param, Lr, X, Y, Lrpower):
  
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
  
  gpu_ind = int(Gpu_ind)
  if gpu_ind == -1:
    print("WARNING: CPU computing would be very slow!")
  else:
    if torch.cuda.is_available():
      device = torch.device('cuda', gpu_ind)
    else:
      device = torch.device('cpu')
      print("WARNING: CPU computing would be very slow!")
  
  p = int(P)   
  k = int(Q)   
  dist = str(Dist)
  s = int(S) 
  n = int(n1) 
  zm = int(Zm)
  hidden_size = int(Hidden_size)
  param = float(Param)
  lr = lr0 = float(Lr)
  option = str(Option)
  
  N = int(N1)
  m = int(M)
  iteration = int(Num_it)
  verb = int(Verb)
  tol = float(Tol)
  lrDecay = int(LrDecay)
  lrpower = float(Lrpower)
  
  size = int(Boot_size) 
  
  class Net(nn.Module):
    def __init__(self, hidden_size): 
      super(Net, self).__init__()
      self.relu = nn.ReLU()
      self.fc1 = nn.Linear(s+zm, hidden_size) 
      self.fc2 = nn.Linear(hidden_size, hidden_size)
      self.fc3 = nn.Linear(hidden_size, hidden_size)
      self.fc_out = nn.Linear(hidden_size, p*k) 
    
    def forward(self, X1):
      out = self.relu( self.fc1(X1) )
      out = self.relu( self.fc2(out) )
      out = self.relu( self.fc3(out) )
      out = self.fc_out(out)
      return out
  
  NN = Net(hidden_size).to(device)
  NN.load_state_dict(torch.load(Save_file))
  print("Successfully loading trained Generator!")
  
  if option == "Train":

    if torch.is_tensor(X) == False: 
      X = torch.from_numpy(X).to(device, dtype=torch.float)
    else: X = X.to(device, dtype=torch.float)
    
    if torch.is_tensor(Y) == False: 
      Y = torch.from_numpy(Y).to(device, dtype=torch.float).view(n,p)
    else: Y = Y.to(device, dtype=torch.float).view(n,p)
    
    if k == 1:
      em0 = 0
      method = "GMS Algorithm 1"
      print("Training G via GMS Algorithm 1!")
    else:
      em0 = 1
      method = "Two-stage Algorithm"
      print("Training G via Two-stage Algorithm")
      
    optimizer = torch.optim.Adam(NN.parameters(), lr=lr)
    LOSS = torch.zeros(iteration)
    
    Generator_start = time.perf_counter()
    for it in range(iteration):
      
      if lrDecay == 1:
        lr = lr0/(float(it+1.0)**lrpower) 
        for param_group in optimizer.param_groups:
          param_group["lr"] = lr
      
      w, w_m = dirichlet(s=s, n=n, m=N, M=m)
    
      input_g = input_G(n=zm, m=N, M=m, w=w)
      if dist == "Gaussian location": output = NN(trans_cuda(input_g)) 
      if dist == "Gaussian scale": output = torch.exp(NN(trans_cuda(input_g)))
      if dist == "Poisson": output = torch.exp(NN(trans_cuda(input_g)))
      if dist == "Gamma rate": output = torch.exp(NN(trans_cuda(input_g)))
      if dist == "Gamma shape": output = torch.exp(NN(trans_cuda(input_g)))
      if dist == "Binomial": output = torch.exp(NN(trans_cuda(input_g)))/(1+torch.exp(NN(trans_cuda(input_g))))
      if dist == "Uniform": output = torch.exp(NN(trans_cuda(input_g)))
      if dist == "Weibull scale": output = torch.exp(NN(trans_cuda(input_g)))
      
      output1 = torch.index_select(output,
                                  1,
                                  torch.arange(0,k*p).reshape(k,p).transpose(1,0).reshape(-1).to(device)).reshape(N*m*p,k)
  
      Theta = torch.gather(output1, 
                           1, 
                           torch.multinomial(torch.ones(k)/k, 
                                             N*m*p,
                                             replacement=True).reshape(N*m*p,1).to(device)).reshape(N,m,p)
  
      result = torch.index_select(torch.matmul(Theta,X.transpose(1,0)).reshape(N,n*m),
                                   1,
                                   torch.arange(0,n*m).reshape(m,n).transpose(1,0).reshape(-1).to(device)).reshape(N,n,m) 
  
      if dist == "Gaussian location":
        normal = torch.distributions.normal.Normal(result, param)
        pdf = torch.exp(normal.log_prob(Y))
        out = trans_cuda(w_m).reshape(N, n) * torch.log(torch.mean(pdf,dim=2))
  
      if dist == "Gaussian scale":
        normal = torch.distributions.normal.Normal(param, result)
        pdf = torch.exp(normal.log_prob(Y))
        out = trans_cuda(w_m).reshape(N, n) * torch.log(torch.mean(pdf,dim=2))
        
      if dist == "Poisson": 
        poi = torch.distributions.poisson.Poisson(result)
        pmf = torch.exp(poi.log_prob(Y)) + 1E-20
        out = trans_cuda(w_m).reshape(N,n) * torch.log(torch.mean(pmf,dim=2))
        
      if dist == "Gamma rate":
        gamm = torch.distributions.gamma.Gamma(param,result)
        pdf = torch.exp(gamm.log_prob(Y))
        out = trans_cuda(w_m).reshape(N,n) * torch.log(torch.mean(pdf,dim=2))
  
      if dist == "Gamma shape":
        gamm = torch.distributions.gamma.Gamma(result,param)
        pdf = torch.exp(gamm.log_prob(Y))
        out = trans_cuda(w_m).reshape(N,n) * torch.log(torch.mean(pdf,dim=2))
       
      if dist == "Binomial":
        bino = torch.distributions.binomial.Binomial(param,probs=result)
        pdf_bin0 = torch.exp(bino.log_prob(Y))
        out = trans_cuda(w_m).reshape(N,n) * torch.log(torch.mean(pdf_bin0,dim=2))  
      
      if dist == "Uniform":
        unif = torch.distributions.uniform.Uniform(low=param,high=result)
        pdf = torch.exp(unif.log_prob(Y))
        out = trans_cuda(w_m).reshape(N,n) * torch.log(torch.mean(pdf,dim=2))  
      
      if dist == "Weibull scale":
        weib = torch.distributions.weibull.Weibull(scale=result, concentration=param)
        pdf = torch.exp(weib.log_prob(Y))
        out = trans_cuda(w_m).reshape(N,n) * torch.log(torch.mean(pdf,dim=2))
      
      loss = - torch.mean(torch.sum(out, dim=1))
      
      optimizer.zero_grad() 
      loss.backward()
      optimizer.step()
        
      LOSS[it] = loss.item()
      if (it+1)%10==0 and verb == 1:
        percent = float((it+1)*100) /iteration
        arrow   = '-' * int(percent/100 *20 -1) + '>'
        spaces  = ' ' * (20-len(arrow))
        train_time = time.perf_counter() - Generator_start
        print('\r[%s/%s]'% (it+1, iteration), 'Progress: [%s%s] %d %%' % (arrow, spaces, percent),
        " Current/Initial Loss: {:.2f}/{:.0f}, Method: {}, Learning rate: {}, Training time: {:.2f}".format(LOSS[it], LOSS[0], str(method), lr, train_time,), end='')
        sys.stdout.flush()
    Generator_time = time.perf_counter() - Generator_start
    
    #EM algorithm starts
    EM_start = time.perf_counter()
    max_iter = 5000
    tau = torch.ones(k)/k
    if em0 == 1:
      pz_sum = np.zeros((n,k))
      Tau = np.zeros((max_iter,k))
      it = 1;sub_size = 20;Tau[0,] = tau;M = 1
      y = torch.cat(sub_size*[Y.view(1,n)],0)
      while it <= max_iter :
          for j in range(k):
              W, w_m = dirichlet(s, n, m=sub_size, M=M)
              with torch.no_grad():
                input_g = trans_cuda(input_G(n=zm, m=sub_size, M=1, w=W))
                output = NN(input_g).reshape(sub_size, p*k)
                output = torch.index_select(output,
                                            1,
                                            torch.arange(0,k*p).reshape(k,p).transpose(1,0).reshape(-1).to(device))
                Theta = output.reshape(sub_size,p,k)
                output = torch.matmul(Theta[:,:,j], X.transpose(1,0))
                
              if dist == "Gaussian location":
                normal = torch.distributions.normal.Normal(output,scale=param)
                prob = torch.exp((normal.log_prob(y)))
                prob = prob.cpu().detach().numpy()
              if dist == "Gaussian scale":
                normal = torch.distributions.normal.Normal(param,scale=output)
                prob = torch.exp((normal.log_prob(y)))
                prob = prob.cpu().detach().numpy()
              if dist == "Poisson": 
                poi = torch.distributions.poisson.Poisson(torch.exp(output))
                prob = torch.exp(poi.log_prob(y)) + 1E-20
                prob = prob.cpu().detach().numpy()
              if dist == "Gamma rate":
                gamm = torch.distributions.gamma.Gamma(param,torch.exp(output))
                prob = torch.exp(gamm.log_prob(y))
                prob = prob.cpu().detach().numpy()
              if dist == "Gamma shape":
                gamm = torch.distributions.gamma.Gamma(torch.exp(output),param)
                prob = torch.exp(gamm.log_prob(y))
                prob = prob.cpu().detach().numpy()
              if dist == "Binomial":
                bino = torch.distributions.binomial.Binomial(param,probs=torch.exp(output)/(1+torch.exp(output)))
                prob = torch.exp(bino.log_prob(y))
                prob = prob.cpu().detach().numpy()
              if dist == "Uniform":
                unif = torch.distributions.uniform.Uniform(low=param, high=torch.exp(output))
                prob = torch.exp(unif.log_prob(y))
                prob = prob.cpu().detach().numpy()
              if dist == "Weibull scale":
                weib = torch.distributions.weibull.Weibull(scale=torch.exp(output), concentration=param)
                prob = torch.exp(weib.log_prob(y))
                prob = prob.cpu().detach().numpy()
  
              pz_sum[:, j] = np.mean(prob, axis=0)
          pz = pz_sum/np.repeat(np.sum(pz_sum,axis=1).reshape(n,1), k, axis=1)
          tau = np.mean(pz, axis=0)
          
          Tau[it,] = tau
          change = np.max(np.abs(Tau[it,] - Tau[it-1,]))
          
          if change < tol : break
          else: it = it+1
    EM_time = time.perf_counter() - EM_start
    return NN, tau, p, k, dist, s, n, zm, Generator_time, EM_time, hidden_size, param, lr, N, M, X, Y, tol, lrDecay
  
  if option == "Sample":
    NN.eval()
    
    Generation_start = time.perf_counter()
    with torch.no_grad(): 
      Theta = torch.zeros(size, p)
      index = torch.tensor(range(k))
      index_num = torch.round(size * trans_cuda(Tau0))
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
    
    print("Generation of Bootstrap samples Done!")
    Generation_time = time.perf_counter() - Generation_start
    return Theta_dist, Generation_time, p 

  
      

  
  
  

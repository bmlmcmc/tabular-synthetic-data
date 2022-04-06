import pandas as pd
import torch
from torch import nn, tensor
import torch.nn.functional as F
import torch.distributions as D
from torchvision.transforms.autoaugment import TrivialAugmentWide

def construct_layer(n_features,act_funcs,close=''):
  layers = []
  nl = len(n_features)
  if nl==1:
    layers.append(nn.Linear(n_features[0],n_features[0]))
    if close=='logit':
      layers.append(nn.Sigmoid())
  else:
    for i in range(nl-1):
      layers.append(nn.Linear(n_features[i],n_features[i+1]))
      if act_funcs[i]=='relu':
        layers.append(nn.ReLU())
      elif act_funcs[i]=='bnorm':
        layers.append(nn.BatchNorm1d(n_features[i+1]))
    if close=='logit':
      layers[-1] = nn.Sigmoid()
    elif close=='bnorm':
      layers[-1] = nn.BatchNorm1d(n_features[i+1])
    elif close is None:
      layers.pop()
  return nn.Sequential(*layers)

def normalize_discrete_columns(data,indexes):
  for i in range(len(indexes)-1):
    cc = data[:,indexes[i]:indexes[i+1]]
    data[:,indexes[i]:indexes[i+1]] = torch.exp(cc)/torch.exp(cc).sum(axis=1).view(-1,1) ## Softmax
    # data[:,indexes[i]:indexes[i+1]] = cc/cc.sum(axis=1).view(-1,1) ## Softmax
  return data

def getmax_discrete_columns(data,indexes):
  dt = data[:,indexes[0]:indexes[1]].argmax(axis=1).view(-1,1)
  for i in range(len(indexes)-2):
    cc = data[:,indexes[i+1]:indexes[i+2]].argmax(axis=1).view(-1,1)
    dt = torch.concat([dt,cc],1)
  return dt

## Kullback-Leibler Divergence

def kl_div(mu,z):
  q = torch.distributions.Normal(mu,torch.exp(mu/2))
  qzx = q.log_prob(z)
  p = torch.distributions.Normal(torch.zeros_like(mu),torch.ones_like(mu))
  pzx = p.log_prob(z)
  kld = qzx-pzx
  return kld.sum()

def kl_div_ber(mu,z):
  q = torch.distributions.Bernoulli(probs=mu)
  qzx = q.log_prob(z) ## log posterior
  p = torch.distributions.Bernoulli(logits=torch.zeros_like(mu))
  pzx = p.log_prob(z) ## log prior
  kld = qzx-pzx
  return kld.mean()

## Kullback-Leibler Divergence using concrete distribution

def kl_div_concrete(y,z,temp=torch.tensor([2])):
  q = D.relaxed_bernoulli.LogitRelaxedBernoulli(temp,logits=y)
  qzx = q.log_prob(z) ## log posterior
  p = D.relaxed_bernoulli.LogitRelaxedBernoulli(temp,logits=torch.zeros_like(y))
  pzx = p.log_prob(z) ## log prior
  kld = qzx-pzx
  return kld.mean()

## generate gumbel distribution

def rgumbel(shape,seed):
  torch.manual_seed(seed)
  u = torch.rand(shape)
  return torch.log(-torch.log(u))

def concrete(g,log_alpha,temperature): ##g = gumbel generator, log_alpha = estimated parameter, temperature=lambda in maddison
  glogalpha = g+log_alpha
  return F.softmax(glogalpha/temperature)

## data verification

def verify_data(aa,bb,nocols):
  aa1 = []
  bb1 = []
  for index in range(len(aa)):
    aa1.append(getmax_discrete_columns(aa[index],nocols))
    bb1.append(getmax_discrete_columns(bb[index],nocols))

  for i in range(len(nocols)-1):
    index = i
    dt1 = pd.DataFrame(torch.cat(aa1,axis=0).numpy())
    dt2 = pd.DataFrame(torch.cat(bb1,axis=0).numpy())
    ac = pd.crosstab(dt1[index],dt1[index])
    print('--real data--'+str(ac.shape))
    # print(pd.crosstab(dt1[index],dt1[index]))
    ac = pd.crosstab(dt2[index],dt2[index])
    print('--synthetic data--'+str(ac.shape))
    # print(pd.crosstab(dt2[index],dt2[index]))
    print(pd.crosstab(dt1[index],dt2[index]))
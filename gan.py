import pandas as pd, matplotlib.pyplot as plt, time
from data_prep import *
from calculations import *
import torch
# from torch.tensor import detach
from torch import optim as torch_optim, nn, tensor
import torch.nn.functional as F
import torch.distributions as D

def train_gan(data,nocols,gen,dis,dist='concrete',temp=2,epochs=50,loss=nn.BCELoss(),gantype='vanilla'):
  s = 0
  ## prepare the components
  generator = gen
  discriminator = dis
  
  for i in generator.parameters():
    input_length = i.size(-1)
    break

  generator_opt = torch.optim.Adam(generator.parameters(),lr=0.001)
  discriminator_opt = torch.optim.Adam(discriminator.parameters(),lr=0.001)

  start1 = time.time()
  real_data = []
  gen_data = []
  for epoch in range(epochs):
    # temp *= temp_decay
    if epoch%10 == 0:
      start = time.time()
    for x,y in data:
      x = x.float()
      batch_size = x.size(0)
      ## Initialisation
      ## Noise
      if dist=='normal':
        d = torch.randn(size = (batch_size,input_length)).float()
        noise = d
      elif dist=='concrete':
        d = D.relaxed_bernoulli.LogitRelaxedBernoulli(temp,logits=torch.zeros(x.size(0),input_length))
        noise = d.sample()
      else:
        return "distribution is only 'normal' and 'concrete'"
      Gz = generator(noise)
      # Gz = normalize_discrete_columns(Gz,nocols) 

      ## Data and labels
      true_labels = torch.ones(batch_size).view(-1,1) ##generate true label
      true_data = x.view(-1,x.size(1))
      fake_labels = torch.zeros(batch_size).view(-1,1)

      ## Discriminator - Real data
      discriminator_opt.zero_grad()
      Dx = discriminator(x)
      
      ## Discriminator - Fake data
      DGz = discriminator(Gz.detach())
      
      if gantype == 'vanilla':
        loss_dx = loss(Dx,true_labels)
        loss_dgz = loss(DGz,fake_labels)
        loss_total = loss_dx+loss_dgz
      elif gantype == 'wasserstein':
        loss_total = torch.mean(Dx) - torch.mean(DGz)
      loss_total.backward()
      discriminator_opt.step()

      if gantype=='wasserstein':
        for p in discriminator.parameters():
          p.data.clamp_(-0.01, 0.01)

      ## Generator - Update
      generator_opt.zero_grad()
      DGz2 = discriminator(Gz)
      if gantype == 'vanilla':
        loss_dgz2 = loss(DGz2,true_labels)
      elif gantype == 'wasserstein':
        loss_dgz2 = torch.mean(DGz2)
      loss_dgz2.backward()
      generator_opt.step()
      
      if epoch+1==epochs:
        real_data.append(true_data)
        gen_data.append(Gz)
    if epoch%10 == 9:
      end = time.time()
      print('computational time for epoch '+str(epoch+1)+': '+str(round(end-start,3)))

  end1 = time.time()
  print('computational time total: '+str(round(end1-start1,1)))

  return real_data, gen_data,generator, discriminator

######## EXAMPLE #########
######### MNIST ##########

class Generator(nn.Module):
  def __init__(self):
    super(Generator,self).__init__()
    self.gen = construct_layer([8,32,128,784],True)
  
  def forward(self,x):
    return self.gen(x)

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator,self).__init__()
    self.dis = construct_layer([784,128,32,8,1],True)

  def forward(self,x):
    return self.dis(x)

############# EXAMPLE ############
######### COVID-19 DATA ##########

## building the generator and discriminator
class Generator(nn.Module):
  def __init__(self,indexes:list,each_cat:int):
    super(Generator,self).__init__()
    ## 3 -> 6 -> 12 -> 24 -> 33
    # self.gen = construct_layer([input_length],True)
    self.gen = nn.Sequential(
        # nn.ReLU(),
        nn.Linear(2,8),
        nn.Dropout(),
        nn.Linear(8,33)
    )
    self.indexes = indexes
    self.each_cat = each_cat

  def forward(self,x):
    genx = self.gen(x)
    if self.each_cat:
      for i in range(len(self.indexes)-1):
        s = nn.Softmax(dim=1)
        genx[:,self.indexes[i]:self.indexes[i+1]] = s(genx[:,self.indexes[i]:self.indexes[i+1]].clone())
      return genx
    else:
        return nn.Sigmoid()(genx)

class Discriminator(nn.Module):
  def __init__(self,gantype:str):
    super(Discriminator,self).__init__()
    # self.dis = construct_layer([input_length,1],True)
    if gantype=='vanilla':
      self.dis = nn.Sequential(
          nn.Linear(33,8),
          nn.Linear(8,1),
          nn.Sigmoid()
      )
    elif gantype=='wasserstein':
      self.dis = nn.Sequential(
        nn.Linear(33,8),
        nn.Linear(8,1)
      )
  def forward(self,x):
    return self.dis(x)

data_covid1, col_index = data_covid(35)

covid_real_data1,covid_gan_generated_data1,covid_gan_loss1,covid_gan_model1 = train_gan(data_covid1,col_index,Generator(col_index,True),Discriminator('vanilla'),dist='normal',epochs=20,gantype='vanilla')
covid_real_data2,covid_gan_generated_data2,covid_gan_loss2,covid_gan_model2 = train_gan(data_covid1,col_index,Generator(col_index,True),Discriminator('wasserstein'),dist='normal',epochs=20,gantype='wasserstein')
covid_real_data3,covid_gan_generated_data3,covid_gan_loss3,covid_gan_model3 = train_gan(data_covid1,col_index,Generator(col_index,True),Discriminator('vanilla'),dist='concrete',epochs=20,gantype='vanilla')
covid_real_data4,covid_gan_generated_data4,covid_gan_loss4,covid_gan_model4 = train_gan(data_covid1,col_index,Generator(col_index,True),Discriminator('wasserstein'),dist='concrete',epochs=20,gantype='wasserstein')

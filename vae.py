import pandas as pd, matplotlib.pyplot as plt, time
from data_prep import *
from calculations import *
import torch
# from torch.tensor import detach
from torch import optim as torch_optim, nn, tensor
import torch.nn.functional as F
import torch.distributions as D

def train_vae(data,model,kld,kld_multiplier=0.0001,loss=nn.BCELoss(),epochs=50):
  ## build AE model
  optimizer = torch_optim.AdamW(model.parameters(),lr=0.001,weight_decay=1e-8)
  start1 = time.time()

  losses = []

  for i in model.parameters():
    ncols = i.size(-1)
    break

  recon_res = []
  real_res = []

  ## loop epoch then image
  for epoch in range(epochs):
    if epoch%10 == 0:
      start = time.time()
    for x,label in data:
      optimizer.zero_grad()
      xx = x.reshape(-1,ncols).float()
      encode,z,recon = model(xx) ## get AE output from image (image reconstruction)

      recon_loss = loss(recon,xx)
      kl = kld(encode,z) ## remove this/set to zero if you use traditional AE
      loss_total = recon_loss+(kld_multiplier*kl) ## loss from the recon-ed image
      loss_total.backward()
      losses.append(loss_total.detach()) ## save loss

      optimizer.step() ## run optimizer

      if epoch == epochs-1:
        real_res.append(xx)
        recon_res.append(recon)
    if epoch%10 == 9:
      end = time.time()
      print('computational time for '+str(epoch+1)+'-th epoch: '+str(round(end-start,3)))
  return real_res,recon_res,losses,model

######## EXAMPLE #########
######### MNIST ##########

## initialise variational autoencoder

class VAE_mnist(nn.Module):
  def __init__(self):
    super(VAE_mnist,self).__init__() ##initialise parent

    # encoder order: 28*28 = 784 ==> 128 ==> 64 ==> 36 ==> 18 ==> 9
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    self.encoder = nn.Sequential(
        nn.Linear(28*28, 128), ##first input: dimension of mnist, nn linear defines no. of input and output
        nn.ReLU(), ##define activation for each layer
        nn.Linear(128,64),
        nn.ReLU(), 
        nn.Linear(64,36),
        nn.ReLU(), 
        nn.Linear(36,18),
        nn.ReLU(), 
        nn.Linear(18,9),##stop here or add more compression
        nn.ReLU(),
        nn.Linear(9,2)
    )

    ## decoder order: 9 ==> 18 ==> 36 ==> 64 ==> 128 ==> 784 ==> 28*28 = 784
    self.decoder = nn.Sequential(
        nn.Linear(2,9),
        nn.ReLU(),
        nn.Linear(9,18),
        nn.ReLU(),
        nn.Linear(18,36),
        nn.ReLU(),
        nn.Linear(36,64),
        nn.ReLU(),
        nn.Linear(64,128),
        nn.ReLU(),
        nn.Linear(128,28*28),
        nn.Sigmoid() ##for decoder there is a last activation function to define 0 and 1 (real and fake?)
    )

  def sampling_latent(self,x):
    # print(x)
    q = D.Normal(x,torch.exp(x/2))
    z = q.rsample()
    return z

  def forward(self,x):
    encoded = self.encoder(x)
    z = self.sampling_latent(encoded) ## just remove this (and return z) if you want to use AE
    # print(z)
    decoded = self.decoder(z)
    # return decoded
    return encoded,z,decoded

### RUN THE VAE ON MNIST
mnist_real_data,mnist_vae_generated_data,mnist_vae_loss,mnist_vae_model = train_vae(data_mnist(32),VAE_mnist(),kl_div,nn.BCELoss(),5)

### PLOT THE RESULTS ###
index = 2
index2 = 11
item = mnist_real_data[index].view(-1,28,28)
item2 = mnist_vae_generated_data[index].view(-1,28,28)

fig = plt.figure(figsize=(10, 7))
fig.add_subplot(1, 3, 1)
plt.imshow(item[index2].detach().numpy())
plt.title('Real data')
fig.add_subplot(1, 3, 2)
plt.imshow(item2[index2].detach().numpy())
plt.title('VAE results')

######### TABULAR DATA: IRIS ##########

## initialise variational autoencoder


class tVAE_iris(nn.Module):
  def __init__(self):
    super(tVAE_iris,self).__init__() ##initialise parent

    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    self.encoder = nn.Sequential(
        nn.Linear(7,1),
        nn.BatchNorm1d(1),
    )

    self.decoder = nn.Sequential(
        nn.Linear(1,7),
        nn.BatchNorm1d(7),  
    )

  def sampling_latent(self,x):
    q = D.Normal(x,torch.exp(x/2))
    z = q.rsample()
    return z

  def forward(self,x):
    encoded = self.encoder(x)
    z = self.sampling_latent(encoded) ## just remove this (and return z) if you want to use AE
    decoded = self.decoder(z)
    return encoded,z,decoded

iris_real_data,iris_vae_generated_data,iris_vae_loss,iris_vae_model = train_vae(data_iris(15),tVAE_iris(),kl_div,150)

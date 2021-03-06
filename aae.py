import pandas as pd, matplotlib.pyplot as plt, time
from data_prep import *
from calculations import *
import torch
# from torch.tensor import detach
from torch import optim as torch_optim, nn, tensor
import torch.nn.functional as F
import torch.distributions as D

def train_aae(data,nocols,enc,dec,dis,latent_length=2,epochs=50,loss_ae=nn.BCELoss(),loss_gan=nn.BCELoss(),
             gantype='vanilla'):
    encoder = enc(latent_length)
    decoder = dec(latent_length,nocols,nocols is not None)
    discriminator = dis(latent_length, gantype)
    
    for i,j in data:
        ncols = i.size(1)
        break
    
    encoder_opt = torch.optim.Adam(encoder.parameters(), lr=3e-04)
    decoder_opt = torch.optim.Adam(decoder.parameters(),lr=3e-04)
    discriminator_opt = torch.optim.Adam(discriminator.parameters(),lr=0.001)

    start1 = time.time()
    real_data = []
    gen_data = []
    
    loss_gen = loss_gan
    loss_dis = loss_gan
    for epoch in range(epochs):
        # temp *= temp_decay
        if epoch%10 == 0:
            start = time.time()
        for (dt,labels) in data:
            batch_size = dt.size(0)
            x = dt.reshape(-1,ncols).float()
            
            #generator_opt.zero_grad()
            #encoded,recon = generator(x)
            #Reconstruction
            z_sample = encoder(x)
            X_sample = decoder(z_sample)
            
            recon_loss = loss_ae(X_sample, x)
            recon_loss.backward()
            encoder_opt.step()
            decoder_opt.step()
            encoder.zero_grad()
            decoder.zero_grad()
            discriminator.zero_grad()
            
            #Discriminator
            z_real = torch.randn(15, latent_length)
            z_fake = encoder(x)
            
            D_real = discriminator(z_real)
            D_fake = discriminator(z_fake)
            
            if gantype == 'vanilla':
                D_loss = -torch.mean(torch.log(D_real) + torch.log(1 - D_fake))
            elif gantype == 'wasserstein':
                D_loss = -(torch.mean(D_real) - torch.mean(D_fake))
                
            D_loss.backward()
            discriminator_opt.step()
            
            if gantype=='wasserstein':
                for p in discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)
            
            encoder.zero_grad()
            decoder.zero_grad()
            discriminator.zero_grad()
            
            z_fake = encoder(x)
            D_fake = discriminator(z_fake)
            
            if gantype == 'vanilla':
                G_loss = -torch.mean(torch.log(D_fake))
            elif gantype == 'wasserstein':
                G_loss = -torch.mean(D_fake)
                
            G_loss.backward()
            encoder_opt.step()
            encoder.zero_grad()
            decoder.zero_grad()
            discriminator.zero_grad()
            
            if epoch+1==epochs:
                real_data.append(x)
                gen_data.append(X_sample)
        if epoch%10 == 9:
            end = time.time()
            print('computational time for epoch '+str(epoch+1)+': '+str(round(end-start,3)))
            
    end1 = time.time()
    print('computational time total: '+str(round(end1-start1,1)))

    return real_data, gen_data,encoder, decoder, discriminator

############# EXAMPLE ############
######### COVID-19 DATA ##########

## building the generator and discriminator
class Generator(nn.Module):
  def __init__(self,latent_length:int,indexes:list,each_cat:int):
    super(Generator,self).__init__()
    self.encoder = nn.Sequential(
        nn.Linear(33,8),
        # nn.Dropout(),
        nn.Linear(8,latent_length)
    )

    self.decoder = nn.Sequential(
        # nn.ReLU(),
        nn.Linear(latent_length,8),
        # nn.Dropout(),
        nn.Linear(8,33),
        # nn.Sigmoid()
    )
    self.indexes = indexes
    self.each_cat = each_cat

  def forward(self,x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    if self.each_cat:
      for i in range(len(self.indexes)-1):
        s = nn.Softmax(dim=1)
        decoded[:,self.indexes[i]:self.indexes[i+1]] = s(decoded[:,self.indexes[i]:self.indexes[i+1]].clone())
      return encoded,decoded
    else:
      return encoded,nn.Sigmoid()(decoded)
    
class Discriminator(nn.Module):
  def __init__(self,latent_length:int,gantype:str):
    super(Discriminator,self).__init__()
    if gantype=='vanilla':
      self.dis = nn.Sequential(
          nn.Linear(latent_length,1),
          nn.Sigmoid()
      )
    elif gantype=='wasserstein':
      self.dis = nn.Sequential(
          nn.Linear(latent_length,1)
      )

  def forward(self,x):
    return self.dis(x)

data_covid1, col_index = data_covid(35)

covid_real_data1,covid_aae_generated_data1,covid_aae_loss1,covid_aae_model_enc1,covid_aae_model_dec1,covid_aae_model_dis1 = train_aae(data_covid1, col_index,Encoder, Decoder, Discriminator,latent_length=8,gantype='vanilla',epochs=20)
covid_real_data2,covid_aae_generated_data2,covid_aae_loss2,covid_aae_model_enc2,covid_aae_model_dec2,covid_aae_model_dis2 = train_aae(data_covid1, col_index,Encoder, Decoder, Discriminator,latent_length=8,gantype='wasserstein',epochs=20)
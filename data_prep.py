import pandas as pd, numpy as np, matplotlib.pyplot as plt, time,math

## basic tutorial pytorch https://pub.towardsai.net/pytorch-tutorial-for-beginners-8331afc552c4

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch import optim as torch_optim, nn, tensor
import torch.nn.functional as F
import torch.distributions as D
from torchvision.transforms.autoaugment import TrivialAugmentWide
torch.manual_seed(10)

from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import load_iris

## Special tabular dataset

### CALL DATA GENERAL ###

class CustomTabularDataset(Dataset):
  def __init__(self, X,y):
    self.X = X
    self.y = y

  def __len__(self):
    return len(self.y)

  def __getitem__(self, idx):
    if str(type(self.X))=="<class 'pandas.core.frame.DataFrame'>":
      data = self.X.iloc[idx].values
    else:
      data = self.X[idx]
    label = self.y[idx]
    return data,label

### CALLING DEFAULT DATA ###

def data_mnist(batch_size):
	# Transforms images to a PyTorch Tensor
	tensor_transform = transforms.ToTensor()

	# Download the MNIST Dataset
	dataset = datasets.MNIST(root = "./data",
							train = True,
							download = True,
							transform = tensor_transform)

	# DataLoader is used to load the dataset
	# for training
	return DataLoader(dataset = dataset,batch_size = batch_size,shuffle = True)
 
def data_iris(batch_size,combine=TrivialAugmentWide):
  iris = load_iris()
  iris_data = torch.tensor(iris.data)
  lb = LabelBinarizer()
  lb.classes_= np.arange(3)
  iris_target = torch.tensor(lb.transform(iris.target))
  if combine: iris_data = torch.cat([iris_data,iris_target],axis=1)
  iris_batched = CustomTabularDataset(iris_data,iris_target)
  return DataLoader(iris_batched, batch_size=batch_size, shuffle=True)

## prepare the data
def data_covid(batch_size):
  data = pd.read_csv('data_covid_synthetic.csv')
  data['kelumur'] = np.where((data['kelUsia_2.Produktif']==0) & (data['kelUsia_3.Lansia']==0),0,
                            np.where((data['kelUsia_2.Produktif']==1) & (data['kelUsia_3.Lansia']==0),1,2))
  data = data.drop(['kelUsia_2.Produktif','kelUsia_3.Lansia'],1)

  ## number of columns
  nocol = data.apply(lambda x: len(x.unique()),axis=0).values
  nocols = nocol.cumsum()
  nocols = np.insert(nocols,0,0)

  ## obtain dummies and convert to torch
  data_dummies = pd.get_dummies(data.astype(str))
  X_list = CustomTabularDataset(data_dummies,data['HASILLAB'])
  X_list = DataLoader(X_list, batch_size=batch_size, shuffle=True)
  return X_list,nocols

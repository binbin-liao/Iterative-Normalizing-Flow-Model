from time import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from tqdm import tqdm_notebook
import torch.nn.functional as F
import torch.autograd as autograd
import pandas as pd
import gc
import scipy
import torch.nn.init as init
import seaborn as sns
import torch.distributions
from numpy.random import shuffle
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from matplotlib.image import NonUniformImage
from matplotlib import cm


NN = 30

class Bn_dense(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(Bn_dense,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(),
            nn.Linear(out_dim,out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU()
        )
        
        self.shortcut = nn.Sequential(
            nn.Linear(in_dim,out_dim),
            nn.BatchNorm1d(out_dim)
        )
        
    def forward(self,x):
        return self.net(x)+self.shortcut(x)
    
def subnet_fc(dim_in,dims_out):
    return nn.Sequential(
        Bn_dense(dim_in,256),
        Bn_dense(256,256),
        Bn_dense(256,dims_out))

class Flow(nn.Module):
    def __init__(self,in_dim):
        super(Flow,self).__init__()
        self.inn = Ff.SequenceINN(in_dim)
        for k in range(8):
            self.inn.append(Fm.AllInOneBlock,subnet_constructor=subnet_fc)
            self.inn.append(PermuteRandom,seed=k)
    
    def forward(self,x,rev=False,jac=True):
        return self.inn(x,rev=rev,jac=jac)



class Flow_cond(nn.Module):
    def __init__(self,in_dim,cond_dim):
        super(Flow_cond,self).__init__()
        self.inn = Ff.SequenceINN(in_dim)
        for k in range(8):
            self.inn.append(Fm.AllInOneBlock, cond=0,cond_shape=(cond_dim,),subnet_constructor=subnet_fc)
            self.inn.append(PermuteRandom,seed=k)
    
    def forward(self,x,c,rev=False,jac=True):
        return self.inn(x,c=[c],rev=rev,jac=jac)

def ewma(x, span=200):
    return pd.DataFrame({'x': x}).ewm(span=span).mean().values[:, 0]
    
def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()    
    
def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)

r_index = [3,11,12,21,23,24]
vs_index = list((set(range(NN))-set([4,5,6,7,8,9,10,11]))-set([27,28,29]))#13
ot_index = list((set(range(NN))-set([27,28,29])))#17
model_pair=dict()
layer_pair=dict(zip(list(range(27)),
                ['Inn1','Inn2','Inn3','Inn4',
                 'Out1','Out2','Out3','Out4','Out5','Out6','Out7','Out8',
                 'Lm1','Lm21','Lm22','Lm23','Lm24','Lm25','Lm26','Lm27','Lm28','Lm3',
                 'Tz1','Tz2','Tz3','LVZ','LID']))
temp=0
for i in ['rad','den','vpv','vsv','vph','vsh','eta']:
    if i == 'rad':
        for j in r_index:
            model_pair.setdefault(layer_pair[j]+'_'+i,temp)
            temp += 1
    elif i in set(['vsv','vsh']):
        for j in vs_index:
            model_pair.setdefault(layer_pair[j]+'_'+i,temp)
            temp += 1
    else:
        for j in ot_index:
            model_pair.setdefault(layer_pair[j]+'_'+i,temp)
            temp += 1


g1 = (191/256,197/256,214/256)
g2 = (128/256,166/256,226/256)
g3 = (251/256,221/256,133/256)
g4 = (244/256,111/256,67/256)
g5 = (207/256,67/256,62/256)
cmap_colors=[g1,
             #(195/256,214/256,206/256),
             #(91/256,115/256,20/256),
             (238/256,190/256,4/256),
             (233/256,160/256,14/256),
             (217/256,112/256,14/256),
             (198/256,49/256,6/256),
             (158/256,29/256,6/256)
             #(1,0,0),
             #(0,0,0)
            ]

custom_cmap = LinearSegmentedColormap.from_list('CustomCmap', cmap_colors)
g1 = (31/256,91/256,37/256)
g2 = (102/256,157/256,98/256)
g3 = (194/256,214/256,164/256)
g4 = (231/256,229/256,204/256)

f1 = (231/256,229/256,204/256)
f2 = (194/256,214/256,164/256)
f3 = (156/256,193/256,132/256)
f4 = (102/256,157/256,98/256)
f5 = (60/256,124/256,61/256)
f6 = (31/256,91/256,37/256)
f7 = (30/256,61/256,20/256)
f8 = (25/256,40/256,19/256)

r1 = (238/256,190/256,4/256)
r2 = (233/256,160/256,14/256)
r3 = (217/256,112/256,14/256)
r4 = (189/256,49/256,6/256)

r1 = (91/256,31/256,37/256)
r2 = (157/256,102/256,98/256)
r3 = (214/256,194/256,164/256)
r4 = (229/256,231/256,204/256)

b1 = (27/256,22/256,91/256)
b2 = (98/256,110/256,157/256)
b3 = (164/256,194/256,214/256)
b4 = (201/256,234/256,229/256)

g = [g4,g3,g2,g1]
r = [r4,r3,r2,r1]
b = [b4,b3,b2,b1]
custom_cmap_b = LinearSegmentedColormap.from_list('CustomCmap', b)
custom_cmap_r = LinearSegmentedColormap.from_list('CustomCmap', r)
custom_cmap_g = LinearSegmentedColormap.from_list('CustomCmap', g)

c1 = (232/256,211/256,192/256)
c2 = (212/256,186/256,173/256)
c3 = (216/256,156/256,122/256)
c4 = (183/256,127/256,112/256)
c5 = (180/256,116/256,107/256)
c6 = (114/256,78/256,82/256)

c = [c1,c2,c3,c4,c5,c6]
custom_cmap_c = LinearSegmentedColormap.from_list('CustomCmap', c)

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 12
        }

g1 = (191/256,197/256,214/256)
cmap_colors3=[g1,g1,g1,g1]
custom_cmap3 = LinearSegmentedColormap.from_list('CustomCmap', cmap_colors3)
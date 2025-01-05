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
import gc
import scipy
import torch.nn.init as init
import seaborn as sns
import torch.distributions
from numpy.random import shuffle

from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom
import FrEIA.framework as Ff
import FrEIA.modules as Fm
#import sys
#sys.path.append('/home/lenovo/Working/Notebook/Normal_mode_inversion/')

from tools import *

import importlib
import tools
importlib.reload(tools)
from tools import *

import validation_tools
importlib.reload(validation_tools)
from validation_tools import *

import Model_generator
importlib.reload(Model_generator)
from Model_generator import *

#import Nets
#from Nets import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from IPython.display import clear_output

obs_real = np.loadtxt("./data/new_obs_real.txt")
obs_std = np.loadtxt("./data/new_obs_err.txt")
obs_real = obs_real[:167]
obs_std = obs_std[:167]

obs_t_real = np.array([1.31929,1.43845,1.75019,1.92525,2.10340,
                       3.09907,3.25352,
                      2.29528,2.48487,2.75300,2.91257,3.83052,4.01381,
                      4.19269,4.36875,
                      3.20008,3.60368,5.05139,4.77537])*1000.
obs_t_std = np.array([0.17e-3, 0.08e-3,0.13e-3,0.12e-3,0.06e-3,
                     0.10e-3,0.06e-3,
                    0.22e-3,0.15e-3,0.33e-3,0.08e-3,0.21e-3,0.24e-3,0.22e-3,
                     0.53e-3,
                     1.73e-3,0.24e-3,0.23e-3,1.86e-3])*1000.
obs_real = np.concatenate([obs_real,obs_t_real])
obs_std = np.concatenate([obs_std,obs_t_std])

obs_min = np.loadtxt('./data/new_obs_min_0.txt')
obs_inter = np.loadtxt('./data/new_obs_inter_0.txt')

var_np = obs_std/obs_inter*2.
var_np = var_np[:INPUT_SIZE]

#load  Forward_net
filename = './net/Forward_net.pth'
Forward_net = Flow(INPUT_SIZE).cuda()
state = torch.load(filename,weights_only=True)
Forward_net.load_state_dict(state)
Forward_net.eval();

obs_fix1 = np.loadtxt('./data/obs_fix1')
obs_fix2 = np.loadtxt('./data/obs_fix2')
obs_fix3 = np.loadtxt('./data/obs_fix3')
obs_fix4 = np.loadtxt('./data/obs_fix4')
obs_fix5 = np.loadtxt('./data/obs_fix5')

#calculate target distribution p(m|\hat{d})
err_dist = torch.distributions.MultivariateNormal(torch.zeros(INPUT_SIZE,dtype=torch.float32),
                                                                    torch.diag(torch.from_numpy(np.power(var_np,2))).
                                                                    type(torch.float32))
obs_tensor = torch.randn(BATCH_SIZE,INPUT_SIZE,device=device,dtype=torch.float32)
obs_tensor[:] = torch.tensor(obs_fix1[0],dtype=torch.float32)
obs_sample = obs_tensor + err_dist.sample((BATCH_SIZE,)).to('cuda')
ert_sample = Forward_net(obs_sample.to(device),rev=True,jac=False)[0].detach().cpu()

#load file
filename = './net/Inverse_flow0_val_1.pth'
Inverse_flow_old1 = Flow_cond(OUTPUT_SIZE,INPUT_SIZE).cuda()
state = torch.load(filename,weights_only=True)
Inverse_flow_old1.load_state_dict(state)
Inverse_flow_old1.eval();

#load file
filename = './net/Inverse_flow0_val_2.pth'
Inverse_flow_old2 = Flow_cond(OUTPUT_SIZE,INPUT_SIZE).cuda()
state = torch.load(filename,weights_only=True)
Inverse_flow_old2.load_state_dict(state)
Inverse_flow_old2.eval();

#load file
filename = './net/Inverse_flow0_val_3.pth'
Inverse_flow_old3 = Flow_cond(OUTPUT_SIZE,INPUT_SIZE).cuda()
state = torch.load(filename,weights_only=True)
Inverse_flow_old3.load_state_dict(state)
Inverse_flow_old3.eval();

#load file
filename = './net/Inverse_flow0_val_4.pth'
Inverse_flow_old4 = Flow_cond(OUTPUT_SIZE,INPUT_SIZE).cuda()
state = torch.load(filename,weights_only=True)
Inverse_flow_old4.load_state_dict(state)
Inverse_flow_old4.eval();

#load file
filename = './net/Inverse_flow1_1_val.pth'
Inverse_flow1 = Flow_cond(OUTPUT_SIZE,INPUT_SIZE).cuda()
state = torch.load(filename,weights_only=True)
Inverse_flow1.load_state_dict(state)
Inverse_flow1.eval();

#load file
filename = './net/Obs_normal_2_val.pth'
Obs_Normal_Flow_s2 = Flow(INPUT_SIZE).cuda()
state = torch.load(filename,weights_only=True)
Obs_Normal_Flow_s2.load_state_dict(state)
Obs_Normal_Flow_s2.eval();

#load file
filename = './net/Mod_normal_2_val.pth'
Mod_Normal_Flow_s2 = Flow(OUTPUT_SIZE).cuda()
state = torch.load(filename,weights_only=True)
Mod_Normal_Flow_s2.load_state_dict(state)
Mod_Normal_Flow_s2.eval();

#load file
filename = './net/Inverse_flow_s2_val.pth'
Inverse_flow_s2 = Flow_cond(OUTPUT_SIZE,INPUT_SIZE).cuda()
state = torch.load(filename,weights_only=True)
Inverse_flow_s2.load_state_dict(state)
Inverse_flow_s2.eval();

#load file
filename = './net/Obs_normal_3_val.pth'
Obs_Normal_Flow_s3 = Flow(INPUT_SIZE).cuda()
state = torch.load(filename,weights_only=True)
Obs_Normal_Flow_s3.load_state_dict(state)
Obs_Normal_Flow_s3.eval();

#load file
filename = './net/Mod_normal_3_val.pth'
Mod_Normal_Flow_s3 = Flow(INPUT_SIZE).cuda()
state = torch.load(filename,weights_only=True)
Mod_Normal_Flow_s3.load_state_dict(state)
Mod_Normal_Flow_s3.eval();

#load file
filename = './net/Inverse_flow_s3_val.pth'
Inverse_flow_s3 = Flow_cond(OUTPUT_SIZE,INPUT_SIZE).cuda()
state = torch.load(filename,weights_only=True)
Inverse_flow_s3.load_state_dict(state)
Inverse_flow_s3.eval();

#load file
filename = './net/Obs_normal_4_val.pth'
Obs_Normal_Flow_s4 = Flow(INPUT_SIZE).cuda()
state = torch.load(filename,weights_only=True)
Obs_Normal_Flow_s4.load_state_dict(state)
Obs_Normal_Flow_s4.eval();

#load file
filename = './net/Mod_normal_4_val.pth'
Mod_Normal_Flow_s4 = Flow(INPUT_SIZE).cuda()
state = torch.load(filename,weights_only=True)
Mod_Normal_Flow_s4.load_state_dict(state)
Mod_Normal_Flow_s4.eval();

#load file
filename = './net/Inverse_flow_s4_val.pth'
Inverse_flow_s4 = Flow_cond(OUTPUT_SIZE,INPUT_SIZE).cuda()
state = torch.load(filename,weights_only=True)
Inverse_flow_s4.load_state_dict(state)
Inverse_flow_s4.eval();

BATCH_SIZE = 256
ert_fake = np.zeros((2,6,BATCH_SIZE,INPUT_SIZE))
obs_tensor = torch.randn(BATCH_SIZE,INPUT_SIZE,device=device,dtype=torch.float32)

ert_fake[1,5]=ert_sample.detach().cpu().numpy()
ert_fake[0,5]=ert_sample.detach().cpu().numpy()

rand_ert = torch.randn(BATCH_SIZE,OUTPUT_SIZE,device=device)

obs_tensor[:] = torch.tensor(obs_fix1[0],dtype=torch.float32)
ert_fake[0,0] = Inverse_flow_old1(rand_ert,obs_tensor,rev=True,jac=False)[0].detach().cpu().numpy() 
ert_fake[0,1] = Inverse_flow_old2(rand_ert,obs_tensor,rev=True,jac=False)[0].detach().cpu().numpy() 
ert_fake[0,2] = Inverse_flow_old3(rand_ert,obs_tensor,rev=True,jac=False)[0].detach().cpu().numpy() 
ert_fake[0,3] = Inverse_flow_old4(rand_ert,obs_tensor,rev=True,jac=False)[0].detach().cpu().numpy() 
#ert_fake[0,4] = Inverse_flow_old4(rand_ert,obs_tensor,rev=True,jac=False)[0].detach().cpu().numpy()

ert_fake[1,0] = Inverse_flow1(rand_ert,obs_tensor,rev=True,jac=False)[0].detach().cpu().numpy()

obs_tensor[:] = torch.tensor(obs_fix2[0],dtype=torch.float32)
obs_code = Obs_Normal_Flow_s2(obs_tensor)[0].detach()
ert_fake_code = Inverse_flow_s2(rand_ert,obs_code,rev=True,jac=False)[0].detach()
ert_fake[1,1] = Mod_Normal_Flow_s2(ert_fake_code,rev=True,jac=False)[0].detach().cpu().numpy()

obs_tensor[:] = torch.tensor(obs_fix3[0],dtype=torch.float32)
obs_code = Obs_Normal_Flow_s3(obs_tensor)[0].detach()
ert_fake_code = Inverse_flow_s3(rand_ert,obs_code,rev=True,jac=False)[0].detach()
ert_fake[1,2] = Mod_Normal_Flow_s3(ert_fake_code,rev=True,jac=False)[0].detach().cpu().numpy()

obs_tensor[:] = torch.tensor(obs_fix4[0],dtype=torch.float32)
obs_code = Obs_Normal_Flow_s4(obs_tensor)[0].detach()
ert_fake_code = Inverse_flow_s4(rand_ert,obs_code,rev=True,jac=False)[0].detach()
ert_fake[1,3] = Mod_Normal_Flow_s4(ert_fake_code,rev=True,jac=False)[0].detach().cpu().numpy()

for i in range(4):
    filename1 = f'./data/Validation/Traditional{i}.txt'
    filename2 = f'./data/Validation/INFM{i}'
    np.savetxt(filename1,ert_fake[0,i])
    np.savetxt(filename2,ert_fake[1,i])
filename = f'./data/Validation/Target.txt'
np.savetxt(filename,ert_fake[0,5])


N1 = model_pair['Lm1_den']
N2 = model_pair['Lm1_vpv']

norm = Normalize(vmin=0, vmax=1)

fig,ax = plt.subplots(4,3,figsize=(6,8), gridspec_kw={'width_ratios': [1, 1, 1.]})

text = ['10k epoches','20k epoches','30k epoches','40k epoches','Real Poterior PDF']
for i in range(3):
    for j in range(4):
        if  j == 3 and i != 2 :
            ax[j,i].set_xticks([-0.3,0.2,0.7])
        else:
            ax[j,i].set_xticks([])
        if i == 0:
            ax[j,i].set_yticks([-0.5,0,0.5])
        else:
            ax[j,i].set_yticks([])

        ax[3,2].set_xticks([0.1,0.2,0.3])
        ax[j,2].set_yticks([-0.1,0.,0.1])
        ax[j,2].yaxis.tick_right()

        #if i == 1 and j == 3:
        #    ax[i,j].set_xticks([0.1,0.2,0.3])
        #    ax[i,j].set_yticks([-0.1,0,0.1])
            
        if i == 0:
            cmap = custom_cmap_b
            kde_plot = sns.kdeplot(x=ert_fake[i,j,:,N1],y=ert_fake[i,j,:,N2],common_norm=False,fill=True,cmap=cmap,ax=ax[j,i],cbar=False,
                                  cbar_kws={'ticks':[],'norm': norm})
        if i == 1:
            cmap = custom_cmap_r
            kde_plot = sns.kdeplot(x=ert_fake[i,j,:,N1],y=ert_fake[i,j,:,N2], common_norm=False,fill=True,cmap=cmap,ax=ax[j,i],cbar=False,
                                cbar_kws={'ticks':[],'norm': norm})
        if i == 2:
            cmap = custom_cmap_r
            kde_plot = sns.kdeplot(x=ert_fake[1,j,:,N1],y=ert_fake[1,j,:,N2], common_norm=False,fill=True,cmap=cmap,ax=ax[j,i],cbar=False,
                                cbar_kws={'ticks':[],'norm': norm})
        kde_plot = sns.kdeplot(x=ert_fake[0,5,:,N1],y=ert_fake[0,5,:,N2], common_norm=False,fill=True,cmap='Greys',ax=ax[j,i])
        if i == 0 and j == 0:
            cbar_mappable = kde_plot.get_children()[0]
        #ax[i,j].set_xticks([-1,0,1])
        #ax[i,j].set_yticks([-1,0,1])
        ax[j,i].set_xlim([-0.3,0.7])
        ax[j,i].set_ylim([-0.5,0.5])        
        if i == 2:
            ax[j,i].set_xlim([0.1,0.3])
            ax[j,i].set_ylim([-0.1,0.1]) 
        else:
            ax[j,i].set_xlim([-0.3,0.7])
            ax[j,i].set_ylim([-0.5,0.5])
for sax in ax.flat:
    sax.tick_params(axis='both', labelsize=10)

plt.tight_layout(pad=0.5, w_pad=0.1, h_pad=0.1)
plt.savefig('./fig/validation_Lm.png',dpi=300,bbox_inches='tight',transparent=True)

ce1 = np.loadtxt('./data/cross_entorpy1.txt')
ce2 = np.loadtxt('./data/cross_entorpy2.txt')
cross_entorpy1_1 = np.loadtxt('./data/cross_entorpy1_1.txt')
cross_entorpy2_1 = np.loadtxt('./data/cross_entorpy2_1.txt')
cross_entorpy1_2 = np.loadtxt('./data/cross_entorpy1_2.txt')
cross_entorpy2_2 = np.loadtxt('./data/cross_entorpy2_2.txt')
cross_entorpy1_3 = np.loadtxt('./data/cross_entorpy1_3.txt')
cross_entorpy2_3 = np.loadtxt('./data/cross_entorpy2_3.txt')
cross_entorpy1_4 = np.loadtxt('./data/cross_entorpy1_4.txt')
cross_entorpy2_4 = np.loadtxt('./data/cross_entorpy2_4.txt')
cross_entorpy1_5 = np.loadtxt('./data/cross_entorpy1_5.txt')
cross_entorpy2_5 = np.loadtxt('./data/cross_entorpy2_5.txt')
c=np.concatenate([cross_entorpy2_1,cross_entorpy1_2+cross_entorpy2_2,cross_entorpy1_3[:100]+cross_entorpy2_3[:100],cross_entorpy1_4[:100]+cross_entorpy2_4[:100]],axis=0)

fig = plt.figure(figsize=(16,4))
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 20,
        }
font2 = {'family': 'serif',
        'weight': 'normal',
        'size': 20,
        }
N = 400
x = np.arange(0,40,0.1)
plt.plot(x,c[:N],lw=3.,label='Iterative Normalizing Flow',c=b2)
plt.plot(x,ce2[:N],lw=3.,label='Iterative Normalizing Flow',c=r2)
# 获取当前的Axes对象
ax = plt.gca()

# 设置 x 轴刻度的字体属性
ax.tick_params(axis='x', which='both',
 labelsize=20, color=font['color'], labelcolor=font['color'])
ax.tick_params(axis='y', which='both', labelsize=20, color=font['color'], labelcolor=font['color'])


# 设置 x 轴刻度值
xticks_values = [0, 10, 20, 30, 40]
ax.set_xticks(xticks_values)
ax.set_yticks([-4,-3,-2,-1,0,1])
xticks_labels = [f"{val}k" for val in xticks_values]
plt.xticks(xticks_values, labels=xticks_labels)
plt.xlabel('Epoches',fontdict=font)
plt.ylabel('Cross entropy',fontdict=font)
plt.legend(prop=font2)
plt.savefig('./fig/KL.png',dpi=300)

ert_val = torch.tensor(ert_fake[1,3],dtype=torch.float32,device=device)
obs_val = Forward_net(ert_val)[0].detach().cpu().numpy()
diff1 = (obs_val-obs_fix1)*obs_inter[:INPUT_SIZE]
x = obs_real[:INPUT_SIZE]
y1 = diff1.std(axis=0)
y2 = diff1.mean(axis=0)
sort_index = np.argsort(x)
x_sorted = x[sort_index]
y1_sorted1 = y1[sort_index]
y2_sorted1 = y2[sort_index]
obs_std_sorted = obs_std[sort_index]

ert_val = torch.tensor(ert_fake[0,3],dtype=torch.float32,device=device)
obs_val = Forward_net(ert_val)[0].detach().cpu().numpy()
diff = (obs_val-obs_fix1)*obs_inter[:INPUT_SIZE]
x = obs_real[:INPUT_SIZE]
y1 = diff.std(axis=0)
y2 = diff.mean(axis=0)
sort_index = np.argsort(x)
x_sorted = x[sort_index]
y1_sorted = y1[sort_index]
y2_sorted = y2[sort_index]
obs_std_sorted = obs_std[sort_index]


fig,(ax2,ax1) = plt.subplots(1,2,figsize=(8.0,4.0))

ymax = 2.*obs_std_sorted
ymin = -2.*obs_std_sorted

ymax1 = y2_sorted1+2.*y1_sorted1
ymin1 = y2_sorted1-2.*y1_sorted1

ymax2 = y2_sorted+2.*y1_sorted
ymin2 = y2_sorted-2.*y1_sorted

ax1.vlines(x,ymax1,ymin1,color=r2)
ax1.vlines(x,ymax,ymin,color='k')


ax1.set_xlabel('Frequency($\mu Hz$)',fontdict=font)
ax1.set_xticks([0,3000,6000,9000])
ax1.set_xticklabels(['0','3k','6k','9k'])
ax1.set_yticks([-12,-6,0,6,12])
ax1.set_title('Iterative Normalizing Flow')

for label in ax1.get_xticklabels()+ax1.get_yticklabels():
    label.set_fontname('serif')


ax2.vlines(x,ymax2,ymin2,color=b2)
ax2.vlines(x,ymax,ymin,color='k')

ax2.set_ylabel('differences($\mu Hz$)',fontdict=font)
ax2.set_xlabel('Frequency($\mu Hz$)',fontdict=font)
ax2.set_xticks([0,3000,6000,9000])
ax2.set_xticklabels(['0','3k','6k','9k'])
ax2.set_yticks([-80,-40,0,40,80])
ax2.set_title('Traditional Normalizing Flow',fontdict=font)

for label in ax2.get_xticklabels()+ax2.get_yticklabels():
    label.set_fontname('serif')

fig.savefig('./fig/validation_std_diff.png',dpi=300)
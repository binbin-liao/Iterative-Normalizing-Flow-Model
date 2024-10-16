import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
import time
import seaborn as sns
from tools import *
import importlib

import tools
importlib.reload(tools)
from tools import *

d_model = np.loadtxt('./data/d_model.txt')
ref_model = np.loadtxt('./data/ref_model.txt')


#mineos_path ='../../../mineos_var_cig'
mineos_path = '/home/benjamin/mineos_var_cig/DEMO/Earth_v6'

# PREM model
rad = []
den = []
vpv = []
vsv = []
qk  = []
qm  = []
vph = []
vsh = []
eta = []
nodes = []


para = dict({'1':vph,'2':vsh,'3':eta,'4':rad,'5':den,'6':vpv,'7':vsv,'8':qk,'0':qm})

# Read PREM
with open('./data/model_prem.txt') as file:
    count = 0
    for line in file:
        count += 1
        if count > 3:
            if count%9 == 4:
                values = line.strip().split()
                rad.append(float(values[2]))
                nodes.append(int(values[0]))
            else:
                values = line.strip().split()
                para[str(count%9)].append([float(X) for X in values])

#Calculate the total mass
def Mass(rad,den):
    nn = 6371
    nl = 30
    Layer = [0]+[int(X) for X in rad]
    rad_pf = np.arange(nn+1)
    m = 0.0
    for i in range(nl):
        for ind in range(Layer[i],Layer[i+1]):
            r = float(rad_pf[ind])
            r1 = float(rad_pf[ind+1])
            x = r/6371.
            rho = den[i,0] + den[i,1]*x + den[i,2]*x*x + den[i,3]*x*x*x
            m += 4./3.*np.pi*(np.power(r1,3)-np.power(r,3))*rho
    return m*1e12

#Calculate the moment of inertia
def Inertia(rad,den):
    nn = 6371
    nl = 30
    Layer = [0]+[int(X*1) for X in rad]
    rad_pf = np.arange(nn+1)
    iner = 0.0
    for i in range(nl):
        for ind in range(Layer[i],Layer[i+1]):
            r = float(rad_pf[ind])
            r1 = float(rad_pf[ind+1])
            x = r/6371.
            rho = den[i,0] + den[i,1]*x + den[i,2]*x*x + den[i,3]*x*x*x
            dm = 4./3.*np.pi*(np.power(r1,3)-np.power(r,3))*rho
            iner += 2./3.*dm*r*r
    return iner*1e18

#
def model_profile(rad,den,vpv,vsv,qk,qm,vph,vsh,eta):
    nn = 6371
    nl = 30
    Layer = [0]+[int(X) for X in rad]
    rad_pf = np.arange(nn)
    den_pf = np.zeros(nn)
    vpv_pf = np.zeros(nn)
    vsv_pf = np.zeros(nn)
    qk_pf = np.zeros(nn)
    qm_pf = np.zeros(nn)
    vph_pf = np.zeros(nn)
    vsh_pf = np.zeros(nn)
    eta_pf = np.zeros(nn)
    for i in range(nl):
        for ind in range(Layer[i],Layer[i+1]):
            r = float(rad_pf[ind])
            x = r/6371.
            den_pf[ind] = den[i,0] + den[i,1]*x + den[i,2]*x*x + den[i,3]*x*x*x
            vpv_pf[ind] = vpv[i,0] + vpv[i,1]*x + vpv[i,2]*x*x + vpv[i,3]*x*x*x
            vsv_pf[ind] = vsv[i,0] + vsv[i,1]*x + vsv[i,2]*x*x + vsv[i,3]*x*x*x
            qk_pf[ind] = qk[i,0]
            qm_pf[ind] = qm[i,0]
            vph_pf[ind] = vph[i,0] + vph[i,1]*x + vph[i,2]*x*x + vph[i,3]*x*x*x
            vsh_pf[ind] = vsh[i,0] + vsh[i,1]*x + vsh[i,2]*x*x + vsh[i,3]*x*x*x
            eta_pf[ind] = eta[i,0] + eta[i,1]*x 
    return rad_pf,den_pf,vpv_pf,vsv_pf,qk_pf,qm_pf,vph_pf,vsh_pf,eta_pf


#Generate the model file of Minos
def model_format(rad,den,vpv,vsv,qk,qm,vph,vsh,eta,file_name,nodes=None):
    NN = 30
    if nodes == None:
        nodes = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,5,5,10,10,10,5,5,5,5]
    rad = [0.]+list(rad)
    t1 = "ANISOTROPIC_PREM"
    t2 = [1,1,0]
    t3 = [NN,
          nodes[0]+nodes[1]+nodes[2]+nodes[3],
          nodes[0]+nodes[1]+nodes[2]+nodes[3]+nodes[4]+nodes[5]+nodes[6]+nodes[7]+nodes[8]+nodes[9]+nodes[10]+nodes[11],
          6371.0]
    fid = open(file_name,mode='w')
    fid.write('{0:16s}\n'.format(t1))
    fid.write('{0[0]:4d}{0[1]:11.5f}{0[2]:3d}\n'.format(t2))
    fid.write('{0[0]:6d}{0[1]:4d}{0[2]:4d}{0[3]:11.5f}\n'.format(t3))
    for i in range(NN):
        fid.write('{0[0]:4d}{0[1]:11.5f}{0[2]:11.5f}\n'.format([nodes[i],rad[i],rad[i+1]]))
        fid.write('{0[0]:9.5f}{0[1]:9.5f}{0[2]:9.5f}{0[3]:9.5f}\n'.format(den[i]))
        fid.write('{0[0]:9.5f}{0[1]:9.5f}{0[2]:9.5f}{0[3]:9.5f}\n'.format(vpv[i]))
        fid.write('{0[0]:9.5f}{0[1]:9.5f}{0[2]:9.5f}{0[3]:9.5f}\n'.format(vsv[i]))
        fid.write('{0[0]:9.5f}\n'.format(qk[i]))
        fid.write('{0[0]:9.5f}\n'.format(qm[i]))
        fid.write('{0[0]:9.5f}{0[1]:9.5f}{0[2]:9.5f}{0[3]:9.5f}\n'.format(vph[i]))
        fid.write('{0[0]:9.5f}{0[1]:9.5f}{0[2]:9.5f}{0[3]:9.5f}\n'.format(vsh[i]))
        fid.write('{0[0]:9.5f}{0[1]:9.5f}\n'.format(eta[i]))
    fid.close()
    
#Generate the PREM model in the format of a numpy.array.
d0 = rad[0]/4.
d1 = (rad[1]-rad[0])/8.
d3 = (rad[3]-rad[2])/8.

rad_p = np.array([d0,d0*2.,d0*3,rad[0],
                 rad[0]+d1,rad[0]+d1*2,rad[0]+d1*3,rad[0]+d1*4,rad[0]+d1*5,rad[0]+d1*6,rad[0]+d1*7,rad[1],
                 rad[2],
                 rad[2]+d3,rad[2]+d3*2,rad[2]+d3*3,rad[2]+d3*4,rad[2]+d3*5,rad[2]+d3*6,rad[2]+d3*7,rad[3],
                 rad[4],rad[5],rad[6],rad[7],rad[8],rad[9],rad[10],rad[11],rad[12]])

temp = den
den_p = np.array([temp[0],temp[0],temp[0],temp[0],
                  temp[1],temp[1],temp[1],temp[1],temp[1],temp[1],temp[1],temp[1],
                  temp[2],temp[3],temp[3],temp[3],temp[3],temp[3],temp[3],temp[3],temp[3],
                  temp[4],temp[5],temp[6],temp[7],temp[8],temp[9],temp[10],temp[11],temp[12]])
temp = vpv
vpv_p = np.array([temp[0],temp[0],temp[0],temp[0],
                  temp[1],temp[1],temp[1],temp[1],temp[1],temp[1],temp[1],temp[1],
                  temp[2],temp[3],temp[3],temp[3],temp[3],temp[3],temp[3],temp[3],temp[3],
                  temp[4],temp[5],temp[6],temp[7],temp[8],temp[9],temp[10],temp[11],temp[12]])
temp = vsv
vsv_p = np.array([temp[0],temp[0],temp[0],temp[0],
                  temp[1],temp[1],temp[1],temp[1],temp[1],temp[1],temp[1],temp[1],
                  temp[2],temp[3],temp[3],temp[3],temp[3],temp[3],temp[3],temp[3],temp[3],
                  temp[4],temp[5],temp[6],temp[7],temp[8],temp[9],temp[10],temp[11],temp[12]])
temp = qk
qk_p = np.array([temp[0],temp[0],temp[0],temp[0],
                  temp[1],temp[1],temp[1],temp[1],temp[1],temp[1],temp[1],temp[1],
                  temp[2],temp[3],temp[3],temp[3],temp[3],temp[3],temp[3],temp[3],temp[3],
                  temp[4],temp[5],temp[6],temp[7],temp[8],temp[9],temp[10],temp[11],temp[12]])
temp = qm
qm_p = np.array([temp[0],temp[0],temp[0],temp[0],
                  temp[1],temp[1],temp[1],temp[1],temp[1],temp[1],temp[1],temp[1],
                  temp[2],temp[3],temp[3],temp[3],temp[3],temp[3],temp[3],temp[3],temp[3],
                  temp[4],temp[5],temp[6],temp[7],temp[8],temp[9],temp[10],temp[11],temp[12]])
temp = vph
vph_p = np.array([temp[0],temp[0],temp[0],temp[0],
                  temp[1],temp[1],temp[1],temp[1],temp[1],temp[1],temp[1],temp[1],
                  temp[2],temp[3],temp[3],temp[3],temp[3],temp[3],temp[3],temp[3],temp[3],
                  temp[4],temp[5],temp[6],temp[7],temp[8],temp[9],temp[10],temp[11],temp[12]])
temp = vsh
vsh_p = np.array([temp[0],temp[0],temp[0],temp[0],
                  temp[1],temp[1],temp[1],temp[1],temp[1],temp[1],temp[1],temp[1],
                  temp[2],temp[3],temp[3],temp[3],temp[3],temp[3],temp[3],temp[3],temp[3],
                  temp[4],temp[5],temp[6],temp[7],temp[8],temp[9],temp[10],temp[11],temp[12]])
temp = eta
eta_p = np.array([temp[0],temp[0],temp[0],temp[0],
                  temp[1],temp[1],temp[1],temp[1],temp[1],temp[1],temp[1],temp[1],
                  temp[2],temp[3],temp[3],temp[3],temp[3],temp[3],temp[3],temp[3],temp[3],
                  temp[4],temp[5],temp[6],temp[7],temp[8],temp[9],temp[10],temp[11],temp[12]])

#Convert the normalized model parameters into a model file.
def Sample(step=1,doc_num=5):
    NN = 30
    Sample_N = 10000
    r_index = [3,11,12,21,23,24]
    vs_index = list((set(range(NN))-set([4,5,6,7,8,9,10,11]))-set([27,28,29]))#13
    ot_index = list((set(range(NN))-set([27,28,29])))#17
    #logvs_index = [64,66,67,68,69,70,71,72,73]

    nump = 27
    nums = 19

    for num in range(1,doc_num+1):
        #file_name = 'Inverse_sample_{0}.txt'.format(num)
        file_name = './data/Inverse_sample_step{0}_{1}.txt'.format(step,num)
        ert_inv = np.loadtxt(file_name)
        #ert_inv = ert_inv1

        ert_para = ref_model+ert_inv*d_model/2.
        for temp in range(Sample_N):
            rad_fin = np.copy(rad_p.reshape(NN,1))
            den_fin = np.copy(den_p)
            vpv_fin = np.copy(vpv_p)
            vsv_fin = np.copy(vsv_p)
            qm_fin  = np.copy(qm_p)
            qk_fin  = np.copy(qk_p)
            vph_fin = np.copy(vph_p)
            vsh_fin = np.copy(vsh_p)
            eta_fin = np.copy(eta_p)

            rad_fin[r_index,0]  = ert_para[temp,0:6]
            den_fin[ot_index,0] = ert_para[temp,6:6+nump] 
            vpv_fin[ot_index,0] = ert_para[temp,6+nump:6+2*nump]
            vsv_fin[vs_index,0] = ert_para[temp,6+2*nump:6+2*nump+nums]
            vph_fin[ot_index,0] = ert_para[temp,6+2*nump+nums:6+3*nump+nums]
            vsh_fin[vs_index,0] = ert_para[temp,6+3*nump+nums:6+3*nump+2*nums]
            eta_fin[ot_index,0] = ert_para[temp,6+3*nump+2*nums:6+4*nump+2*nums]
            #qm_fin[vs_index,0]  = np.power(10,ert_para[temp,64:73])
            #qk_fin[ot_index,0]  = np.power(10,ert_para[temp,73:83])
            #qk_fin[qk_fin>300000.] = 300000.
            #qk_fin[qk_fin<30.] = 30.
            #den_fin[ot_index,1] = ert_para[temp,83:93]
            #vpv_fin[ot_index,1] = ert_para[temp,93:103]
            #vsv_fin[vs_index,1] = ert_para[temp,103:112]
            #vph_fin[ot_index,1] = ert_para[temp,112:122]
            #vsh_fin[vs_index,1] = ert_para[temp,122:131]
            model_name = mineos_path+'/Earth_model_s{0}_{1}/Earth_{2}.txt'.format(step,num,temp)
            model_format(rad_fin.reshape([NN]),den_fin,vpv_fin,vsv_fin,qk_fin,qm_fin,vph_fin,vsh_fin,eta_fin,model_name)

def Sample_test(step=1,doc_num=1,Sample_N=200):
    NN = 30
    r_index = [3,11,12,21,23,24]
    vs_index = list((set(range(NN))-set([4,5,6,7,8,9,10,11]))-set([27,28,29]))#19
    ot_index = list((set(range(NN))-set([27,28,29])))#27
    #logvs_index = [64,66,67,68,69,70,71,72,73]

    nump = 27
    nums = 19

    for num in range(1,doc_num+1):
        #file_name = 'Inverse_sample_{0}.txt'.format(num)
        file_name = './data/Inverse_sample_step{0}_{1}_test.txt'.format(step,num)
        ert_inv = np.loadtxt(file_name)
        #ert_inv = ert_inv1

        ert_para = ref_model+ert_inv*d_model/2.
        for temp in range(Sample_N):
            rad_fin = np.copy(rad_p.reshape(NN,1))
            den_fin = np.copy(den_p)
            vpv_fin = np.copy(vpv_p)
            vsv_fin = np.copy(vsv_p)
            qm_fin  = np.copy(qm_p)
            qk_fin  = np.copy(qk_p)
            vph_fin = np.copy(vph_p)
            vsh_fin = np.copy(vsh_p)
            eta_fin = np.copy(eta_p)

            rad_fin[r_index,0]  = ert_para[temp,0:6]
            den_fin[ot_index,0] = ert_para[temp,6:6+nump] 
            vpv_fin[ot_index,0] = ert_para[temp,6+nump:6+2*nump]
            vsv_fin[vs_index,0] = ert_para[temp,6+2*nump:6+2*nump+nums]
            vph_fin[ot_index,0] = ert_para[temp,6+2*nump+nums:6+3*nump+nums]
            vsh_fin[vs_index,0] = ert_para[temp,6+3*nump+nums:6+3*nump+2*nums]
            eta_fin[ot_index,0] = ert_para[temp,6+3*nump+2*nums:6+4*nump+2*nums]
            #qm_fin[vs_index,0]  = np.power(10,ert_para[temp,64:73])
            #qk_fin[ot_index,0]  = np.power(10,ert_para[temp,73:83])
            #qk_fin[qk_fin>300000.] = 300000.
            #qk_fin[qk_fin<30.] = 30.
            #den_fin[ot_index,1] = ert_para[temp,83:93]
            #vpv_fin[ot_index,1] = ert_para[temp,93:103]
            #vsv_fin[vs_index,1] = ert_para[temp,103:112]
            #vph_fin[ot_index,1] = ert_para[temp,112:122]
            #vsh_fin[vs_index,1] = ert_para[temp,122:131]
            model_name = mineos_path+'/Earth_model_s{0}_{1}_test/Earth_{2}.txt'.format(step,num,temp)
            model_format(rad_fin.reshape([NN]),den_fin,vpv_fin,vsv_fin,qk_fin,qm_fin,vph_fin,vsh_fin,eta_fin,model_name)

#Obtain the identification numbers of the normal modes
def add_freq_indx(n,m):
    global temp
    try:
        for i in iter(m):
            freq_pair.setdefault(str(n)+'S'+str(i),temp)
            temp += 1
            if n == 0:
                freq_indx.append(i-2)
            elif n == 1:
                freq_indx.append(i-2+25)
            else:
                freq_indx.append(n*26+i-2-1)
    except TypeError:
        freq_pair.setdefault(str(n)+'S'+str(m),temp)
        temp += 1
        if n == 0:
            freq_indx.append(m-2)
        elif n == 1:
            freq_indx.append(m-2+25)
        else:
            freq_indx.append(n*26+m-2-1)

#pick out target frequency
freq_indx = []
freq_pair = dict([])
temp = 0
add_freq_indx(0,range(2,10))
add_freq_indx(0,range(11,18))
add_freq_indx(0,[19,20,21])
add_freq_indx(1,range(2,17))
add_freq_indx(2,[1])
add_freq_indx(2,range(3,17))
add_freq_indx(2,[25])
add_freq_indx(3,[1,2,6,7,8,9,25,26])
add_freq_indx(4,[1,2,3,4,5])
add_freq_indx(5,range(2,9))
add_freq_indx(5,[11,12,14,15,16,17])
add_freq_indx(6,[3,9,10,15,18])
add_freq_indx(7,range(5,10))
add_freq_indx(8,[1,5,6,7,10])
add_freq_indx(9,[2,3,4,6,8,10,11,12,13,14,15])
add_freq_indx(10,[10,17,18,19,20,21])
add_freq_indx(11,[1,4,5,6,9,10,12,14,23,24,25])
add_freq_indx(12,[6,7,8,11,12,13,14,15,16,17])
add_freq_indx(13,[1,2,3,6,15,16,18,19,20])
add_freq_indx(14,[4,7,8,9,13,14])
add_freq_indx(15,[3,4,12,15,16])
add_freq_indx(16,[5,6,7,10,11,14])
add_freq_indx(17,[1,8,12,13,14,15])
add_freq_indx(18,[3,4,6])
add_freq_indx(19,[10,11])
add_freq_indx(20,[1,5])

def add_freq_indx_T(n,m):
    global temp
    try:
        for i in iter(m):
            freq_pair_T.setdefault(str(n)+'T'+str(i),temp)
            temp += 1
            if n == 0:
                freq_indx_T.append(i-2)
            else:
                freq_indx_T.append(i-2+24*n)
    except TypeError:
        freq_pair_T.setdefault(str(n)+'T'+str(m),temp)
        temp += 1
        if n == 0:
            freq_indx_T.append(m-2)
        else:
            freq_indx_T.append(n*24+m-2)

freq_indx_T = []
freq_pair_T = dict([])
add_freq_indx_T(1,[2,3,5,6,7,13,14])
add_freq_indx_T(2,[3,5,7,8,13,14,15,16])
add_freq_indx_T(3,[1,7,16])
add_freq_indx_T(4,9)

def Read_file(step=1,doc_num=5):
    model_num = 10000
    mode_S_num = 544
    mode_T_num = 119
    elected_num = [[] for i in range(doc_num+1)]
    for file_indx in range(1,doc_num+1):
        for i in range(model_num):
            filename_S = mineos_path+'/Normal_mode_s{0}_'.format(step)+str(file_indx)+'/Earth_'+str(i)+'_S'
            filename_T = mineos_path+'/Normal_mode_t{0}_'.format(step)+str(file_indx)+'/Earth_'+str(i)+'_T'
            a = np.genfromtxt(filename_S)
            b = np.genfromtxt(filename_T)
            if a.shape[0] == mode_S_num and b.shape[0] == mode_T_num:
                elected_num[file_indx].append(i)
    
    for file_indx in range(1,doc_num+1):
        file_name = './data/Inverse_sample_step{0}_{1}.txt'.format(step,file_indx)
        if file_indx == 1:
            model_syn = np.loadtxt(file_name)[elected_num[file_indx]]
        else:
            model_syn = np.concatenate([model_syn,np.loadtxt(file_name)[elected_num[file_indx]]],axis=0)
    
    #to concatenate the frequency data
    data_num = model_syn.shape[0]
    freq_S = np.zeros((data_num,mode_S_num))
    q0_S = np.zeros((data_num,mode_S_num))
    freq_T = np.zeros((data_num,mode_T_num))
    q0_T = np.zeros((data_num,mode_T_num))
    j = 0
    for file_indx in range(1,doc_num+1):
        for i in elected_num[file_indx]:
            filename_S = mineos_path+'/Normal_mode_s{0}_'.format(step)+str(file_indx)+'/Earth_'+str(i)+'_S'
            filename_T = mineos_path+'/Normal_mode_t{0}_'.format(step)+str(file_indx)+'/Earth_'+str(i)+'_T'
            a = np.genfromtxt(filename_S)
            b = np.genfromtxt(filename_T)
            freq_S[j] = a[:,4]*1000.
            freq_T[j] = b[:,4]*1000.
            q0_S[j] = a[:,7]
            q0_T[j]= b[:,7]
            j+=1

    obs_syn = np.concatenate([freq_S[:,freq_indx],freq_T[:,freq_indx_T]],axis=1)
    obs_max = (np.max(obs_syn,axis=0)).reshape(obs_syn.shape[1])
    obs_min = np.min(obs_syn,axis=0).reshape(obs_syn.shape[1])
    obs_inter = obs_max - obs_min
    obs_normal = (obs_syn - obs_min)/obs_inter*2.-1.
    
    np.savetxt('./data/new_obs_normal_{0}.txt'.format(step),obs_normal)
    np.savetxt('./data/new_model_syn_{0}.txt'.format(step),model_syn)
    np.savetxt('./data/new_obs_min_{0}.txt'.format(step),obs_min)
    np.savetxt('./data/new_obs_inter_{0}.txt'.format(step),obs_inter)
    

def Read_file_test(step=1,doc_num=1,model_num=100):
    mode_S_num = 544
    mode_T_num = 119
    elected_num = [[] for i in range(doc_num+1)]
    for file_indx in range(1,doc_num+1):
        for i in range(model_num):
            filename_S = mineos_path+'/Normal_mode_s{0}_'.format(step)+str(file_indx)+'_test/Earth_'+str(i)+'_S'
            filename_T = mineos_path+'/Normal_mode_t{0}_'.format(step)+str(file_indx)+'_test/Earth_'+str(i)+'_T'
            a = np.genfromtxt(filename_S)
            b = np.genfromtxt(filename_T)
            if a.shape[0] == mode_S_num and b.shape[0] == mode_T_num:
                elected_num[file_indx].append(i)
    
    for file_indx in range(1,doc_num+1):
        file_name = './data/Inverse_sample_step{0}_{1}_test.txt'.format(step,file_indx)
        if file_indx == 1:
            model_syn = np.loadtxt(file_name)[elected_num[file_indx]]
        else:
            model_syn = np.concatenate([model_syn,np.loadtxt(file_name)[elected_num[file_indx]]],axis=0)
    
    #to concatenate the frequency data
    data_num = model_syn.shape[0]
    freq_S = np.zeros((data_num,mode_S_num))
    q0_S = np.zeros((data_num,mode_S_num))
    freq_T = np.zeros((data_num,mode_T_num))
    q0_T = np.zeros((data_num,mode_T_num))
    j = 0
    for file_indx in range(1,doc_num+1):
        for i in elected_num[file_indx]:
            filename_S = mineos_path+'/Normal_mode_s{0}_'.format(step)+str(file_indx)+'_test/Earth_'+str(i)+'_S'
            filename_T = mineos_path+'/Normal_mode_t{0}_'.format(step)+str(file_indx)+'_test/Earth_'+str(i)+'_T'
            a = np.genfromtxt(filename_S)
            b = np.genfromtxt(filename_T)
            freq_S[j] = a[:,4]*1000.
            freq_T[j] = b[:,4]*1000.
            q0_S[j] = a[:,7]
            q0_T[j]= b[:,7]
            j+=1
    
    obs_syn = np.concatenate([freq_S[:,freq_indx],freq_T[:,freq_indx_T]],axis=1)
    obs_max = (np.max(obs_syn,axis=0)).reshape(obs_syn.shape[1])
    obs_min = np.min(obs_syn,axis=0).reshape(obs_syn.shape[1])
    obs_inter = obs_max - obs_min
    obs_normal = (obs_syn - obs_min)/obs_inter*2.-1.
    
    np.savetxt('./data/new_obs_normal_{0}_test.txt'.format(step),obs_normal)
    np.savetxt('./data/new_model_syn_{0}_test.txt'.format(step),model_syn)
    np.savetxt('./data/new_obs_min_{0}_test.txt'.format(step),obs_min)
    np.savetxt('./data/new_obs_inter_{0}_test.txt'.format(step),obs_inter)
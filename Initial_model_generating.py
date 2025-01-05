import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
import time
import seaborn as sns
from tools import *
import importlib
import Model_generator
importlib.reload(Model_generator)
from Model_generator import *
import argparse

parser = argparse.ArgumentParser(description="Generate the intial prior Earth model parameter dataset.")

parser.add_argument("Num",type=int,help="Total number of samples, unit:10k")

args = parser.parse_args()
num = args.Num

print(num)

#Reference Earth parameters and their prior range
#radiu
NN = 30
R = rad_p[-1]
rad_0 = np.copy(rad_p)
d_rad = np.zeros_like(rad_p)
rad_0[3] = R - 5169.5;  d_rad[3] = 5169.5 - 5129.5 #0->1->3
rad_0[11] = R - 2971;   d_rad[11] = 2971 - 2871 #1->5->11
rad_0[12] = R - 2761;   d_rad[12] = 2761 - 2721 #2->6->12
rad_0[21] = R - 700;    d_rad[21] = 700 - 640 #4->11
rad_0[23] = R - 430;    d_rad[23] = 430 - 370 #6->13
rad_0[24] = R - 240;    d_rad[24] = 240 - 200 #7->14->24
rad_ref = rad_0 + d_rad/2.

#velocity and density
rate = np.array([5.,5.,5.,5.,5.,5.,5.,5.,5.,5.,5.,5.,10.,10.,10.,10.,10.,10.,10.,10.,10.,10.,10.,20.,20.,15.,15.,0.,0.,0.])*0.01
rate2 = np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,0.,0.])*0.01*5
den_ref = np.zeros(NN)
vpv_ref = np.zeros(NN)
vsv_ref = np.zeros(NN)
vph_ref = np.zeros(NN)
vsh_ref = np.zeros(NN)
eta_ref = np.zeros(NN)

for i in range(NN):
    if i == 0:
        r1 = 0.0
        r2 = rad_p[0]
    else:
        r1 = rad_p[i-1]
        r2 = rad_p[i]
    r = (r1+r2)/2.
    x = r/6371.
    den_ref[i] = den_p[i,0] + den_p[i,1]*x + den_p[i,2]*x*x + den_p[i,3]*x*x*x
    vpv_ref[i] = vpv_p[i,0] + vpv_p[i,1]*x + vpv_p[i,2]*x*x + vpv_p[i,3]*x*x*x
    vsv_ref[i] = vsv_p[i,0] + vsv_p[i,1]*x + vsv_p[i,2]*x*x + vsv_p[i,3]*x*x*x
    vph_ref[i] = vph_p[i,0] + vph_p[i,1]*x + vph_p[i,2]*x*x + vph_p[i,3]*x*x*x
    vsh_ref[i] = vsh_p[i,0] + vsh_p[i,1]*x + vsh_p[i,2]*x*x + vsh_p[i,3]*x*x*x
    eta_ref[i] = eta_p[i,0] + eta_p[i,1]*x

d_den = den_ref*rate
d_vpv = vpv_ref*rate
d_vsv = vsv_ref*rate
d_vph = vph_ref*rate
d_vsh = vsh_ref*rate
d_eta = eta_ref*rate2

den_ref = den_p[:,0]
vpv_ref = vpv_p[:,0]
vsv_ref = vsv_p[:,0]
vph_ref = vph_p[:,0]
vsh_ref = vsh_p[:,0]
eta_ref = eta_p[:,0]

# save 
r_index = [3,11,12,21,23,24]
vs_index = list((set(range(NN))-set([4,5,6,7,8,9,10,11]))-set([27,28,29]))#19
ot_index = list((set(range(NN))-set([27,28,29])))#27
d_model = np.concatenate([d_rad[r_index],d_den[ot_index],d_vpv[ot_index],d_vsv[vs_index],
                            d_vph[ot_index],d_vsh[vs_index],d_eta[ot_index]],axis=0)
ref_model = np.concatenate([rad_ref[r_index],den_ref[ot_index],vpv_ref[ot_index],vsv_ref[vs_index],
                            vph_ref[ot_index],vsh_ref[vs_index],eta_ref[ot_index]],axis=0)
np.savetxt('./data/d_model.txt',d_model)
np.savetxt('./data/ref_model.txt',ref_model)

depth=6371-rad_p

#Generate prior model samples

"""
We adopt a simple parallel strategy for computation. 
The total number of samples is given by num * Sample_N. 
You can adjust the parameter num to control the number of parallel computation tasks, thereby improving computational efficiency.
Additionally, you can change Sample_N to represent the number of samples processed in each task.
"""
Sample_N = 10000
for seed in range(1,num+1):
    np.random.seed(seed)
    mass_ref = 5.9733e24
    d_mass = 0.0090e24
    inertia_ref = 8.018e37
    d_inertia = 0.012e37
    NN = 30
    
    rad_save = np.zeros((Sample_N,NN))
    den_save = np.zeros((Sample_N,NN))
    vpv_save = np.zeros((Sample_N,NN))
    vsv_save = np.zeros((Sample_N,NN))
    vph_save = np.zeros((Sample_N,NN))
    vsh_save = np.zeros((Sample_N,NN))
    eta_save = np.zeros((Sample_N,NN))
    qm_save  = np.zeros((Sample_N,NN))
    qk_save  = np.zeros((Sample_N,NN))
    den_grad_save = np.zeros((Sample_N,NN))
    vpv_grad_save = np.zeros((Sample_N,NN))
    vsv_grad_save = np.zeros((Sample_N,NN))
    vph_grad_save = np.zeros((Sample_N,NN))
    vsh_grad_save = np.zeros((Sample_N,NN))
    
    #mass_save = np.zeros(Sample_N)
    #iner_save = np.zeros(Sample_N)
    
    aa = time()
    
    for temp in range(Sample_N):
        
        rand_rad = np.random.uniform(-1,1,NN)
        rand_den = np.random.uniform(-1,1,NN)
        rand_vpv = np.random.uniform(-1,1,NN)
        rand_vsv = np.random.uniform(-1,1,NN)
        rand_vph = np.random.uniform(-1,1,NN)
        rand_vsh = np.random.uniform(-1,1,NN)
        rand_eta = np.random.uniform(-1,1,NN)
        
        rad_fin = np.copy(rad_p.reshape(NN,1))
        den_fin = np.copy(den_p)
        vpv_fin = np.copy(vpv_p)
        vsv_fin = np.copy(vsv_p)
        qm_fin  = np.copy(qm_p)
        qk_fin  = np.copy(qk_p)
        vph_fin = np.copy(vph_p)
        vsh_fin = np.copy(vsh_p)
        eta_fin = np.copy(eta_p)
        
        rad_test = np.copy(rad_ref + rand_rad*d_rad/2.)
        den_test = np.copy(den_ref + rand_den*d_den/2.)
        vpv_test = np.copy(vpv_ref + rand_vpv*d_vpv/2.)
        vsv_test = np.copy(vsv_ref + rand_vsv*d_vsv/2.)
        vph_test = np.copy(vph_ref + rand_vph*d_vph/2.)
        vsh_test = np.copy(vsh_ref + rand_vsh*d_vsh/2.)
        eta_test = np.copy(eta_ref + rand_eta*d_eta/2.)
        
        rad_fin[r_index,0] = rad_test[r_index]
        den_fin[ot_index,0] = den_test[ot_index]
        vpv_fin[ot_index,0] = vpv_test[ot_index]
        vsv_fin[vs_index,0] = vsv_test[vs_index]
        vph_fin[ot_index,0] = vph_test[ot_index]
        vsh_fin[vs_index,0] = vsh_test[vs_index]
        eta_fin[ot_index,0] = eta_test[ot_index]
        #Adjust the density to ensure that the total mass and moment of inertia match the observed values.
        #Calculate the total mass and adjust the density of the last three layers.
        mass = Mass(rad_fin,den_fin)
        mass_diff = mass_ref-mass
        if np.abs(mass_diff) > d_mass:
            mass = Mass(rad_fin,den_fin)
            mass_diff = mass_ref-mass
            mass_rate = mass_diff/mass
            d_rho_rate = mass_rate*den_fin[:-3,0]/d_den[:-3]*2.
            rand_den[:-3] += d_rho_rate
            den_fin[:,0] = den_ref + rand_den*d_den/2.
        
            
        #Calculate the moment of inertia and adjust the density of layers 4-6.
        inertia = Inertia(rad_fin,den_fin)
        iner_diff = inertia_ref-inertia
        if np.abs(iner_diff) > d_inertia:
            r1 = rad_test[NN-6]
            r2 = rad_test[NN-4]
            N  = 100
            dr = (r2-r1)/N
            iner_temp = 0.0
            for i in range(N):
                r = r1+i*dr
                dm = 4./3.*np.pi*(np.power(r+dr,3)-np.power(r,3))
                iner_temp += 2./3.*dm*r*r
            iner_temp = iner_temp*1e18

            for i in range(4,6):
                d_rho1 = iner_diff/iner_temp
                d_rho_rate1 = d_rho1/d_den[NN-i]*2.
                rand_den[NN-i] += d_rho_rate1
                den_fin[NN-i,0] = den_ref[NN-i] + rand_den[NN-i]*d_den[NN-i]/2.
                
        #Calculate the total mass and adjust the overall density.
        mass = Mass(rad_fin,den_fin)
        mass_diff = mass_ref-mass
        if np.abs(mass_diff) > d_mass:
            r1 = 0.0
            r2 = rad_test[11]
            N = 100
            dr = (r2-r1)/N
            mass_temp = 0.0
            for i in range(N):
                r = r1+i*dr
                mass_temp += 4./3.*np.pi*(np.power(r+dr,3)-np.power(r,3))
            mass_temp = mass_temp*1e12
            d_rho = mass_diff/mass_temp
    
            for i in range(12):
                d_rho_rate = d_rho/d_den[i]*2.
                rand_den[i] += d_rho_rate
                den_fin[i,0] = den_ref[i] + rand_den[i]*d_den[i]/2.
        rad_save[temp] = rand_rad
        den_save[temp] = rand_den
        vpv_save[temp] = rand_vpv
        vsv_save[temp] = rand_vsv
        vph_save[temp] = rand_vph
        vsh_save[temp] = rand_vsh
        eta_save[temp] = rand_eta
        #mass_save[temp] = Mass(rad_fin,den_fin)
        #iner_save[temp] = Inertia(rad_fin,den_fin)
        file_name = mineos_path+'/Earth_model_step0_{0}/Earth_{1}.txt'.format(seed,temp)
        model_format(rad_fin.reshape([NN]),den_fin,vpv_fin,vsv_fin,qk_fin,qm_fin,vph_fin,vsh_fin,eta_fin,file_name)
    bb = time()
    cc = bb - aa
    #print(cc)
    save_np = np.concatenate([rad_save[:,r_index],den_save[:,ot_index],vpv_save[:,ot_index],vsv_save[:,vs_index],
                              vph_save[:,ot_index],vsh_save[:,vs_index],eta_save[:,ot_index]],axis=1)
    np.savetxt('./data/Inverse_sample_step0_{0}.txt'.format(seed),save_np)
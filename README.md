# Iterative-Normalizing-Flow-Model
INFM is a new method for Bayesian inversion that utilizes an iterative strategy. In this repo, we use Earth's free oscillation normal modes as observations to invert the 1D structure of the Earth.


## Requirement
- torch >= 2.2.1
- Python >= 3.10.13
- Numpy >= 1.26.4
- Scipy >=1.11.4
- matplotlib >= 3.8.0  
- Jupyter notebook
- gc
- tqdm
- FrEIA
- Mineos

We recommend using this software on a Linux system (e.g., Ubuntu). The download for FrEIA can be referenced from the tutorial at https://github.com/vislearn/FrEIA.git. We have made some modifications to Mineos. However, don’t worry, just follow the instructions under 'Installation' below to complete the setup. Other dependencies can be installed using Anaconda or pip.

## Installation
We slightly modified the Mineos code, so it needs to be recompiled.
```
cd ./Mineos
chmod u+x ./install.sh
./install.sh
```

## Inversion Workflow
This software is mainly divided into two sections: validation experiments and actual normal mode inversion. Each section consists of two stages: 'Training' and 'Plotting.' The training stage primarily uses the INFM method for multiple iterations. If you wish to reproduce this stage, it is recommended to use Jupyter Notebook to run the corresponding code. The 'Plotting' stage is mainly for presenting results and outputting numerical data. We have already included pre-trained data and networks in the 'data' and 'net' directories, so readers can skip the training stage and directly use this part of the code to view the relevant results.

### Validation Experiments
### Training 
Use Jupyter Notebook to Open “/Validation_training.ipynb”. Execute the blocks sequentially in the notebook to ensure all dependencies and steps are correctly completed.

Files Saved After Completion
- './net/Forward_net.pth':A flow-based model(FBM) used for approximating the forward mapping.
- './net/Obs_normal_0_val.pth': A flow-based model as the result of traditional FBM method.
- './net/Obs_normal_i_val.pth','./net/Mod_normal_i_val.pth','./net/Inverse_flow_si_val.pth'(i=1~4): the results of INFM method.
- './data/cross_entorpyi_j.txt':the cross-entropys
- './data/obs_fixi': Normalized real observation values of each step.

### Plotting
Run the following python file:
```
python3 ./Validation_plotting.py
```
or you can use Jupyter Notebook to Run "./Validation_plotting.ipynb".

The results of validation('KL.png','validation_Lm.png','validation_std_diff.png') are shown in 'fig'.For a detailed explanation of the image, please refer to the corresponding literature of this software. Additionally, the module outputs posterior probability samples at 10k, 20k, 30k, and 40k epochs in the "./data/Validation" folder, with file numbers corresponding to (0, 1, 2, 3, 4). The filename "Traditional" refers to the traditional method, and "Target.txt" refers to the target distribution.

### Actual Normal Mode Inversion
### Training
Use Jupyter Notebook to Open “/Real_inversion_training.ipynb”. Execute the blocks sequentially in the notebook to ensure all dependencies and steps are correctly completed.

“/Initial_model_generating.sh” or "/Initial_model_generating.ipynb" is used to generate the prior training samples. For the convenience of the readers, we have provided the pre-trained dataset. For more details, please refer to "/Real_inversion_training.ipynb".

At each step, you need to execute some bash codes to run the Mineos forward modeling. It is recommended that readers run the commands directly in the “./Mineos/DEMO/Earth” directory, so the lines "cd ./Mineos/DEMO/Earth" in the code can be omitted.

Files Saved After Completion
- './net/Obs_normal_i_real.pth'(i=0~3): the encoder-decoder of observations $f^d_i$
- './net/Mod_normal_i_real.pth' (i=0~3):the encoder-decoder of models $f^m_i$
- './net/Inverse_flow_si_real.pth': the main flow model $f_i$

It is worth noting that after each iteration step, a small batch of forward modeling is used to determine $\alpha$. The code provided by the module allows readers to generate posterior probability samples on their own.

### Plotting
Run the following python file:
```
python3 ./Real_inversion_plotting.py
```
or you can use Jupyter Notebook to Run "./Real_inversion_plotting.ipynb".

The deviation of observations are saved as "./fig/Real_std_diff.png", and the posterior pdf results are saved as "./fig/density.png","./fig/xi_mantle.png", './fig/eta.png' and './fig/phi_mantle.png'

The posterior samples for radius , density, vpv, vsv, vph, vsh and eta are saved in "./data/Real/"














 

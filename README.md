# Iterative-Normalizing-Flow-Model
INFM is a new method for Bayesian inversion that utilizes an iterative strategy. In this repo, we use Earth's free oscillation normal modes as observations to invert the 1D structure of the Earth.


## Requirement
- torch >= 2.2.1
- Python >= 3.10.13
- Numpy >= 1.26.4
- Scipy >=1.11.4
- matplotlib >= 3.8.0
- minos  
- FrEIA

## Installation
We slightly modified the Mineos code, so it needs to be recompiled.
```
cd ./Mineos
chmod u+x ./install.sh
./install.sh
```

## Inversion Workflow
- use the script "Model_generator.ipynb" to generate prior Earth Model and corresponding Observation
- use the script "Flow.ipynb" to implement Bayesian Inversion using INFM method
- use the script "plot.ipynb" to visualize the inversion results.
 

#!/bin/sh 

n=$1
i=$2

#============ get the file name =========== 
export PATH=$HOME/cig/bin:$PATH
modelpath="$(pwd)/Earth_model_step${n}_${i}_test" 
for file_a in ${modelpath}/*
do 
model=`basename $file_a .txt` 
#=========================================================
# 1. run minos_bran program for fundamental S  mode,
# where,  n=0, 0 < f <  0.2 Hz,
#
echo "Step 1:  minos_bran runs for S modes ....................."

time minos_bran << EOF
$(pwd)/Earth_model_step${n}_${i}_test/${model}.txt
$(pwd)/Normal_mode_step${n}_sph_${i}_test/${model}_S
3 1 26 0.0 50.0 0 20
EOF

echo "Step 2:  minos_bran runs for T modes ....................."

time minos_bran << EOF
$(pwd)/Earth_model_step${n}_${i}_test/${model}.txt
$(pwd)/Normal_mode_step${n}_tor_${i}_test/${model}_T
2 1 24 0.0 50.0 0 4
EOF

done
exit


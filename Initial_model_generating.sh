#!/bin/bash
num=$1
cd ./Mineos/DEMO/Earth
chmod u+x ./make_folder.sh
chmod u+x ./RUN_MIEOS.sh
./make_folder.sh 0 $num
python3 ./Initial_model_generating.py "$num"
for ((i=1;i<=num;i++)); do
    ./RUN_MINEOS.sh 0 $i
done

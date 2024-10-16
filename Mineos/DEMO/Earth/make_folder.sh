#!/bin/bash

echo $(pwd)

if [ -z "$1" ]; then
  echo "please supply an int: n"
  exit 1
fi

n=$1

#change '10' if you want to change the num of folders
for i in {1..10}
do
  folder_name1="./Earth_model_step${n}_${i}"
  folder_name2="./Normal_mode_step${n}_sph_${i}"
  folder_name3="./Normal_mode_step${n}_tor_${i}"
  
  if [ -d "$folder_name1" ]; then
    echo "folder $folder_name1 exist，pass..."
  else
    mkdir "$folder_name1"
    echo "$folder_name1 is creating"
  fi
  
  if [ -d "$folder_name2" ]; then
    echo "folder $folder_name2 exist，pass..."
  else
    mkdir "$folder_name2"
    echo "$folder_name2 is creating"
  fi
  
  if [ -d "$folder_name3" ]; then
    echo "folder $folder_name3 exist，pass..."
  else
    mkdir "$folder_name3"
    echo "$folder_name3 is creating"
  fi
done


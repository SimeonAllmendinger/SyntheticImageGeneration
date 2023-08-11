#!/bin/bash

if [ ! -d "$TMP/SyntheticImageGeneration/" ]; then
    mkdir $TMP/SyntheticImageGeneration/
fi

if [ ! -d "$TMP/SyntheticImageGeneration/data/" ]; then
    mkdir $TMP/SyntheticImageGeneration/data/
fi

if [ ! -d "$TMP/SyntheticImageGeneration/src/" ]; then
    mkdir $TMP/SyntheticImageGeneration/src/
fi

# Extract compressed input data files on local SSD
## Assets
if [ ! -d "$TMP/SyntheticImageGeneration/src/assets/" ]; then
    tar -C $TMP/ -xvzf $(ws_find data-ssd)/SyntheticImageGeneration/src/assets.tgz
fi

if [ ! -d "$TMP/SyntheticImageGeneration/src/components/" ]; then
    tar -C $TMP/ -xvzf $(ws_find data-ssd)/SyntheticImageGeneration/src/components.tgz
fi

## Configs
if [ ! -d "$TMP/SyntheticImageGeneration/configs/" ]; then
    tar -C $TMP/ -xvzf $(ws_find data-ssd)/SyntheticImageGeneration/configs.tgz
fi

## Data
if [ ! -d "$TMP/SyntheticImageGeneration/data/CholecT45/" ]; then
    tar -C $TMP/ -xvzf $(ws_find data-ssd)/SyntheticImageGeneration/data/CholecT45.tgz
fi

## Virtual Environment
if [ ! -d "$TMP/SyntheticImageGeneration/venv/" ]; then
    tar -C $TMP/ -xvzf $(ws_find data-ssd)/SyntheticImageGeneration/venv.tgz
fi

# Create results and scripts directory
mkdir $TMP/SyntheticImageGeneration/scripts
mkdir $TMP/SyntheticImageGeneration/results
cd $TMP/SyntheticImageGeneration/results
mkdir $TMP/SyntheticImageGeneration/results/training

# Activate virtual environment (venv)
cd $TMP/SyntheticImageGeneration/
source ./venv/bin/activate

# Install yq if it is not already installed
if ! command -v yq &> /dev/null
then
    echo "yq not found. Installing..."
    sudo wget https://github.com/mikefarah/yq/releases/download/v4.6.3/yq_linux_amd64 -O /usr/bin/yq
    sudo chmod +x /usr/bin/yq
fi

# Load the config file
config_file="./configs/models/config_rendevouz.yaml"

# Extract the values of the variables in the config using yq
## Type
train=$(yq e '.type.train' $config_file)
test=$(yq e '.type.test' $config_file)

## Data
dataset_variant=$(yq e '.data.dataset_variant' $config_file)
data_dir=$(yq e '.data.data_dir' $config_file)
kfold=$(yq e '.data.kfold' $config_file)
image_width=$(yq e '.data.image_width' $config_file)
image_height=$(yq e '.data.image_height' $config_file)
test_ckpt=$(yq e '.data.test_ckpt' $config_file)
pretrain_dir=$(yq e '.data.pretrain_dir' $config_file)

## Params
use_ln=$(yq e '.params.use_ln' $config_file)
val_interval=$(yq e '.params.val_interval' $config_file)
batch=$(yq e '.params.batch' $config_file)
epochs=$(yq e '.params.epochs' $config_file)
version=$(yq e '.params.version' $config_file)
gpu=$(yq e '.params.gpu' $config_file)
accelerate=$(yq e '.params.accelerate' $config_file)

# Use the variables to start the script
#if [ "$test" = true ] ;
#then
#    echo "----- TESTING -----"
#    ./venv/bin/python3 src/components/rendezvous/pytorch/run.py -e --data_dir=$data_dir --dataset_variant=$dataset_variant --use_ln --kfold $kfold --batch $batch --version=$version --test_ckpt=$test_ckpt
#fi

# Use the variables to start the script
if [ "$train" = true ]
then
    echo "----- TRAINING -----"
    ./venv/bin/python3 \
        src/components/rendezvous/pytorch/run.py -t -e\
        --data_dir=$data_dir \
        --dataset_variant=$dataset_variant \
        --kfold $kfold \
        --batch $batch \
        --version=$version \
        --pretrain_dir=$pretrain_dir \
        --image_width=$image_width \
        --image_height=$image_height \
        --val_interval=$val_interval \
        --epochs=$epochs \
        --version=$version \
        --gpu=$gpu 
        #--accelerate=$accelerate
        #--use_ln \
fi
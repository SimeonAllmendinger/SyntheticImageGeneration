#!/bin/bash

# Start session
screen -S RENDEVOUZ

cd /home/stud01/SyntheticImageGeneration
source /home/stud01/SyntheticImageGeneration/venv/bin/activate

# Install yq if it is not already installed
if ! command -v yq &> /dev/null
then
    echo "yq not found. Installing..."
    sudo wget https://github.com/mikefarah/yq/releases/download/v4.6.3/yq_linux_amd64 -O /usr/bin/yq
    sudo chmod +x /usr/bin/yq
fi

# Load the config file
config_file="./configs/config_rendevouz.yaml"

# Extract the values of the variables in the config using yq
## Type
train=$(yq e '.type.train' $config_file)
test=$(yq e '.type.test' $config_file)
## Data
dataset_variant=$(yq e '.data.dataset_variant' $config_file)
data_dir=$(yq e '.data.data_dir' $config_file)
test_ckpt=$(yq e '.data.test_ckpt' $config_file)
pretrain_dir=$(yq e '.data.pretrain_dir' $config_file)
## Params
use_ln=$(yq e '.params.use_ln' $config_file)
kfold=$(yq e '.params.kfold' $config_file)
batch=$(yq e '.params.batch' $config_file)
version=$(yq e '.params.version' $config_file)

# Use the variables in your script
if $test #&& $use_ln
then
    echo "----- TESTING -----"
    python3 src/components/rendezvous/pytorch/run.py -e --data_dir=$data_dir --dataset_variant=$dataset_variant --use_ln --kfold $kfold --batch $batch --version=$version --test_ckpt=$test_ckpt
fi

# Use the variables in your script
if $train #&& $use_ln
then
    echo "----- TRAINING -----"
    python3 src/components/rendezvous/pytorch/run.py -t --data_dir=$data_dir --dataset_variant=$dataset_variant --use_ln --kfold $kfold --batch $batch --version=$version --pretrain_dir=$pretrain_dir
fi
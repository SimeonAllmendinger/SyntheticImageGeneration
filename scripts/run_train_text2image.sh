#!/bin/bash

# Start session
# screen -S RAY_TUNING

#./scripts/extract_data.sh

#
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
## Data
if [ ! -d "$TMP/SyntheticImageGeneration/data/Cholec80/" ]; then
    tar -C $TMP/ -xvzf $(ws_find data-ssd)/SyntheticImageGeneration/data/Cholec80.tgz
fi

if [ ! -d "$TMP/SyntheticImageGeneration/data/CholecSeg8k/" ]; then
    tar -C $TMP/ -xvzf $(ws_find data-ssd)/SyntheticImageGeneration/data/CholecSeg8k.tgz
fi

if [ ! -d "$TMP/SyntheticImageGeneration/data/CholecT45/" ]; then
    tar -C $TMP/ -xvzf $(ws_find data-ssd)/SyntheticImageGeneration/data/CholecT45.tgz
fi

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

## Scripts
if [ ! -d "$TMP/SyntheticImageGeneration/scripts/" ]; then
    tar -C $TMP/ -xvzf $(ws_find data-ssd)/SyntheticImageGeneration/scripts.tgz
fi

## Cache
if [ ! -d "$TMP/.cache/huggingface/" ]; then
    tar -C $TMP/ -xvzf $(ws_find data-ssd)/cache_config.tgz
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

# Start parameter tuning
#accelerate launch ./src/components/imagen/training/train_imagen.py --path_data_dir=$TMP/SyntheticImageGeneration/
./venv/bin/python3 ./src/components/imagen/training/train_imagen.py --path_data_dir=$TMP/SyntheticImageGeneration/
#./venv/bin/python3 ./src/components/dalle2/model/build_dalle2.py --path_data_dir=$TMP/SyntheticImageGeneration/

cp -r $TMP/SyntheticImageGeneration/src/assets/elucidated_imagen/models $HOME/SyntheticImageGeneration/src/assets/elucidated_imagen/
#cp -r $TMP/SyntheticImageGeneration/src/assets/dalle2/models $HOME/SyntheticImageGeneration/src/assets/dalle2/
#cp -r $TMP/SyntheticImageGeneration/src/assets/data/CholecT45/clip_embeds $HOME/SyntheticImageGeneration/src/assets/data/CholecT45/
#cp -r $TMP/SyntheticImageGeneration/src/assets/data/CholecT45/clip_tokens $HOME/SyntheticImageGeneration/src/assets/data/CholecT45/
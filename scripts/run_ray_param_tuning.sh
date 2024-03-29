#!/bin/bash

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
if [ ! -d "$TMP/SyntheticImageGeneration/data/Cholec80/" ]; then
    tar -C $TMP/ -xvzf $(ws_find data-ssd)/SyntheticImageGeneration/data/Cholec80.tgz
fi

if [ ! -d "$TMP/SyntheticImageGeneration/data/CholecSeg8k/" ]; then
    tar -C $TMP/ -xvzf $(ws_find data-ssd)/SyntheticImageGeneration/data/CholecSeg8k.tgz
fi

if [ ! -d "$TMP/SyntheticImageGeneration/data/CholecT45/" ]; then
    tar -C $TMP/ -xvzf $(ws_find data-ssd)/SyntheticImageGeneration/data/CholecT45.tgz
fi

if [ ! -d "$TMP/SyntheticImageGeneration/src/assets/" ]; then
    tar -C $TMP/ -xvzf $(ws_find data-ssd)/SyntheticImageGeneration/src/assets.tgz
fi

# Activate virtual environment (venv)
source ./venv/bin/activate

# Start parameter tuning
./venv/bin/python3 ./src/components/imagen/training/ray_tuning.py --path_data_dir=$TMP/SyntheticImageGeneration/

#cp -r $TMP/SyntheticImageGeneration/src/assets/ $HOME/SyntheticImageGeneration/src/assets/
#!/bin/bash

# Start session
# screen -S RAY_TUNING

./scripts/extract_data.sh

# Activate virtual environment (venv)
# cd /home/stud01/SyntheticImageGeneration
source ./venv/bin/activate

# Start parameter tuning
./venv/bin/python3 ./src/components/imagen/training/train_imagen.py --path_data_dir=$TMP/SyntheticImageGeneration/

cp -r $TMP/SyntheticImageGeneration/src/assets/ $HOME/SyntheticImageGeneration/src/assets/
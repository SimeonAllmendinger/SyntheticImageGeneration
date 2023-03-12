#!/bin/bash

# Activate virtual environment (venv)
cd /home/stud01/SyntheticImageGeneration
source /home/stud01/SyntheticImageGeneration/venv/bin/activate

# Start data preparation
./venv/bin/python3 ./src/components/imagen/data_manager/preparation/segment_cholecSeg8k.py
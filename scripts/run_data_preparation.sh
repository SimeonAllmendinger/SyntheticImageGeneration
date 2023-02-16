#!/bin/bash

# Activate virtual environment (venv)
cd /home/stud01/SyntheticImageGeneration
source /home/stud01/SyntheticImageGeneration/venv/bin/activate

# Start data preparation
/home/stud01/SyntheticImageGeneration/venv/bin/python3 /home/stud01/SyntheticImageGeneration/src/components/imagen/data_manager/preparation/segment_cholecSeg8k.py
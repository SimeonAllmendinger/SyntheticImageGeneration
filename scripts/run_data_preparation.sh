#!/bin/bash

# Activate virtual environment (venv)
source ./venv/bin/activate

# Start data preparation
./venv/bin/python3 ./src/components/imagen/data_manager/preparation/segment_cholecSeg8k.py
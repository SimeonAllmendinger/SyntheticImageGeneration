#!/bin/bash

# Start session
screen -S RAY_TUNING

# Activate virtual environment (venv)
cd /home/stud01/SyntheticImageGeneration
source /home/stud01/SyntheticImageGeneration/venv/bin/activate

# Detach the screen
screen -X -d eval "stuff 'screen -X detach'"

# Start parameter tuning
/home/stud01/SyntheticImageGeneration/venv/bin/python3 /home/stud01/SyntheticImageGeneration/src/components/imagen/training/ray_tuning.py
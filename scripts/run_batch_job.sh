#!/bin/bash

if [ ! -d "$(ws_find data-ssd)/SyntheticImageGeneration/" ]; then
    mkdir $(ws_find data-ssd)/SyntheticImageGeneration/
fi

if [ ! -d "$(ws_find data-ssd)/SyntheticImageGeneration/data/" ]; then
    mkdir $(ws_find data-ssd)/SyntheticImageGeneration/data/
fi

if [ ! -d "$(ws_find data-ssd)/SyntheticImageGeneration/src/" ]; then
    mkdir $(ws_find data-ssd)/SyntheticImageGeneration/src/
fi

# Create tar files on temp drive
cd 
tar -cvzf $(ws_find data-ssd)/SyntheticImageGeneration/sripts.tgz SyntheticImageGeneration/scripts/
#tar -cvzf $(ws_find data-ssd)/SyntheticImageGeneration/src/assets.tgz SyntheticImageGeneration/src/assets/
tar -cvzf $(ws_find data-ssd)/SyntheticImageGeneration/src/components.tgz SyntheticImageGeneration/src/components/
#tar -cvzf $(ws_find data-ssd)/SyntheticImageGeneration/venv.tgz SyntheticImageGeneration/venv/
tar -cvzf $(ws_find data-ssd)/SyntheticImageGeneration/configs.tgz SyntheticImageGeneration/configs/
#tar -cvzf $(ws_find data-ssd)/cache_config.tgz .cache/huggingface/
#tar -cvzf $(ws_find data-ssd)/SyntheticImageGeneration/data/CholecT45.tgz SyntheticImageGeneration/data/CholecT45/
#tar -cvzf $(ws_find data-ssd)/SyntheticImageGeneration/data/Cholec80.tgz SyntheticImageGeneration/data/Cholec80/
#tar -cvzf $(ws_find data-ssd)/SyntheticImageGeneration/data/CholecSeg8k.tgz SyntheticImageGeneration/data/CholecSeg8k/


cd SyntheticImageGeneration
source venv/bin/activate

# Install yq if it is not already installed
if ! command -v yq &> /dev/null
then
    echo "yq not found. Installing..."
    wget https://github.com/mikefarah/yq/releases/download/v4.6.3/yq_linux_amd64 -O ./venv/bin/yq
    chmod +x ./venv/bin/yq
fi

# Load the config file
config_file="configs/utils/config_SLURM_jobs.yaml"

# Extract the values of the variables in the config using yq
## Type
time=$(yq e '.args.time' $config_file)
nodes=$(yq e '.args.nodes' $config_file)
ntasks=$(yq e '.args.ntasks' $config_file)
ntasks_per_node=$(yq e '.args.ntasks_per_node' $config_file)
cpus_per_task=$(yq e '.args.cpus_per_task' $config_file)
mem_per_gpu=$(yq e '.args.mem_per_gpu' $config_file)
mail_type=$(yq e '.args.mail_type' $config_file)
mail_user=$(yq e '.args.mail_user' $config_file)
output=$(yq e '.args.output' $config_file)
error=$(yq e '.args.error' $config_file)
job_name=$(yq e '.args.job_name' $config_file)
partition=$(yq e '.args.partition' $config_file)
gres=$(yq e '.args.gres' $config_file)
job=$(yq e '.args.job' $config_file)

if [[ $job == 'salloc' ]]; then
    # Actions to be performed when $job equals 'salloc'
    echo "salloc"
    salloc --partition=$partition --gres=$gres --time=$time --nodes=$nodes --ntasks=$ntasks --mem-per-gpu=$mem_per_gpu --ntasks-per-node=$ntasks_per_node --cpus-per-task=$cpus_per_task
    
else
    # Actions to be performed when $job is not equal to 'salloc'
    echo "sbatch"
    sbatch --partition=$partition --gres=$gres --time=$time --nodes=$nodes --ntasks=$ntasks --mem-per-gpu=$mem_per_gpu --ntasks-per-node=$ntasks_per_node --mail-type=$mail_type --mail-user=$mail_user --output=$output --error=$error --cpus-per-task=$cpus_per_task --job-name=$job_name --reservation=$reservation $job 
    
fi

squeue
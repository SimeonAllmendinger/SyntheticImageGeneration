#!/bin/bash

cd 
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
config_file="/home/kit/stud/uerib/SyntheticImageGeneration/configs/config_bwUnicluster_jobs.yaml"

# Extract the values of the variables in the config using yq
## Type
time=$(yq e '.args.time' $config_file)
nodes=$(yq e '.args.nodes' $config_file)
ntasks=$(yq e '.args.ntasks' $config_file)
ntasks_per_node=$(yq e '.args.ntasks_per_node' $config_file)
cpus_per_task=$(yq e '.args.cpus_per_task' $config_file)
mem=$(yq e '.args.mem' $config_file)
mem_per_cpu=$(yq e '.args.mem_per_cpu' $config_file)
mem_per_gpu=$(yq e '.args.mem_per_gpu' $config_file)
cpu_per_gpu=$(yq e '.args.cpu_per_gpu' $config_file)
mail_type=$(yq e '.args.mail_type' $config_file)
mail_user=$(yq e '.args.mail_user' $config_file)
output=$(yq e '.args.output' $config_file)
error=$(yq e '.args.error' $config_file)
job_name=$(yq e '.args.job_name' $config_file)
partition=$(yq e '.args.partition' $config_file)
gres=$(yq e '.args.gres' $config_file)
job=$(yq e '.args.job' $config_file)

sbatch --partition=$partition --gres=$gres --time=$time --nodes=$nodes --ntasks=$ntasks --mem-per-gpu=$mem_per_gpu --ntasks-per-node=$ntasks_per_node --mail-type=$mail_type --mail-user=$mail_user --output=$output --error=$error --cpus-per-task=$cpus_per_task --job-name=$job_name --reservation=$reservation $job 
    #--cpu-per-gpu=$cpu_per_gpu  --mem-per-cpu=$mem_per_cpu  --mem=$mem 

squeue
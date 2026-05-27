#!/bin/bash

#SBATCH --job-name=flow_se        # Job name
#SBATCH --partition=ai              # Partition name (from your sinfo)
#SBATCH --gres=gpu:4                # Request 4 GPUs (generic resource)
#SBATCH --cpus-per-task=32           # Number of CPU cores per task (adjust as needed)

# load modules or conda environments here
source ~/.bashrc
conda activate eigenet

######## Build Experiment Environment ###########
# file dir
exp_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

echo work_dir: $work_dir

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8
export SWANLAB_API_KEY="o9QXwNCi5te7YakofmbxO"
export HF_ENDPOINT=https://hf-mirror.com

# export OMP_NUM_THREADS=18

######## Set Experiment Configuration ###########
exp_config="$exp_dir/EigeNet.json"
exp_name="debug"

######## Train Model ###########

accelerate launch \
    --main_process_port 13566 \
    --mixed_precision="bf16" \
    "${work_dir}"/train.py \
    --config=$exp_config \
    --exp_name=$exp_name \
    --log_level debug \
    --dataloader_seed 6006 \
    #--resume
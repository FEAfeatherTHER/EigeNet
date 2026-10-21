#!/bin/bash
set -euo pipefail

# Reuse an existing salloc allocation; this script never submits a new job.
# Usage: bash egs/rir/EigeNet/EigeNet.sh JOB_ID NUM_GPUS [train.py arguments]
if (( $# < 2 )); then
    echo "Usage: $0 JOB_ID NUM_GPUS [train.py arguments]" >&2
    exit 2
fi
job_id="$1"
num_processes="$2"
shift 2
if [[ ! "$job_id" =~ ^[0-9]+$ || ! "$num_processes" =~ ^[1-9][0-9]*$ ]]; then
    echo "JOB_ID and NUM_GPUS must be positive integers" >&2
    exit 2
fi

exp_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
work_dir="$(cd -- "$exp_dir/../../.." && pwd)"
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate rir
export WORK_DIR="$work_dir"
export PYTHONPATH="$work_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONIOENCODING=UTF-8
export SWANLAB_API_KEY="o9QXwNCi5te7YakofmbxO"
export HF_ENDPOINT=https://hf-mirror.com

# Each invocation gets its own output directory; existing results are not reused.
exp_name="rebuttal_${job_id}_$(date +%Y%m%d_%H%M%S)_$$"
launch_args=(--num_processes "$num_processes" --num_machines 1 --machine_rank 0
             --mixed_precision bf16 --main_process_port "${MAIN_PROCESS_PORT:-13566}")
if (( num_processes > 1 )); then
    launch_args+=(--multi_gpu)
fi

cd -- "$work_dir"
exec srun --jobid="$job_id" --nodes=1 --ntasks=1 --cpus-per-task=16 \
    --gres="gpu:$num_processes" --kill-on-bad-exit=1 \
    "$CONDA_PREFIX/bin/accelerate" launch "${launch_args[@]}" \
    "$work_dir/train.py" --config="$exp_dir/EigeNet.json" \
    --exp_name="$exp_name" --log_level debug --dataloader_seed 6006 "$@"

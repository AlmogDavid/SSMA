#!/bin/bash

export USE_WANDB=1

# Check if the number of arguments is correct
echo "Usage: $0 <sweep_id> <config> <gpu_1> <gpu_2> ... <gpu_n>"
if [ "$#" -lt 3 ]; then
  echo "Not enough arguments (expects at least 3)"
  exit 1
fi

sweep_id=$1
config=$2
gpu_ids=("${@:3}")
wandb_project='ssma'

if [ "$sweep_id" = "NONE" ]; then
  echo "Sweep ID is NONE, generating new one"
  output=$(wandb sweep ${config} --project=${wandb_project} 2>&1)
  sweep_id=$(echo "$output" | grep -o 'Creating sweep with ID: [[:alnum:]]*' | awk '{print $5}')
fi

# Print the sweep ID and total number of GPUs
echo "Sweep ID: $sweep_id"
echo "Total Number of GPUs: ${#gpu_ids[@]}"

# Iterate over the number of GPUs
for gpu_id in "${gpu_ids[@]}"; do
  echo "Launching agent on GPU $gpu_id"
  CUDA_VISIBLE_DEVICES=$gpu_id wandb agent almogdavid/${wandb_project}/${sweep_id} &
done

echo "Finished launch of agents (${#gpu_ids[@]} agents)"
echo "Sweep ID: $sweep_id"


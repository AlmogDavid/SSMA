#!/bin/bash

export USE_WANDB=1

# Check if the number of arguments is not equal to 3
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 dataset=<dataset_name> layer=<layer_name> ssma=<use ssma> <extra params (key=value format)>"
    exit 1
fi

# Parse command line arguments
# Extract dataset, layer, and SSMA values
dataset_name=""
layer_name=""
use_ssma=""
other_args=""

for arg in "$@"; do
    case "$arg" in
        dataset=*)
            dataset_name="${arg#*=}"
            ;;
        layer=*)
            layer_name="${arg#*=}"
            ;;
        ssma=*)
            use_ssma="${arg#*=}"
            ;;
        *)
            other_args="$other_args $arg"
            ;;
    esac
done

command_args=""

# Add SSMA
if [ "$use_ssma" = "true" ]; then
    if [ "$layer_name" = "pna" ]; then
        command_args=${command_args}" aggr=[mean,max,min,std,SSMA]"
    else
        command_args=${command_args}" aggr=SSMA"
    fi
else
    if [ "$layer_name" = "pna" ]; then
        command_args=${command_args}" aggr=[mean,max,min,std]"
    else
        command_args=${command_args}" aggr=add"
    fi
fi

# Run the training
command_to_run="python train.py -cn $dataset_name layer_type=$layer_name $command_args $other_args"
echo "Executing command: $command_to_run "
sleep 5
$command_to_run
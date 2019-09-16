#!/bin/bash
# 
# > Train final agent
# > USAGE: bash train_main.sh [device_id]

source define_vars.sh

cd ..

out_dir="main"
device="${1:-0}"

arguments="python train.py -config $config_file -exp $out_dir -device $device"
echo $arguments
eval $arguments


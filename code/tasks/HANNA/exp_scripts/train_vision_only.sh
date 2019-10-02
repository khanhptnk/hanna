#!/bin/bash
#
# > Train agent with vision-only assistance (Tables 4 and 11)
# > USAGE: bash train_vision_only.sh [device_id]

source define_vars.sh

cd ..

exp_name="vision_only"
out_dir="${exp_name}"
device="${1:-0}"

extra="$extra -instruction_baseline vision_only"

command="python train.py -config $config_file -exp $out_dir $extra -device $device"
echo $command
eval $command


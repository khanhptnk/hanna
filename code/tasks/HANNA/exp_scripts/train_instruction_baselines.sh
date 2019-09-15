#!/bin/bash

source define_vars.sh

cd ..

exp_name=$1
out_dir="${exp_name}"
device="${2:-1}"

extra=""

if [ "$exp_name" == "language_only" ]
then
  extra="$extra -instruction_baseline language_only"
elif [ "$exp_name" == "vision_only" ]
then      
  extra="$extra -instruction_baseline vision_only"
else
  echo "Usage: bash train_instruction_baselines.sh [language_only|vision_only]"
  exit 1
fi

command="python -u train.py -config $config_file -exp $out_dir $extra -device $device"
echo $command
$command


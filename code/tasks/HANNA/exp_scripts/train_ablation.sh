#!/bin/bash
#
# > Train ablation studied agents (Tables 8 and 9)
# > USAGE: bash train_ablation.sh [exp_name] [device_id]
# >   - exp_name must be 'no_reason' (no condition prediction, beta = 0) or 
# >                      'no_curious' (no curiosity-encouraging, alpha = 0) or 
# >                      'no_sim_attend' (no cosine similarity attention) or
# >                      'no_reset_inter' (no inter-task reset)

source define_vars.sh

cd ..

exp_name=$1
out_dir="${exp_name}"
device="${2:-0}"

extra=""

if [ "$exp_name" == "no_reason" ]
then 
  extra="$extra -no_reason 1"
elif [ "$exp_name" == "no_curious" ] 
then 
  extra="$extra -alpha 0"
elif [ "$exp_name" == "no_sim_attend" ] 
then 
  extra="$extra -no_sim_attend 1"
elif [ "$exp_name" == "no_reset_inter" ]
then
  extra="$extra -no_reset_inter 1"
else
  printf "ERROR: exp_name must be 'no_reason' or 'no_curious' or 'no_sim_attend' or 'no_reset_inter'"
  exit 1
fi

command="python train.py -config $config_file -exp $out_dir $extra -device $device"
echo $command
eval $command


#!/bin/bash

source define_vars.sh

cd ..

exp_name=$1
out_dir="no_learn_${exp_name}"

extra="-eval_only 1"

if [ "$exp_name" == "forward" ]
then
  extra="$extra -forward_agent 1"
elif [ "$exp_name" == "random" ]
then
  extra="$extra -random_agent 1"
elif [ "$exp_name" == "shortest" ]
then
  extra="$extra -shortest_agent 1"
else
  echo "Usage: bash non_learn_baselines.sh [forward|random|shortest]"
  exit 1
fi


command="python -u train.py -config $config_file -exp $out_dir $extra"
echo $command
$command


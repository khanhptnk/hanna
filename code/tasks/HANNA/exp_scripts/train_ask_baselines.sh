#!/bin/bash

source define_vars.sh

cd ..

exp_name=$1
out_dir="${exp_name}"
device="${2:-0}"

extra=""

if [ "$exp_name" == "no_ask" ]
then
  extra="$extra -ask_baseline no_ask"
elif [ "$exp_name" == "ask_every_5" ]
then
  extra="$extra -ask_baseline ask_every,5"
elif [ "$exp_name" == "random_ask_0_2" ]
then
  extra="$extra -ask_baseline random_ask,0.2"
else
  echo "Usage: bash train_ask_baselines.sh [no_ask|ask_every_5]random_ask_0_2] [device_id]"
  exit 1
fi

extra="$extra -alpha 0.5"

command="python -u train.py -config $config_file -exp $out_dir $extra -device $device"
echo $command
$command


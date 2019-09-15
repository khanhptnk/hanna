#!/bin/bash

source define_vars.sh

cd ..

exp_name=$1

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
  echo "Usage: bash eval_ask_baselines.sh [no_ask|ask_every_5|random_ask_0_2] [split]"
  exit 1
fi


split=$2
out_dir="${exp_name}"
model_path="${PT_OUTPUT_DIR}/${out_dir}/${out_dir}"

if [ "$split" == "seen_env" ]
then
  model_path="${model_path}_val_seen_env_unseen_anna.ckpt"
elif [ "$split" == "unseen_all" ] 
then
  model_path="${model_path}_val_unseen.ckpt" 
else
  echo "Split must be 'seen_env' or 'unseen_all'"
  exit 1
fi

extra="-eval_only 1 -load_path $model_path"

command="python -u train.py -config $config_file -exp $out_dir $extra"
echo $command
$command


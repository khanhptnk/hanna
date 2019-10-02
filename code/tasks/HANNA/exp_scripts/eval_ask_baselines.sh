#!/bin/bash
#
# > Evaluate agent with baseline help-request policies (Table 12)
# > USAGE: base eval_ask_baselines.sh [ask_baseline] [split]
# >   - ask_baseline must be 'no_ask' or 'ask_every_5' or 'random_ask_0_2'
# >   - split must be 'seen_env' or 'unseen_all'

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
  echo "ERROR: ask_baseline must be 'no_ask' or 'ask_every_5' or 'random_ask_0_2'"
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
  echo "ERROR: split must be 'seen_env' or 'unseen_all'"
  exit 1
fi

extra="$extra -eval_only 1 -load_path $model_path"

command="python train.py -config $config_file -exp $out_dir $extra -batch_size 1"
echo $command
eval $command


#!/bin/bash
#
# > Evaluate final agent
# > USAGE: bash eval_main.sh [split] [special_mode]
# >   - split must be 'seen_env' or 'unseen_all'
# >   - special_mode (optional) can be 'perfect_interpret' (perfect language interpretation)

source define_vars.sh

cd ..

out_dir="main"
split=$1
model_path="${PT_OUTPUT_DIR}/${out_dir}/${out_dir}"

if [ "$split" == "seen_env" ]
then
  model_path="${model_path}_val_seen_env_unseen_anna.ckpt"
elif [ "$split" == "unseen_all" ]
then
  model_path="${model_path}_val_unseen.ckpt"
else
  printf "ERROR: split must be 'seen_env' or 'unseen_all'!\n"  
  exit 1
fi

extra="-eval_only 1 -load_path $model_path"

special_mode=${2:-0}
if [ "$special_mode" == "perfect_interpret" ] 
then 
  extra="$extra -perfect_interpret 1"
fi

command="python train.py -config $config_file -exp $out_dir $extra -batch_size 1"
echo $command
eval $command


#!/bin/bash
#
# > Evaluate no-curiosity agent (alpha = 0) on TEST splits (Table 13)
# > USAGE: base eval_no_curious.sh [split]
# >   - split must be 'seen_env' or 'unseen_all'

source define_vars.sh

cd ..

split=$1
out_dir="no_curious"
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

extra="-eval_only 1 -load_path $model_path -alpha 0"

command="python train.py -config $config_file -exp $out_dir $extra -batch_size 1"
echo $command
eval $command



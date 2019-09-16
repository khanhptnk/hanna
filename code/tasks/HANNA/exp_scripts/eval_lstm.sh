#!/bin/bash
#
# > Evaluate LSTM-seq2seq baseline
# > USAGE: bash eval_lstm.sh [split] 
# >   - split must be 'seen_env' or 'unseen_all'

source define_vars.sh

cd ..

out_dir="seq2seq"
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

command="python train.py -config $config_file -exp $out_dir $extra"
echo $command
eval $command


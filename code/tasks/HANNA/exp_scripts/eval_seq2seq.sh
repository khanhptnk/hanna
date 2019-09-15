#!/bin/bash

source define_vars.sh

cd ..

out_dir="seq2seq"
split=$1
model_path="${PT_OUTPUT_DIR}/${out_dir}/${out_dir}"

if [ "$split" == "unseen_lang" ]
then
  model_path="${model_path}_val_seen_env_unseen_anna.ckpt"
elif [ "$split" == "unseen_all" ]
then
  model_path="${model_path}_val_unseen.ckpt"
else
  echo "Split must be 'unseen_lang' or 'unseen_all'"
  exit 1
fi

extra="-eval_only 1 -load_path $model_path"


command="python -u train.py -config $config_file -exp $out_dir $extra"
echo $command
$command

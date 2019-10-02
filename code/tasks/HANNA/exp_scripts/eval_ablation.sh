#!/bin/bash
#
# > Evaluate ablation studied agents (Table 9). NOTE: evaluation is conducted on VAL splits
# > USAGE: base eval_ablation.sh [exp_name] [split]
# >   - exp_name must be 'no_reason' (no condition prediction, beta = 0) or
# >                      'no_curious' (no curiosity-encouraging, alpha = 0) or
# >                      'no_sim_attend' (no cosine similarity attention) or
# >                      'no_reset_inter' (no inter-task reset)
# >   - split must be 'seen_env' or 'unseen_all'

source define_vars.sh

cd ..

exp_name=$1

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
  echo "ERROR: exp_name must be 'no_reason' or 'no_curious' or 'no_sim_attend' or 'no_reset_inter'"
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

extra="$extra -eval_only 1 -load_path $model_path -eval_on_val 1"

command="python train.py -config $config_file -exp $out_dir $extra -batch_size 1"
echo $command
eval $command



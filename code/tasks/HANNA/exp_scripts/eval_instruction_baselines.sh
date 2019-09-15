#!/bin/bash

source define_vars.sh

cd ..

exp_name=$1                                                  

if [ "$exp_name" == "language_only" ]
then
  extra="$extra -instruction_baseline language_only"
elif [ "$exp_name" == "vision_only" ]
then      
  extra="$extra -instruction_baseline vision_only"
else
  echo "Usage: bash eval_instruction_baselines.sh [language_only|vision_only] [split]"
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


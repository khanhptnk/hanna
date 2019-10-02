#!/bin/bash
#
# > Evaluate agent trained with vision-only assistance (Table 11)
# > USAGE: bash eval_vision_only.sh [language_only|vision_only] [split]
# >   - split can be 'seen_env' or 'unseen_all'

source define_vars.sh

cd ..

exp_name="vision_only"

split=$1
out_dir="${exp_name}"                                                           
model_path="${PT_OUTPUT_DIR}/${out_dir}/${out_dir}"          
if [ "$split" == "seen_env" ]                                                
then                                                                            
  model_path="${model_path}_val_seen_env_unseen_anna.ckpt"                      
elif [ "$split" == "unseen_all" ]                                               
then                                                                            
  model_path="${model_path}_val_unseen.ckpt"                                    
else                                                                            
  echo "ERROR: Split must be 'seen_env' or 'unseen_all'"
  exit 1                                                                        
fi   

extra="-eval_only 1 -load_path $model_path -instruction_baseline vision_only" 


command="python train.py -config $config_file -exp $out_dir $extra -batch_size 1"
echo $command
eval $command


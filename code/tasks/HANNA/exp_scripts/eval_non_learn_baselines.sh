#!/bin/bash
#
# > Evaluate non-learning agents
# > USAGE: bash eval_non_learn_baselines.sh [agent_name]
# >   - agent_name must be 'forward' or 'random' or 'shortest'

source define_vars.sh

cd ..


exp_name=$1
out_dir="no_learn_${exp_name}"

extra="-eval_only 1"

if [ "$exp_name" == "forward" ]
then
  extra="$extra -forward_agent 1"
elif [ "$exp_name" == "random" ]
then
  extra="$extra -random_agent 1"
elif [ "$exp_name" == "shortest" ]
then
  extra="$extra -shortest_agent 1"
else
  printf "ERROR: agent_name must be 'forward' or 'random' or 'shortest'\n"
  exit 1
fi


command="python train.py -config $config_file -exp $out_dir $extra"
echo $command
eval $command


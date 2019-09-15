#!/bin/bash

source define_vars.sh

cd ..

out_dir="alpha_0"
device="${1:-0}"

command="python -u train.py -config $config_file -exp $out_dir -device $device -alpha 0"
echo $command
$command


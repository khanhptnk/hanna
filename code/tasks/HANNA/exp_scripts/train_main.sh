#!/bin/bash

source define_vars.sh

cd ..

out_dir="main"
device="${1:-0}"

command="python3 -u train.py -config $config_file -exp $out_dir -device $device"
echo $command
$command


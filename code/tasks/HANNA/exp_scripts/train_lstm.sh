#!/bin/bash
# 
# > Train LSTM-seq2seq baseline
# > USAGE: bash train_lstm.sh [device_id]

source define_vars.sh

cd ..

out_dir="lstm"
device="${1:-0}"

arguments="python train.py -config $config_file -exp $out_dir -device $device -log 1"
echo $arguments
eval $arguments


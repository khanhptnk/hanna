#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

git clone --recursive https://github.com/peteanderson80/Matterport3DSimulator.git
cp -r Matterport3DSimulator/connectivity . 
yes | rm -rf Matterport3DSimulator

wget -O anna_data.zip https://www.dropbox.com/s/bc9y43qj1014hic/anna_data.zip?dl=1
unzip anna_data.zip
mv anna_data/* .
rm -rf anna_data

mkdir img_features
cd img_features
wget -O ResNet-152-imagenet.zip https://www.dropbox.com/s/715bbj8yjz32ekf/ResNet-152-imagenet.zip?dl=1
unzip ResNet-152-imagenet.zip



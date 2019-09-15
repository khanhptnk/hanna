#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

git clone --recursive https://github.com/peteanderson80/Matterport3DSimulator.git
cp -r Matterport3DSimulator/connectivity . 
yes | rm -rf Matterport3DSimulator

wget -O hanna_data.zip https://www.dropbox.com/s/qfahoe22vh1cdhn/hanna_data.zip?dl=1
unzip hanna_data.zip

#mkdir img_features
#cd img_features
#wget -O ResNet-152-imagenet.zip https://www.dropbox.com/s/715bbj8yjz32ekf/ResNet-152-imagenet.zip?dl=1
#unzip ResNet-152-imagenet.zip



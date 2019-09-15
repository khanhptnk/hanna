#!/bin/bash

ANNA_DATA_DIR=$(realpath ../data)

echo $MATTERPORT_DATA_DIR
echo $ANNA_DATA_DIR

nvidia-docker run -it --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/hanna/code/data/v1/scans,readonly --mount type=bind,source=$ANNA_DATA_DIR,target=/root/mount/hanna/data,readonly --volume `pwd`:/root/mount/hanna/code hanna

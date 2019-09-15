
ANNA_DATA_DIR=$(realpath ../data)

echo $MATTERPORT_DATA_DIR
echo $ANNA_DATA_DIR

xhost + 
nvidia-docker run -it -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/hanna/code/data/v1/scans,readonly --mount type=bind,source=$ANNA_DATA_DIR,target=/root/mount/hanna/data,readonly --volume `pwd`:/root/mount/hanna/code hanna

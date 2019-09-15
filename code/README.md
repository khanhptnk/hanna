
Please first [download the data](https://github.com/khanhptnk/hanna-private/tree/master/data)

HANNA extends the [Matterport3D simulator](https://github.com/peteanderson80/Matterport3DSimulator). It is easiest to setup on our pre-configured Docker image. If you want to setup with Docker, please make sure the following tools have been installed:
* [docker](https://docs.docker.com/install/)
* [nvidia-docker 2.0](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0))

You can also install HANNA without Docker (similar to [installing the Matterport3D simulator without Docker](https://github.com/peteanderson80/Matterport3DSimulator#building-without-docker)) but it would be complicated to resolve the dependency requirements!

1. Build Docker image
```
$ sudo bash scripts/build_docker.sh
```

2. Run Docker image

Export link to the Matterport3D root directory

```
$ export MATTERPORT_DATA_DIR=<some_path>/v1/scans
```

Without graphics
```
$ sudo -E bash scripts/run_docker.sh
```

With graphics
```
$ sudo -E bash scripts/run_docker_graphics.sh
```

3. Build the simulator

Inside the Docker image, run
```
root@66a410498bfb:~/mount/hanna/code# bash scripts/setup.sh
```

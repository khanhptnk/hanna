# Matterport3DSimulator
# Requires nvidia gpu with driver 384.xx or higher


FROM nvidia/cudagl:10.0-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

# Install a few libraries to support both EGL and OSMESA options
RUN apt-get update && apt-get install -y wget doxygen curl libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev libopencv-dev python-opencv python-setuptools python-dev python3-pip python3 libcanberra-gtk-module libcanberra-gtk3-module 
RUN pip3 install torch==1.1.0 torchvision pandas networkx termcolor sklearn opencv-python
RUN echo 'alias python="python3"' >> ~/.bashrc
RUN echo 'cd root/mount/hanna/code' >> ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"

#install latest cmake
ADD https://cmake.org/files/v3.12/cmake-3.12.2-Linux-x86_64.sh /cmake-3.12.2-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.12.2-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version


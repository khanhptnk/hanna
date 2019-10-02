FROM nvidia/cudagl:10.0-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

# Install a few libraries to support both EGL and OSMESA options
RUN apt-get update && apt-get install -y wget doxygen curl libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev libopencv-dev python-opencv python-setuptools python-dev python3-pip python3 libcanberra-gtk-module libcanberra-gtk3-module 

# Install Python packages
RUN rm -rf torch-1.1.0-cp36-cp36m-linux_x86_64.whl
RUN wget -q https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
RUN pip3 install --upgrade --force-reinstall --no-cache-dir torch-1.1.0-cp36-cp36m-linux_x86_64.whl
RUN rm -rf torch-1.1.0-cp36-cp36m-linux_x86_64.whl
RUN pip3 install torchvision==0.3.0 pandas networkx termcolor sklearn opencv-python

# Some useful stuffs
RUN echo 'alias python="python3"' >> ~/.bashrc
RUN echo 'cd root/mount/hanna/code' >> ~/.bashrc
RUN echo 'export QT_X11_NO_MITSHM=1' >> ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"

# Install latest cmake
ADD https://cmake.org/files/v3.12/cmake-3.12.2-Linux-x86_64.sh /cmake-3.12.2-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.12.2-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version


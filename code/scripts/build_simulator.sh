#!/bin/bash

rm -rf build
mkdir build && cd build
cmake -DEGL_RENDERING=ON ..
make


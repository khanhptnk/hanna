# HANNA: Visual Navigation with Natural Multimodal Assistance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <img src="teaser/pytorch-logo-dark.png" width="10%"> 

Authors: [Khanh Nguyen](https://khanhptnk.github.io), [Hal Daumé III](http://users.umiacs.umd.edu/~hal/)

This repo contains code and data-downloading scripts for the paper [Help, Anna! Visual Navigation with Natural Multimodal Assistance via Retrospective Curiosity-Encouraging Imitation Learning](https://arxiv.org/abs/1909.01871) (EMNLP 2019). We develop *Help, Anna!* (HANNA), an interactive photo-realistic simulator in which an agent fulfills object-finding tasks by requesting and interpreting natural language-and-vision assistance.

### System requirements

The environment simulator is installed and runs on a Docker image. Please install:

* [docker](https://docs.docker.com/install/)
* [nvidia-docker 2.0](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0))

You can also install the simulator [without source code](https://github.com/peteanderson80/Matterport3DSimulator#building-without-docker) but it is not recommended!

### Let's get started!

0. Clone this repo `git clone --recursive https://github.com/khanhptnk/hanna-private.git` (don't forget the `recursive` flag!)
1. [Download data](https://github.com/khanhptnk/hanna-private/tree/master/data). 
2. [Setup simulator](https://github.com/khanhptnk/hanna-private/tree/master/code). 
3. [Run experiments](https://github.com/khanhptnk/hanna-private/tree/master/code/tasks/HANNA). 

### What are the differences between this task and other photo-realistic visual navigation tasks?

A: In R2R, the agent's task is given by a **detailed** language instruction (e.g., "Go the table, turn left, walk to the stairs, wait there"). The agent has to execute the instruction **without additional assistance**. 

In VNLA (our task), the task is described as a **high-level end-goal** (the steps for accomplishing the task are _not_ described) (e.g., "Find a cup in the kitchen"). The agent is capable of **actively requesting additional assistance** (in the form of language subgoals) while trying to fulfill the task. 

### Citation

If you use the code or data in this repo, please cite our paper using the following bibtex code

```
@inproceedings{nguyen2019hanna,
  author = {Nguyen, Khanh and Daum{\'e} III, Hal},
  title = {Help, Anna! Visual Navigation with Natural Multimodal Assistance via Retrospective Curiosity-Encouraging Imitation Learning},
  booktitle = {Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  month = {November},
  year = {2019},
}
```

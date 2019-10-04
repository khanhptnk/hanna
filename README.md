# HANNA: Visual Navigation with Natural Multimodal Assistance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <img src="teaser/pytorch-logo-dark.png" width="10%"> 

EMNLP'19 Paper: [Help, Anna! Visual Navigation with Natural Multimodal Assistance via Retrospective Curiosity-Encouraging Imitation Learning](https://arxiv.org/abs/1909.01871)

Authors: [Khanh Nguyen](https://khanhptnk.github.io), [Hal Daumé III](http://users.umiacs.umd.edu/~hal/)

## What is HANNA?

HANNA is an interactive photo-realistic simulator that mimics an agent fulfilling object-finding tasks by leveraging natural language-and-vision assistance.

<p align="center">
<img src="teaser/example.png" alt="IMAGE ALT TEXT HERE" width="1000" border="10"/>
An example HANNA task.
</p>


## How is HANNA different from other photo-realistic visual navigation tasks?

<p align="center">
<img src="teaser/difference.png" alt="IMAGE ALT TEXT HERE" width="500" border="10"/>
  
Comparison HANNA with VLN <a href="https://arxiv.org/abs/1711.07280">(Anderson et al., 2018b)</a>, EQA <a href="https://arxiv.org/abs/1904.03461">(Wijmans et al., 2019)</a>, VNLA <a href="https://arxiv.org/abs/1812.04155">(Nguyen et al., 2019)</a>, CVDN <a href="https://arxiv.org/abs/1907.04957">(Thomason et al., 2019)</a>.
</p>

## Let's play with HANNA!

0. `git clone --recursive https://github.com/khanhptnk/hanna.git` (don't forget the `recursive` flag!)
1. [Download data](https://github.com/khanhptnk/hanna-private/tree/master/data). 
2. [Setup simulator](https://github.com/khanhptnk/hanna-private/tree/master/code). 
3. [Run experiments](https://github.com/khanhptnk/hanna-private/tree/master/code/tasks/HANNA). 



## Citation

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

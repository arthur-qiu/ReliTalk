# ReliTalk - Official PyTorch Implementation

This repository provides the official PyTorch implementation for the following paper:

**ReliTalk: Relightable Talking Portrait Generation from a Single Video**</br>
[Haonan Qiu](http://haonanqiu.com/), [Zhaoxi Chen](https://frozenburning.github.io), [Yuming Jiang](https://yumingj.github.io/), [Hang Zhou](https://hangz-nju-cuhk.github.io/), [Wayne Wu](https://wywu.github.io/), [Xiangyu Fan](https://github.com/arthur-qiu/ReliTalk), [Lei Yang](https://scholar.google.com.hk/citations?user=jZH2IPYAAAAJ&hl=en), and [Ziwei Liu](https://liuziwei7.github.io/)</br>

From [MMLab@NTU](https://www.mmlab-ntu.com/index.html) affliated with S-Lab, Nanyang Technological University and SenseTime Research.

[**[Project Page]**](https://github.com/arthur-qiu/ReliTalk) | [**[Paper]**](https://github.com/arthur-qiu/ReliTalk) | [**[Demo Video]**](https://github.com/arthur-qiu/ReliTalk)

## Datasets

Video Data: [HDTF](https://github.com/MRzzm/HDTF)

## Getting Started
* Clone this repo: `git clone --recursive git@github.com:arthur-qiu/ReliTalk.git`
* Create a conda environment `conda env create -f environment.yml` and activate `conda activate IMavatar` 
* We use `libmise` to extract 3D meshes, build `libmise` by running `cd code; python setup.py install`
* Download [FLAME model](https://flame.is.tue.mpg.de/download.php), choose **FLAME 2020** and unzip it, copy 'generic_model.pkl' into `./code/flame/FLAME2020`

## Preparing Dataset

Prepare the dataset following intructions in `./preprocess/README.md`.

Link the dataset folder to `./data/datasets`. Link the experiment output folder to `./data/experiments`.

## Training

## Inference

## We are organizing the code and will release it soon. 

## Acknowledgments
This code borrows heavily from [IMavatar](https://github.com/zhengyuf/IMavatar).
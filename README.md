# Dual Attention Residual Group Networks for Single Image Deraining
## Introduction
In this paper, we propose a novel dual attention residual group networks (DARGNet) for better deraining performance. Specifically, the proposed framework of dual attention includes spatial attention and channel attention. The spatial attention extracts the multiscale feature. Meanwhile, channel attention separates channel domain and spatial attention which can extract multiple attributes from different channels and guide the selection of the most important features. Furthermore, to simplify the structure, we integrate the dual attention module and convolution layers to the residual groups. The residual information can enhance the information flow. Extensive experiments on synthesized and real-world datasets verify the superiority of the proposed network over the state-of-the-art image deraining. 
## Prerequisites

- Python 3.6, PyTorch >= 0.4.0 
- Requirements: opencv-python, tensorboardX
- Platforms: Ubuntu 16.04, cuda-8.0 & cuDNN v-5.1 (higher versions also work well)
## Datasets

our model are evaluated on three datasets: 
Rain100H , Rain100L , Rain12 .   
Please download the testing datasets from [BaiduYun](https://pan.baidu.com/s/115-OBqATI9JGS3ZG0-BUsA) Access Code ：xbts   
and place the unzipped folders into `./datasets/test/`.

To train the models, please download training datasets:   
RainTrainH , RainTrainL from [BaiduYun](https://pan.baidu.com/s/115-OBqATI9JGS3ZG0-BUsA) Access Code ：xbts    
and place the unzipped folders into `./datasets/train/`


## Getting Started

### 1) Testing

We have placed our pre-trained models into `./logs/`. 
Run scripts to test the models:

```python
python test_Rain100H.py   # test models on Rain100H
python test_Rain100L.py   # test models on Rain100L
python test_Rain12.py     # test models on Rain12
python test_real.py       # test models on real rainy images
```

All the results in the paper are also available at [BaiduYun](https://pan.baidu.com/s/1HRXNR05y5tWgeb8eFhylug )   Access Code：bj1j 

### 2) Training
Run scripts to train the models:

```python
python train_rain100H.py   # test models on Rain100H
python train_rain100H.py   # test models on Rain100L
```

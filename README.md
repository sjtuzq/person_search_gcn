# Person Search GCN Project

This repository hosts the code for our paper [Learning Context Graph for Person Search], CVPR2019 Oral. Since any technique used for re-id model can be embodied in our model such as part-based attention, we remove them when release our code. Besides, the part of person detectin is modified from the py-faster-rcnn written by Ross Girshick and not included in this code. This repository shows how we can use graph convolution model to improve person search performance.


## Environment:
python(3.6),
pytorch(0.4.1),
numpy(1.14.1), 
matplotlib(3.0.2),
tqdm,
pickle


## Installation

1. Clone this repo 

  ```Shell
  git clone https://github.com/sjtuzq/person_search_gcn.git
  ```

2. modify data_path, log_path and neighbor_num in config.py

3. download dataset [here](https://drive.google.com/open?id=1-pjZd-bZFTqV2F_34jr0q77-iEmjE4P5),then put it into data_path folder. The dataset are generate by the paper [Joint Detection and Identification Feature Learning for Person Search](https://arxiv.org/abs/1604.01850), whose code can be found [here](https://github.com/ShuangLI59/person\_search).

4. prepare dataset 
  generate the raw data feature into the paired form: persons in the same images are grouped together.

  ```Shell
  python prepare.py
  ```

5. pretrain the gcn model
  pretrain graph convolution model: with the pair selected by the distance data.

  ```Shell
  python train_gcn1.py
  ```
An example output:

  ```Shell
  test_eval0:0.7766  eval:0.7769  now_best:0.7769
  ```

6. train and test the gcn model
 train and test graph convolution model to do person re-id: using the pretrained model and use one attention model to select the companies to refine the feature.

  ```Shell
  python train_gcn2.py
  ```
  
An example output:

  ```Shell
  epoch:0  acc:0.8410   map:0.8401
  ```

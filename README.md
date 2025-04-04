

# Traffic Sign Localization and Orientation Classification for Automated Map Updating


## Introduction

This paper develops an automated traffic sign update system, AutoTS, aimed at extracting the geo-location and orientation of traffic signs.
To facilitate the evaluation and comparison, we construct a traffic sign localization and orientation classification benchmark, KITTI-TS, based on the KITTI dataset.


## Requirements
Set up an environment for the code.
```bash
conda create -n projectname python=3.10
conda activate projectname
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```


## Datasets

The images in our dataset are sourced from the KITTI dataset, and the annotations are stored in the sign_id_GT.json file. Both the images and the annotation file can be accessed via [this link](https://drive.google.com/drive/folders/1tBI-o6KKN_ZdPLeszFe0Muzj_agnrraq?usp=sharing).
## Evaluation and Training

For evaluation, please download our model checkpoint from [this link](https://drive.google.com/drive/folders/1tBI-o6KKN_ZdPLeszFe0Muzj_agnrraq?usp=drive_link).

Evaluation
```bash
python orientatin_train.py --eval-only
```
Training Detector
```bash 
python train_net.py
```

Training 
```bash 
python orientatin_train.py
```

## Paper and Citing 
If you find this project helps your research, please kindly consider citing our papers in your publications. 

(Under Review)

## Acknowledge

This repository is built on Detectron2.
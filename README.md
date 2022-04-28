# Robustifying Vision Transformer without Retraining from Scratch by Test-Time Class-Conditional Feature Alignment

This is the official implementation of `Robustifying Vision Transformer without Retraining from Scratch by Test-Time Class-Conditional Feature Alignment` (IJCAI-ECAI2022, Short).

## Installation

### Prerequisite
- Python==3.8
- torch==1.9.0
- torchvision==0.10.0

### timm library
- For ViT
```
pip install timm==0.4.9
```

- For the other models (ViT-AugReg, DeiT, MLP-Mixer, BeiT, ResNet)
```
pip install git+https://github.com/rwightman/pytorch-image-models@more_datasets # 0.5.0
```

### The others
```
pip install -r requirements.txt
```

## Dataset Preparation

## Argument Setting
```
model=
dataset=
```

## 1st Phase : Calculating distribution statistics on source dataset

## 2nd Phase : Test-Time Adaptation by CFA
```
python main.py 
```

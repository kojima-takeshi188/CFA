# Robustifying Vision Transformer without Retraining from Scratch by Test-Time Class-Conditional Feature Alignment

This is the official implementation of `Robustifying Vision Transformer without Retraining from Scratch by Test-Time Class-Conditional Feature Alignment` (IJCAI-ECAI2022, Short).

## Installation

### Prerequisite (Hardware)
- GPU : A100 x 1GPU (40 GB Memory)
- Disk Space : About 300 GB

### Prerequisite (Software)
- Python==3.8
- torch==1.9.0
- torchvision==0.10.0

### timm library
- For ViT
```
pip install timm==0.4.9
```

- For the other models (ViT-AugReg, DeiT, MLP-Mixer, BeiT)
```
pip install git+https://github.com/rwightman/pytorch-image-models@more_datasets # 0.5.0
```

### The others
```
pip install -r requirements.txt
```

## Dataset Preparation

Download each datasets and unzip them under the following directory.

- ImageNet-2012 (as Source)
```
datasets/imagenet2012/train
datasets/imagenet2012/val
```

- ImageNet-C (as Target)
```
datasets/imagenet2012/val_c
```

## Argument Setting
```
model={'ViT-B_16', 'ViT-L_16', 'ViT_AugReg-B_16', 'ViT_AugReg-L_16', 'resnet50', 'resnet101', 'mlpmixer_B16', 'mlpmixer_L16', 'DeiT-B', 'DeiT-S', 'Beit-B16_224', 'Beit-L16_224', 'Beit-L16_512'}
loss_function={'cfa', 't3a', 'shot-im', 'tent', 'pl'}
```

## 1st Phase : Fine-Tuning (Skip)
In this implementation, we use models that are already fine-tuned on ImageNet-2012 dataset.
Our method does not need to alter training phase, i.e., does not need to retrain models from scratch.
Therefore, we can skip fine-tuning phase.

## 2nd Phase : Calculation of distribution statistics on source dataset
```
python main.py 
```

## 3rd Phase : Test-Time Adaptation (TTA)
```
python main.py 
```

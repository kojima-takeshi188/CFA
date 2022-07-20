# Robustifying Vision Transformer without Retraining from Scratch by Test-Time Class-Conditional Feature Alignment

This is the official implementation of `Robustifying Vision Transformer without Retraining from Scratch by Test-Time Class-Conditional Feature Alignment` (IJCAI-ECAI2022, Short).

The paper is available at [IJCAI-ECAI2022](https://www.ijcai.org/proceedings/2022/141) (main only) and [arXiv](https://arxiv.org/abs/2206.13951) (main and appendix).

## Installation

### Prerequisite

#### Hardware
- GPU : A100 x 1GPU (40 GB Memory)
- Disk Space : About 300 GB

#### Software
- Python==3.7.13
- torch==1.9.0
- torchvision==0.10.0

### [timm](https://github.com/rwightman/pytorch-image-models) library
- For ViT
```
pip install timm==0.4.9
```

- For the other models (ViT-AugReg, DeiT, MLP-Mixer, BeiT)
```
pip install git+https://github.com/rwightman/pytorch-image-models@more_datasets # 0.5.0
```

### The other libraries
```
pip install -r requirements.txt
```

## Dataset

Download each datasets and unzip them under the following directory.

- [ImageNet-2012](https://image-net.org/index.php) (as Source)
```
./datasets/imagenet2012/train
./datasets/imagenet2012/val
```

- [ImageNet-C](https://github.com/hendrycks/robustness) (as Target)
```
./datasets/imagenet2012/val_c
```

## Quick start

#### (1) Argument Setting
```
model={'ViT-B_16', 'ViT-L_16', 'ViT_AugReg-B_16', 'ViT_AugReg-L_16', 'resnet50', 'resnet101', 'mlpmixer_B16', 'mlpmixer_L16', 'DeiT-B', 'DeiT-S', 'Beit-B16_224', 'Beit-L16_224'}
method={'cfa', 't3a', 'shot-im', 'tent', 'pl', 'source'}
```

#### (2) Fine-Tuning (Skip)
Our method does not need to alter training phase, i.e., does not need to retrain models from scratch.
Therefore, if a fine-tuned model is available, we can skip fine-tuning phase.
In this implementation, we use models that are already fine-tuned on ImageNet-2012 dataset.

#### (3) Calculation of distribution statistics on source dataset
```
python main.py --calc_statistics_flag --model=${model} --method=${method}
```

#### (4) Test-Time Adaptation (TTA) on target dataset
```
python main.py --tta_flag --model=${model} --method=${method}
```

#### Expected results

Top-1 Error Rate on ImageNet-C with severity level=5. ViT_B16 is used as a backbone network.

|                                                            | mean | gauss_noise | shot_noise | impulse_noise | defocus_blur | glass_blur | motion_blur | zoom_blur | snow | frost |  fog | brightness | contrast | elastic_trans | pixelate | jpeg |
|------------|-----:|------------:|-----------:|--------------:|-------------:|-----------:|------------:|----------:|-----:|------:|-----:|-----------:|---------:|--------------:|---------:|-----:|
| source     | 61.9 |        77.7 |       75.1 |          77.0 |         66.9 |       69.1 |        58.5 |      62.8 | 60.9 |  57.6 | 62.9 |       31.6 |     88.9 |          51.9 |     45.3 | 42.9 |
| CFA       | 43.9 |        56.3 |       54.3 |          55.4 |         48.5 |       47.1 |        44.3 |      44.4 | 44.8 |  44.8 | 41.1 |       25.7 |     54.2 |          33.3 |     30.5 | 33.5 |

## Citation
```
@inproceedings{kojima2022robustvit,
  title     = {Robustifying Vision Transformer without Retraining from Scratch by Test-Time Class-Conditional Feature Alignment},
  author    = {Kojima, Takeshi and Matsuo, Yutaka and Iwasawa, Yusuke},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  pages     = {1009--1016},
  year      = {2022},
  month     = {7},
  url       = {https://doi.org/10.24963/ijcai.2022/141},
}
```

## Contact
- t.kojima@weblab.t.u-tokyo.ac.jp
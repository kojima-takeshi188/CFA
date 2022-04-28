import os
import math
import numpy as np
import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import matplotlib.pyplot as plt
import torchvision
import subprocess
import random
import timm
import datetime
from PIL import Image

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def safe_log(x, ver):
    if ver == 1:
        return torch.log(x + 1e-5)
    elif ver == 2:
        return torch.log(x + 1e-7)
    elif ver == 3:
        return torch.clamp(torch.log(x), min=-100)
    else:
        raise ValueError("safe_log version is not properly defined !!!")
    
def h_std(args, h):
    if args.h_std_version == 1:
        h_hat = nn.Tanh()(nn.LayerNorm(h.shape[-1], eps=1e-05, elementwise_affine=False)(h))
    else:
        raise ValueError("h_std_version is not properly defined !!!")
    return h_hat

def print_current_time(information=None):
    print(str(datetime.datetime.now()) + " : " + str(information))
    
def save_image_sample(args, img):
    os.makedirs(args.save_image_sample_path, exist_ok=True)
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), 
                                 os.path.join(args.save_image_sample_path, current_time+'_'+args.loss_function+'.png'))
    
def softmax_entropy_tent(x):
    return -(x.softmax(-1) * x.log_softmax(-1)).sum(dim=-1) # [b,c] -> [b]

def softmax_diversity_regularizer(x):
    x2 = x.softmax(-1).mean(0) # [b, c] -> [c]
    return (x2 * safe_log(x2, ver=3)).sum()
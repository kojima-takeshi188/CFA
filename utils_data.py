import os
import math
import numpy as np
import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import subprocess
import random
import datetime
from torchvision import transforms, datasets
from utils import *

def norm_mean_and_std(args):
    # https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
    if ("resnet" in args.model) or ("DeiT" in args.model):
        if args.dataset == "imagenet":
            normalize_mean = [0.485, 0.456, 0.406]
            normalize_std = [0.229, 0.224, 0.225]
        else:
            raise ValueError("transforms.Normalize error 1 !!!")
    elif ("ViT" in args.model) or ("mlpmixer" in args.model) or ("Beit" in args.model):
        normalize_mean = [0.5, 0.5, 0.5]
        normalize_std = [0.5, 0.5, 0.5]
    else:
        raise ValueError("transforms.Normalize error 2 !!!")
    return normalize_mean, normalize_std
    
def setup_data_loader(args, minibatch_size, data, corruption_name=None, severity=None, shuffle=True):

    # With "fix_seed" function, the data order becomes the same across all the methods within a model.
    # But if the model (network architecture) is changed, the data order is changed.
    # The workaround is enabling "strict_fix_of_dataloader_seed_flag" as described below.
    fix_seed(args.random_seed)
    
    # Fix randomness of data loader to strictly ensure reproducibility.
    # https://pytorch.org/docs/stable/notes/randomness.html
    if args.strict_fix_of_dataloader_seed_flag:
        print("strict_fix_of_dataloader_seed")
        worker_seed = torch.initial_seed() % 2**32
        print("worker_seed : {}".format(worker_seed))
        def seed_worker(worker_id):
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        g = torch.Generator()
        g.manual_seed(worker_seed)
    else:
        seed_worker = None
        g = None
    
    dataloader_num_workers = multiprocessing.cpu_count() #torch.cuda.device_count() * 4 #multiprocessing.cpu_count() # 5
    dataloader_num_workers = min(dataloader_num_workers, args.max_num_worker)
    print("dataloader_num_workers: " + str(dataloader_num_workers))
    
    normalize_mean, normalize_std = norm_mean_and_std(args)
    
    print("transforms.Normalize")
    print(normalize_mean)
    print(normalize_std)

    transform_without_da = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
    ])
    transform_without_da_imagenet = transforms.Compose([
        transforms.Resize(args.image_crop_size), # Imagenet-C dataset : 256
        transforms.CenterCrop(args.image_size), # Imagenet-C dataset : 224
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
    ])
    
    if data == "imagenet-c":
        transform_train = transform_without_da
        transform_test = transform_without_da
    elif data == "imagenet":
        transform_train = transform_without_da_imagenet
        transform_test = transform_without_da_imagenet
    else:
        raise ValueError("transforms setting error !!!")
    
    if data == "imagenet":
                        
        imagenet_trainset = torchvision.datasets.ImageNet(
                                    root=args.data_path + 'imagenet2012/',
                                    split='train',
                                    transform=transform_train)

        imagenet_testset = torchvision.datasets.ImageNet(
                                    root=args.data_path + 'imagenet2012/',
                                    split='val',
                                    transform=transform_test)
        
        imagenet_train_loader = torch.utils.data.DataLoader(imagenet_trainset,
                                  shuffle=shuffle,
                                  batch_size=minibatch_size,
                                  drop_last=args.dataloader_drop_last,
                                  num_workers=dataloader_num_workers,
                                  worker_init_fn=seed_worker,
                                  generator=g,
                                  pin_memory=True)
        
        imagenet_test_loader = torch.utils.data.DataLoader(imagenet_testset,
                                  shuffle=shuffle,
                                  batch_size=minibatch_size,
                                  drop_last=args.dataloader_drop_last,
                                  num_workers=dataloader_num_workers,
                                  worker_init_fn=seed_worker,
                                  generator=g,
                                  pin_memory=True)
        
        return imagenet_train_loader, imagenet_test_loader

    if data == "imagenet-c":
        
        imagenetc_path = args.data_path  + 'imagenet2012/' + 'val_c/' + corruption_name + '/' + str(severity) + '/'
        
        # ImageFolder Function...
        # https://zenn.dev/hidetoshi/articles/20210717_pytorch_dataset_for_imagenet
        imagenet_c_testset = torchvision.datasets.ImageFolder( \
                             root = imagenetc_path, \
                             transform = transform_test) #transform_test_imagenet)
        
        imagenet_c_test_loader = torch.utils.data.DataLoader(imagenet_c_testset,
                                  shuffle=shuffle,
                                  batch_size=minibatch_size,
                                  drop_last=args.dataloader_drop_last,
                                  num_workers=dataloader_num_workers,
                                  worker_init_fn=seed_worker,
                                  generator=g,
                                  pin_memory=True)
        
        return imagenet_c_test_loader

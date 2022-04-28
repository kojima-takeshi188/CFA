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

def construct_model(args):
  
  if "resnet" in args.model:
    
    if "resnet50" in args.model:
        model = torchvision.models.resnet50(pretrained=True).to(args.device)
    elif "resnet101" in args.model:
        model = torchvision.models.resnet101(pretrained=True).to(args.device)
    else:
        raise ValueError("resnet model construction error !!!")
  
  # ViT original models ...
  elif ("ViT" in args.model) and ("ViT_AugReg" not in args.model):
    
    if str(timm.__version__) != "0.4.9":
        print("pip uninstall timm")
        print("pip install timm==0.4.9")
        print("If \' The file might be corrupted \' error happens, \
               cd ~/.cache/torch/hub/checkpoints/ and rm the file and try this script again ...")
        raise ValueError("timm version should be 0.4.9 for ViT_Origin !!!")
        
    if "ViT-B_16" == args.model:
        model = timm.create_model('vit_base_patch16_224', pretrained=True).to(args.device)
    elif "ViT-L_16" == args.model:
        model = timm.create_model('vit_large_patch16_224', pretrained=True).to(args.device)
    else:
        raise ValueError("ViT_Origin model construction error !!!")
        
  else:
    
    if str(timm.__version__) != "0.5.0":
        print("pip uninstall timm")
        print("pip install git+https://github.com/rwightman/pytorch-image-models")
        raise ValueError("timm version should be 0.5.0 !!!")
    
    # https://github.com/rwightman/pytorch-image-models/
    # https://paperswithcode.com/lib/timm
    if "ViT_AugReg-B_16" == args.model:
        model = timm.create_model('vit_base_patch16_224', pretrained=True).to(args.device)
    elif "ViT_AugReg-L_16" == args.model:
        model = timm.create_model('vit_large_patch16_224', pretrained=True).to(args.device)
    elif "mlpmixer_B16" == args.model:
        model = timm.create_model('mixer_b16_224', pretrained=True).to(args.device)
    elif "mlpmixer_L16" == args.model:
        model = timm.create_model('mixer_l16_224', pretrained=True).to(args.device)
    elif "DeiT-B" == args.model:
        model = timm.create_model('deit_base_distilled_patch16_224', pretrained=True).to(args.device)
    elif "DeiT-S" == args.model:
        model = timm.create_model('deit_small_distilled_patch16_224', pretrained=True).to(args.device)
    elif "Beit-B16_224" == args.model:
        model = timm.create_model('beit_base_patch16_224', pretrained=True).to(args.device)
    elif "Beit-L16_224" == args.model:
        model = timm.create_model('beit_large_patch16_224', pretrained=True).to(args.device)
    else:
        raise ValueError("model construction error(model selection) !!!")
  
  def hook_fn(m, input):
        global bridging_variables
        bridging_variables = input[0]
  
  class Model_Wrapper(nn.Module):
        def __init__(self, model):
            super(Model_Wrapper, self).__init__()
            self.model = model
            if "resnet" in args.model:
                self.classifier = model.fc
                self.model.fc.register_forward_pre_hook(hook_fn)
            else:
                self.classifier = model.head
        def forward(self, x):
            logits = self.model(x)
            h = bridging_variables
            return logits, h
  
  model = Model_Wrapper(model)
  return model

def load_model_from_saved_file(args):
  model = construct_model(args)
  model.load_state_dict(torch.load(args.model_save_file)) #, strict=False)
  print("The model is restored ...")
  return model

def save_model_to_file(model, file_path):
  torch.save(model.state_dict(), file_path)
  print("The model is saved ...")

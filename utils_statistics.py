import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import random
import math
import copy
import os
from utils import *
from datetime import timedelta

def calc_cmd_statistics(args, model, train_loader):
  
  model.eval()

  # No gradient during evaluation ...
  with torch.no_grad():
    total = 0
    total_mid_cls_cms = 0
    mid_ent_cms = [0] * args.save_max_moment
    mid_cls_cms = [0] * args.save_max_moment
    
    for i, data in enumerate(train_loader, 0):
        if i % 100 == 0: print_current_time(i)
        x, y = data
        x, y = x.to(args.device), y.to(args.device)
        logits, h = model(x)
        h = h_std(args, h)
        total += y.size(0)

        mid_ent_cms[0] += torch.sum(h, dim=0, keepdim=True) # [1,d]

        ####################################################
        y_onehot = F.one_hot(y, num_classes=logits.shape[-1]) # [b] -> [b, c]
        y_onehot = torch.unsqueeze(y_onehot, dim=-1).to(args.device) # [b, c] -> [b, c, 1]
        h_ext = torch.unsqueeze(h, dim=1) # [b, d] -> [b, 1, d]
        mid_cls_cms[0] += torch.sum(y_onehot * h_ext, dim=0, keepdim=True) # [b, c, d] -> [1, c, d]
        total_mid_cls_cms += torch.sum(y_onehot, dim=0, keepdim=True) # [1, c, 1]
        ####################################################

    mid_ent_cms[0] = copy.deepcopy(mid_ent_cms[0] / total)
    mid_cls_cms[0] = copy.deepcopy(mid_cls_cms[0] / total_mid_cls_cms)
    
    total = 0
    total_mid_cls_cms = 0
    for i, data in enumerate(train_loader, 0):
        if i % 100 == 0: print_current_time(i)
        x, y = data
        x, y = x.to(args.device), y.to(args.device)
        logits, h = model(x)
        h = h_std(args, h)
        total += y.size(0)
        
        ####################################################
        y_onehot = F.one_hot(y, num_classes=logits.shape[-1]) # [b] -> [b, c]
        y_onehot = torch.unsqueeze(y_onehot, dim=-1).to(args.device) # [b, c] -> [b, c, 1]
        h_ext = torch.unsqueeze(h, dim=1) # [b, d] -> [b, 1, d]
        total_mid_cls_cms += torch.sum(y_onehot, dim=0, keepdim=True) # [1, c, 1]
        ####################################################

        for j in range(args.save_max_moment):
            if j > 0:
                mid_ent_cms[j] += torch.sum(torch.pow(h - mid_ent_cms[0], j+1), dim=0, keepdim=True)
                mid_cls_cms_add = torch.sum(torch.pow(y_onehot * h_ext - y_onehot * mid_cls_cms[0], 
                                                      j+1), dim=0, keepdim=True) # [1,c,d]
                mid_cls_cms[j] += mid_cls_cms_add # [5][1,c,d]
    
    for j in range(args.save_max_moment):
        if j > 0:
            mid_ent_cms[j] = mid_ent_cms[j] / total
            mid_cls_cms[j] = mid_cls_cms[j] / total_mid_cls_cms
    
    torch.save({'cmd_base_mid': mid_ent_cms, \
                'cmd_base_mid_cls': mid_cls_cms} , args.cm_file)

def load_cmd_statistics(args):
    
    statistics = {}
    statistics['cmd_base_mid'] = torch.load(args.cm_file, map_location='cuda:0')['cmd_base_mid']
    statistics['cmd_base_mid_cls'] = torch.load(args.cm_file, map_location='cuda:0')['cmd_base_mid_cls']
    print("cmd_base_mid_mean: " + str(statistics['cmd_base_mid'][0][:,:20]))
    print("cmd_base_mid_var: " + str(statistics['cmd_base_mid'][1][:,:20]))
    print("cmd_base_mid_cls_mean: " + str(statistics['cmd_base_mid_cls'][0][:, :3, :20]))
    print("cmd_base_mid_cls_var: " + str(statistics['cmd_base_mid_cls'][1][:, :3, :20]))
    
    return statistics
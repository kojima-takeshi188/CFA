from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torchvision import transforms

import io
import os
import torch
import numpy as np
import argparse

from utils_data import *
from utils_adapt import *
from utils_model import *
from utils_statistics import *
from utils import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', default='imagenet', type=str, choices=['imagenet'])
parser.add_argument('--dataloader_drop_last', action='store_true')
parser.add_argument('--image_crop_size', default=256, type=int, choices=[256, 384, 512])
parser.add_argument('--image_size', default=224, type=int, choices=[224, 384, 512])
parser.add_argument('--model', default='ViT-B_16', type=str, choices=['ViT-B_16', 'ViT-L_16', 'ViT_AugReg-B_16', 'ViT_AugReg-L_16', 'resnet50', 'resnet101', 'mlpmixer_B16', 'mlpmixer_L16', 'DeiT-B', 'DeiT-S', 'Beit-B16_224', 'Beit-L16_224'])
parser.add_argument('--learning_rate_test', default=0.001, type=float)
parser.add_argument('--sgd_momentum_test', default=0.9, type=float)
parser.add_argument('--weight_decay_test', default=0.0, type=float)
parser.add_argument('--adapt_optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
parser.add_argument('--minibatch_size_test', default=64, type=int)
parser.add_argument('--dropout_on_flag', action='store_true')
parser.add_argument('--clip_grad_off', action='store_true')
parser.add_argument('--adapt_steps_per_sample', default=1, type=int)
parser.add_argument('--max_grad_norm', default=1.0, type=float)
parser.add_argument('--method', default='cfa', type=str, choices=['cfa', 't3a', 'tent', 'pl', 'shot-im', 'source'])
parser.add_argument('--t3a_filter_k', default=100, type=int)
parser.add_argument('--value_max', default=1.0, type=float)
parser.add_argument('--value_min', default=-1.0, type=float)
parser.add_argument('--lambda1', default=1.0, type=float)
parser.add_argument('--lambda2', default=1.0, type=float)
parser.add_argument('--save_max_moment', default=5, type=int)
parser.add_argument('--full_max_moment', default=3, type=int)
parser.add_argument('--cls_max_moment', default=1, type=int)
parser.add_argument('--adapt_parameters', default='partial', type=str, choices=['partial', 'partial_cls', 'partial_feature', 'all'])
parser.add_argument('--h_std_version', default=1, type=int)
parser.add_argument('--random_seed', default=1, type=int)
parser.add_argument('--strict_fix_of_dataloader_seed_flag', action='store_true')
parser.add_argument('--specify_corruption_type', default=None, type=str, choices=['gaussian_noise', 'shot_noise', 'impulse_noise','defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur','snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', 'sketch', 'mnist', 'mnistm'])
parser.add_argument('--specify_corruption_severity', default=None, type=int, choices=[1, 2, 3, 4, 5])
parser.add_argument('--save_image_sample_flag', action='store_true')
parser.add_argument('--save_image_sample_path', default="./sample_image/", type=str)
parser.add_argument('--model_save_path', default="./save_model/", type=str)
parser.add_argument('--data_path', default="./datasets/", type=str)
parser.add_argument('--max_num_worker', default=18, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--show_error_rate_transition', action='store_true')
parser.add_argument('--calc_statistics_flag', action='store_true')
parser.add_argument('--tta_flag', action='store_true')
args = parser.parse_args()

if args.h_std_version == 1:
    suffix = ""
else:
    raise ValueError("h_std_version is not properly defined !!!")

if args.dataset == "imagenet":
    args.num_classes = 1000
    if args.specify_corruption_type is None:
        CORRUPTION_TYPE = \
          ['gaussian_noise', 'shot_noise', 'impulse_noise',
          'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
          'snow', 'frost', 'fog', 'brightness', 'contrast',
          'elastic_transform', 'pixelate', 'jpeg_compression']
    else:
        CORRUPTION_TYPE = [args.specify_corruption_type]
    if args.specify_corruption_severity is None:
        CORRUPTION_SEVERITY = [5, 4, 3, 2, 1]
    else:
        CORRUPTION_SEVERITY = [int(args.specify_corruption_severity)]
else:
    raise ValueError("dataset is not properly defined !!!")

args.model_save_file = args.model_save_path + args.model + "." + args.dataset + ".model"
args.cm_file = args.model_save_path + args.model + "." + args.dataset + ".statistics" + suffix

def main(args):
  print(args)
  print(args.dataset)
  print(args.method)
  print(CORRUPTION_TYPE)
  print(CORRUPTION_SEVERITY)

  fix_seed(args.random_seed)
  
  try:
    os.makedirs(args.model_save_path)
  except FileExistsError:
    pass
  
  model = None
  
  minibatch_size_target = args.minibatch_size_test
  print(minibatch_size_target)
  
  if args.calc_statistics_flag:
      
      print("**********************************")
      print("Fine Tuning on Source Data (Skip)")
      print("**********************************")
      if os.path.exists(args.model_save_file) == False:
        print("Preparing pre-trained model on ImageNet-2012 ...")
        model = construct_model(args)
        print("Done.")
        print("Fine-Tuning is skipped ...")
        save_model_to_file(model, args.model_save_file)
        print("Model Saved to file...")

      print("**********************************")
      print("Prepare Data Loader")
      print("**********************************")
      source_train_loader, source_test_loader = setup_data_loader(args, minibatch_size_target, args.dataset)

      print("**********************************")
      print("Source Data : Train")
      print("Calculate Statistics")
      print("**********************************")
      del model
      model = load_model_from_saved_file(args)
      print(args.cm_file)
      if os.path.exists(args.cm_file) == False:
          calc_cmd_statistics(args, model, source_train_loader)
      statistics = load_cmd_statistics(args)

      print("**********************************")
      print("Source Data : Test")
      print("Calculate Error Rate")
      print("**********************************")
      error_rate = testdata_adapt_and_evaluation(args, model, source_test_loader, statistics, adapt_flag=False)
      print("Top-1 Error Rate: " + str(error_rate) + str(" %"))
  
  if args.tta_flag:
      
      print("**********************************")
      print("Load Statistics")
      print("**********************************")
      # Load statistics of source data from the saved file ...
      statistics = load_cmd_statistics(args)
      
      print("**********************************")
      print("Target Data : ")
      print("**********************************")
      for j in range(len(CORRUPTION_SEVERITY)):

        errorrate_list = []
        for i in range(len(CORRUPTION_TYPE)):    

          print("**********************************")
          print("CORRUPTION_TYPE: " + CORRUPTION_TYPE[i])
          print("CORRUPTION_SEVERITY: " + str(CORRUPTION_SEVERITY[j]))
          print("**********************************")
          if args.dataset == "imagenet":
              target_loader = setup_data_loader(args, minibatch_size_target, "imagenet-c", \
                              corruption_name=CORRUPTION_TYPE[i], severity=CORRUPTION_SEVERITY[j])

          # if 'source', no adaptation .
          if args.method == 'source':
            adapt_flag = False
          else:
            adapt_flag = True
          
          # Adaptation & Evaluation ...
          del model
          model = load_model_from_saved_file(args)
          error_rate = testdata_adapt_and_evaluation(args, model, target_loader, statistics, adapt_flag=adapt_flag)
          errorrate_list.append(error_rate)
          print("Top-1 Error Rate: " + str(error_rate))

        print("dataset: " + args.dataset)
        print("corruption_severity: " + str(CORRUPTION_SEVERITY))
        print("model: " + args.model)
        print("method: " + args.method)
        print("adapt_parameters: " + args.adapt_parameters)
        print("learning_rate_test: " + str(args.learning_rate_test))
        print("random_seed: " + str(args.random_seed))

        print("errorrate_list")
        print(errorrate_list)

#############################
# Execute Main Function
#############################
main(args)

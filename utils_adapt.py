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
import gc
from utils import *
from datetime import timedelta

celoss = nn.CrossEntropyLoss(reduction='none')
logsoftmax = nn.LogSoftmax(dim=-1)
softmax = nn.Softmax(dim=-1)

##############################
# Function Setting
##############################

def configure_model_for_resnet(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model

def cmd_optimization(args, h, batch_dim, statistics, value_min=None, value_max=None):
    
    loss = 0
    moments = [0] * args.full_max_moment
    for j in range(args.full_max_moment):
        if j == 0:
            moment = torch.mean(h, dim=batch_dim, keepdim=True) # [L, 1, h, q]
        else:
            moment = torch.mean(torch.pow(h - moments[0], j+1), dim=batch_dim, keepdim=True) # [L,1,h,q]
        moments[j] = moment
        loss += torch.sqrt(torch.sum(torch.square(moments[j] - statistics[j]))) \
                / (((value_max-value_min) ** (j+1)))
        
    return loss

def cmd_optimization_for_cls(args, logits, h, batch_dim, statistics, value_min=None, value_max=None, loss_ver=1):

    h = torch.unsqueeze(h, dim=1) # [b, d] -> [b, 1, d]
    num_class = logits.shape[-1]
    
    argmax_class = F.one_hot(torch.argmax(logits, dim=1), num_classes=num_class) # [b, c] -> [b] -> [b, c]        
    argmax_class = argmax_class.clone().detach().to(args.device)
    argmax_class = torch.unsqueeze(argmax_class, dim=-1) # [b, c] -> [b, c, 1]
    argmax_class_num = torch.clamp(torch.sum(argmax_class, dim=0), min=1) #  [1, c, 1]
    argmax_class_idx = torch.clamp(torch.sum(argmax_class, dim=0), min=0, max=1) #  [1, c, 1]
    
    loss = 0
    moments = [0] * args.cls_max_moment
    for j in range(args.cls_max_moment):
        if j == 0:
            moment = torch.sum(argmax_class * h, dim=batch_dim, keepdim=True) / argmax_class_num # [1, c, d]
        else:
            moment = torch.sum(torch.pow(argmax_class * h - argmax_class * moments[0], j+1), \
                                dim=batch_dim, keepdim=True) / argmax_class_num
        moments[j] = moment
        if loss_ver == 1: # For CFA ...
            loss_cls = torch.sqrt(torch.sum(torch.square(moments[j] - statistics[j]), dim=-1, keepdim=True)) # [1, c, 1]
            loss_cls = loss_cls / (((value_max-value_min) ** (j+1))) # [1, c, 1]
        else:
            raise ValueError("loss_ver is not properly defined in cmd_optimization_for_cls function !!!")
        loss += torch.sum(loss_cls * argmax_class_idx) / torch.sum(argmax_class_idx)
    
    return loss

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def testdata_adapt_and_evaluation(args, model, test_loader, statistics, adapt_flag):
    
    # in case of resnet, there is no dropout but bn statistics needs to be updated ...
    if adapt_flag and "resnet" in args.model:
        model = configure_model_for_resnet(model)
    
    if adapt_flag:
        if args.method == "t3a":
            model = T3A(args, model)
        else:
            parameters = collect_params(args, model)
            optimizer = set_optimizer(args, parameters)
            model = adapt_gradient_based_model(args, model, parameters, optimizer, statistics)
    else:
        model = without_adapt_model(args, model, statistics)        
        
    if args.dropout_on_flag: # Dropout ON
        model.train()
    else:  # Dropout OFF. Even if the model is set eval(), backward and gradient is possible ...
        model.eval()
    
    if adapt_flag and "resnet" in args.model:
        print("check the batch norm setting ...")
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                print(m)
    
    total = 0
    correct_list = []
    epoch_size = 1 # epoch size should be 1 for online adaptation ...
    for i in range(epoch_size):
        for nth_batch, data_test in enumerate(test_loader):
            x_test, y_test = data_test
            x_test, y_test = x_test.to(args.device), y_test.to(args.device)
            logits = model(x_test)
            _, predicted = torch.max(logits, -1)
            correct = (predicted == y_test).sum().item()
            correct_list.append(correct)
            total += y_test.size(0)
    error_rate = (1.0 - (sum(correct_list) * 1.0 / total)) * 100 # Top-1 Error Rate ...
    
    if args.save_image_sample_flag:
        save_image_sample(args, x_test)
    
    if args.show_error_rate_transition:
        print("error_rate_transition:")
        print(correct_list)
    
    return error_rate
    
class T3A(nn.Module):
    """
    Test Time Template Adjustments (T3A)
    
    """
    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = model
        self.classifier = model.classifier
        self.classifier.weight.requires_grad = False # To save memory ...
        self.classifier.bias.requires_grad = False # To save memory ...

        self.warmup_supports = self.classifier.weight.data
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=args.num_classes).float()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.filter_K = args.t3a_filter_k
        self.num_classes = args.num_classes
        self.softmax = torch.nn.Softmax(-1)
        
    def forward(self, x):
        with torch.no_grad():
            _, z = self.model(x)
        
        # online adaptation
        p = self.classifier(z)
        yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
        ent = softmax_entropy(p)

        # prediction
        self.supports = self.supports.to(z.device)
        self.labels = self.labels.to(z.device)
        self.ent = self.ent.to(z.device)
        self.supports = torch.cat([self.supports, z])
        self.labels = torch.cat([self.labels, yhat])
        self.ent = torch.cat([self.ent, ent])
        
        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        return z @ torch.nn.functional.normalize(weights, dim=0)

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s))))
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat==i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        
        return self.supports, self.labels

class without_adapt_model(nn.Module):
    def __init__(self, args, model, statistics):
        super().__init__()
        self.args = args
        self.model = model
        self.statistics = statistics
    def forward(self, x):
        # Prediction ...
        with torch.no_grad():
            logits, h = self.model(x)
        return logits

class adapt_gradient_based_model(nn.Module):
    def __init__(self, args, model, parameters, optimizer, statistics):
        super().__init__()
        self.args = args
        self.model = model
        self.parameters = parameters
        self.optimizer = optimizer
        self.statistics = statistics
        self.steps = args.adapt_steps_per_sample
        self.prev_moments = torch.zeros([0], device=args.device, dtype=torch.float)
        self.prev_logits = torch.zeros([0], device=args.device, dtype=torch.float)
        self.x_hat = torch.zeros([0], device=args.device, dtype=torch.float)
        assert self.steps > 0, "requires >= 1 step(s) to forward and update"

    def forward(self, x):
        for _ in range(self.steps):
            logits = self.forward_and_adapt(x)
        return logits

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        """
        Forward and adapt model on batch of data.
        Measure test-time adaptation loss, take gradients, and update params.
        """

        self.model.zero_grad()
        
        logits, h = self.model(x) # [b,c] [b,d]
        h = h_std(self.args, h)

        if self.args.method == "tent":
            loss = softmax_entropy_tent(logits) # [b] 
            loss = torch.mean(loss)

        elif self.args.method == "pl":
            argmax_class = torch.argmax(logits, dim=1) # [b, c] -> [b]
            argmax_class = argmax_class.clone().detach().to(self.args.device)
            loss = torch.mean(celoss(logits, argmax_class))

        elif self.args.method == "shot-im":
            loss1 = softmax_entropy_tent(logits)
            loss1 = torch.mean(loss1)
            loss2 = softmax_diversity_regularizer(logits)
            loss = loss1 + loss2
            
        elif self.args.method == "cfa":
            loss1 = 0
            loss2 = 0
            
            # Full moment matching ...
            if self.args.full_max_moment != 0:
                loss1 = cmd_optimization(self.args, h=h, batch_dim=0, \
                                           statistics=self.statistics["cmd_base_mid"], \
                                           value_min=self.args.value_min, value_max=self.args.value_max)

            # class-based moment matching ...
            if self.args.cls_max_moment != 0:
                loss2 = cmd_optimization_for_cls(self.args, logits=logits, h=h, batch_dim=0, \
                                                 statistics=self.statistics["cmd_base_mid_cls"], \
                                                 value_min=self.args.value_min, value_max=self.args.value_max)

            loss = (self.args.lambda1 * loss1) + (self.args.lambda2 * loss2)

        loss.backward()
        loss.detach()
        if not self.args.clip_grad_off:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        self.model.zero_grad()
        
        #with torch.no_grad():
        #    logits, _, _ = self.model(x)

        return logits.detach().to(self.args.device)

def set_optimizer(args, opt_parameters):
    if args.adapt_optimizer == "sgd":
        optimizer = torch.optim.SGD(opt_parameters,
                            lr=args.learning_rate_test,
                            momentum=args.sgd_momentum_test, #0.9, # 0.9
                            weight_decay=args.weight_decay_test)
    elif args.adapt_optimizer == "adam":
        optimizer = torch.optim.Adam(opt_parameters, 
                            lr=args.learning_rate_test,
                            betas=(0.9, 0.999), 
                            eps=1e-06)
    else:
        raise ValueError("Optimizer Setting Error !!!")
    return optimizer

def collect_params(args, model):
    """Collect the affine scale + shift parameters from batch/layer norms.
    Walk the model's modules and collect all batch/layer normalization parameters.
    Return the parameters and their names.
    """
    
    total_params = 0
    partial_params = 0
    opt_parameters = []
    for name, param in model.named_parameters():
        total_params += param.numel()
        if "ViT" in args.model:
            if args.adapt_parameters == 'all':
                    opt_parameters.append({'params' : param})
                    print(name)
                    param.requires_grad = True
                    partial_params += param.numel()
            elif args.adapt_parameters == 'partial':
                if ('norm' in name):
                    opt_parameters.append({'params' : param})
                    print(name)
                    param.requires_grad = True
                    partial_params += param.numel()
                else:
                    param.requires_grad = False
            elif args.adapt_parameters == 'partial_cls':
                if ('cls_token' in name):
                    opt_parameters.append({'params' : param})
                    print(name)
                    param.requires_grad = True
                    partial_params += param.numel()
                else:
                    param.requires_grad = False
            elif args.adapt_parameters == 'partial_feature':
                if ('head' not in name):
                    opt_parameters.append({'params' : param})
                    print(name)
                    param.requires_grad = True
                    partial_params += param.numel()
                else:
                    param.requires_grad = False
        elif "mlpmixer" in args.model:
            if args.adapt_parameters == 'partial':
                if ('norm' in name):
                    opt_parameters.append({'params' : param})
                    print(name)
                    param.requires_grad = True
                    partial_params += param.numel()
                else:
                    param.requires_grad = False
        elif "DeiT" in args.model:
            if args.adapt_parameters == 'partial':
                if ('norm' in name):
                    opt_parameters.append({'params' : param})
                    print(name)
                    param.requires_grad = True
                    partial_params += param.numel()
                else:
                    param.requires_grad = False
        elif "Beit" in args.model:
            if args.adapt_parameters == 'partial':
                if ('norm' in name):
                    opt_parameters.append({'params' : param})
                    print(name)
                    param.requires_grad = True
                    partial_params += param.numel()
                else:
                    param.requires_grad = False
        elif "resnet" in args.model:
            if args.adapt_parameters == 'partial':
                if ('bn' in name):
                    opt_parameters.append({'params' : param})
                    print(name)
                    param.requires_grad = True
                    partial_params += param.numel()
                else:
                    param.requires_grad = False
        else:
            raise ValueError("adapt_parameters error 1 !!!")
    
    print(args.model)
    print("total_params: {}".format(total_params))
    print("partial_params: {}".format(partial_params))
    
    if opt_parameters == []:
        raise ValueError("adapt_parameters error 2 !!!")
    
    return opt_parameters
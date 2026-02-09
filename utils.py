from scipy.fftpack import fft
from testers import *
import scipy.io as spio
import pickle
import numpy as np
import torch
import yaml
import copy
import os

def accuracy_k(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = min(max(topk), output.shape[1])
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def load_layer_config(args,model,task):
    # fixed_layer: shared layers for all tasks while config is set to be 0
    
    
    prune_ratios = {}
    pruned_layer = []

    # For fixed layer
    fixed_layer = []
    if args.dataset == 'cifar' or args.dataset == 'mixture':
        fixed_layer = ['module.fc1.bias']
    elif args.dataset == 'mnist':
        fixed_layer = ['module.fc1.bias','module.fc2.bias']

    # For output layer
    output_layer = []
    if args.dataset == 'cifar' or args.dataset == 'mixture':
        output_layer = ['module.fc2.weight','module.fc2.bias']
    elif args.dataset == 'mnist':
        output_layer = ['module.fc3.weight','module.fc3.bias']
    elif args.dataset in ['rfmls', 'dronerc']:
        # print( if name.contains("heads") else "")
        if args.arch == 'rfnet':
            output_layer = [name for name, _ in model.named_parameters() if "fc." in name]
        elif args.arch == "evidential":
            output_layer = [name for name, _ in model.named_parameters() if "ds" in name]
        else:
            output_layer = [name for name, _ in model.named_parameters() if "head" in name]
        print(output_layer)
        # ['ds_module.1.ds1_activate.eta.weight','ds_module.1.ds1_activate.xi.weight', 'ds_module.ds1_activate.xi.weight']
    # For pruned layer
    config_setting = list(map(float, args.config_setting.split(",")))


    if len(config_setting) == 1:
        sparse_setting = float(config_setting[0])
    else:
        sparse_setting = float(config_setting[task])/float(sum(config_setting))

    sparse_setting = 1-sparse_setting*args.config_shrink

    with torch.no_grad():
        for name, W in (model.named_parameters()):
            if args.dataset == 'cifar' or args.dataset == 'mixture':
                if 'weight' in name and name!='module.fc2.weight' and name not in fixed_layer:
                #if 'weight' in name and name!='module.fc3.weight' and name not in fixed_layer:
                    prune_ratios[name] = sparse_setting
                    pruned_layer.append(name)
            elif args.dataset == 'mnist':
                if 'weight' in name and name!='module.fc3.weight' and name not in fixed_layer:
                #if 'weight' in name and name!='module.fc3.weight' and name not in fixed_layer:
                    prune_ratios[name] = sparse_setting
                    pruned_layer.append(name)
            elif args.dataset == 'rfmls' or 'dronerc':
                if 'weight' in name and 'bn' not in name and 'ds' not in name and 'fc' not in name:
                    prune_ratios[name] = sparse_setting
                    pruned_layer.append(name)

    args.prune_ratios = prune_ratios
    print('Pruned ratio:',sparse_setting)
            
    args.pruned_layer = pruned_layer
    args.fixed_layer = fixed_layer
    args.output_layer = output_layer
    print('Pruned layer:',pruned_layer)
    print('Fixed layer:',fixed_layer)
    print('Output layer:',output_layer)
    return args

def model_loader(args):
    if args.adaptive_mask:
        # from models.masknet import CifarNet, MnistNet
        from models.masknet import ResNet50_1d, ResNet18_1d
        from models.bayes_resnet import Net
        from models.dst_resnet import Net as EvidentialNet
    else:
        # from models.cifarnet import CifarNet
        # from models.mnistnet import MnistNet
        from models.masknet import ResNet50_1d, ResNet18_1d
        from models.bayes_resnet import Net
        from models.bayesnet import BayesianClassifier
        from models.dst_resnet import Net as EvidentialNet


    if args.arch == 'rfnet' and args.multi_head:
        model = ResNet18_1d(args.input_size, args.classes, classes_per_task=[5,5])
    elif args.arch == 'rfnet':#and "radar" not in args.base_path.lower():
        # model = BayesianClassifier(input_size=args.input_size, num_classes=args.classes, args=args)
        model = ResNet18_1d(args.input_size, args.classes)
    elif args.arch == "evidential":
        model = EvidentialNet(input_size=args.input_size, num_classes=args.classes, args=args)
    elif args.arch == 'bayes_rfnet':
        model = Net(input_size=args.input_size, num_classes=args.classes, args=args)
    
 
    if args.multi_gpu:
        model = torch.nn.DataParallel(model)
    
    return model

def mask_joint(args,mask1,mask2):
    '''
    mask1 has more 1 than mask2
    return: new mask with only 0s and 1s
    '''

    masks = copy.deepcopy(mask1)
    if not mask2:
        return mask1
    for name in mask1:
        if name in args.output_layer:
            continue
        if name not in args.fixed_layer and name in args.pruned_layer:
            non_zeros1,non_zeros2 = mask1[name], mask2[name]
            non_zeros = non_zeros1 + non_zeros2
            
            # Fake float version of |
            under_threshold = non_zeros < 0.5
            above_threshold = non_zeros > 0.5
            non_zeros[above_threshold] = 1
            non_zeros[under_threshold] = 0
            
            masks[name] = non_zeros
    return masks

def mask_reverse(args, mask):
    mask_reverse = copy.deepcopy(mask)
    for name in mask:
        if name in args.pruned_layer:
            mask_reverse[name] = 1.0-mask[name]
    return mask_reverse

def set_model_mask(model,mask):
    '''
    mask:{non-zero:1 ; zero:0}
    '''
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, W in (model.named_parameters()):
            if name in mask:
                # mask_tensor = mask[name].to(W.device, dtype=W.dtype)
                # W.data *= mask_tensor
                # module_name, _, _ = name.rpartition(".")
                # module = modules.get(module_name, None)
                # if module is not None and hasattr(module, "w_mask"):
                #     module.w_mask.data = mask_tensor.to(
                #         module.w_mask.device, dtype=module.w_mask.dtype
                #     )
                W.data *= mask[name].cuda()

def get_model_mask(model=None):
    masks = {}
    for name, W in (model.named_parameters()):
        if 'mask' in name or "head" in name:
            continue
        weight = W.cpu().detach().numpy()
        non_zeros = (weight != 0)
        non_zeros = non_zeros.astype(np.float32)
        zero_mask = torch.from_numpy(non_zeros)
        W = torch.from_numpy(weight).cuda()
        W.data = W
        masks[name] = zero_mask
        #print(name,zero_mask.nonzero().shape)
    return masks

def cumulate_model(args, task):
    '''
    Cumulate models for individual task.
    '''
    state_dict = {}
    save_path = os.path.join(args.save_path_exp,'task'+str(task))
    
    #state_dict = torch.load(save_path + "/retrained.pt")
    # Trigger for experiment [leave space for future learning]
    if task < args.tasks-1:
        state_dict = torch.load(save_path + "/retrained.pt")
    else: # for last task
        state_dict = torch.load(save_path +"/{}{}.pt".format(args.arch, args.depth) )
            
    if 0 < task:
        save_path = os.path.join(args.save_path_exp,'task'+str(task-1))
        state_dict_prev = torch.load(save_path + "/cumu_model.pt")
        for name, param in state_dict_prev.items():
            if name in args.pruned_layer:
                state_dict[name].copy_(state_dict[name].data + param.data)
    
    save_path = os.path.join(args.save_path_exp,'task'+str(task))
    torch.save(state_dict, save_path+"/cumu_model.pt")

def set_adaptive_mask(model, reset=False, assign_value='', requires_grad=False):
    for name, W in (model.named_parameters()):
        if 'mask' in name:
            
            # set mask to be one
            if reset:
                weight = W.cpu().detach().numpy()
                W.data = torch.ones(weight.shape).cuda()
            
            # set mask to be given value
            elif assign_value:
                weight_name = name.replace('w_mask', 'weight')
                if weight_name in assign_value:
                    W.data = assign_value[weight_name].cuda()
                
            W.requires_grad = requires_grad
            
def load_state_dict(args, model, state_dict, target_keys=[], masks=[]):
    """Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. The keys of :attr:`state_dict` must
    exactly match the keys returned by this module's :func:`state_dict()`
    function.
    Arguments:
        state_dict (dict): A dict containing parameters and
            persistent buffers.
        masks: set target params to be 1.
    """
    own_state = model.state_dict()
    
    if target_keys:
        for name, param in state_dict.items():
            if name in target_keys:       # changed here
                if name not in own_state:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
                param = param.data
                own_state[name].copy_(param)
    elif masks:
        print('Loading layer...')
        for name, param in state_dict.items():
            if name in args.pruned_layer:     # changed here
                param = param.data
                param_t = own_state[name].data
                mask = masks[name].cuda()
                own_state[name].copy_(param + param_t*mask)
                #print(name)
    else:
        print('Loading layer...')
        for name, param in state_dict.items():
            if name not in own_state:     # changed here
                continue
            param = param.data
            own_state[name].copy_(param)
                
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.detach()
            if val.numel() == 1:
                val = val.item()
            else:
                val = val.mean().item()
        self.val = float(val)
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count

# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = min(max(topk), output.shape[1])
#         batch_size = target.size(0)

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))

#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res

def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    return need_hour, need_mins, need_secs

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.3 ** (epoch // args.lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


import torch
import torch.nn as nn

def check_loss_autograd(loss, name="loss"):
    if not isinstance(loss, torch.Tensor):
        raise TypeError(f"{name} is not a tensor — it's type {type(loss)}")

    if not loss.requires_grad:
        print(f"{name} does NOT require gradients — it might be detached from the graph!")

    if torch.isnan(loss).any():
        raise ValueError(f" NaN detected in {name}")

    if torch.isinf(loss).any():
        raise ValueError(f" Inf detected in {name}")

    if loss.dtype != torch.float32 and loss.dtype != torch.float16:
        print(f"{name} has unexpected dtype: {loss.dtype}")

    print(f"{name} passed all autograd checks.")

import math
import torch.nn.functional as F

class EvidentialLoss(nn.Module):
    def __init__(self, num_classes, kl_warmup_epochs=35, lmda=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.lmda = 10
        self.kl_warmup_epochs = kl_warmup_epochs

    def forward(self, E_preds, targets, beliefs=None, epoch=None):
        if E_preds.max() > 1.0 + 1e-4:
            print(f"Warning: E_preds max {E_preds.max().item():.6f} > 1.0")
        E = E_preds.clamp(1e-6, 1 - 1e-6)  # stability

        yk = F.one_hot(targets, num_classes=self.num_classes).float().to(E.device)

        # base EU BCE loss
        log_probs = yk * torch.log(E) + (1 - yk) * torch.log(1 - E)
        base = -torch.sum(log_probs, dim=1)  # [B]

        # KL( normalized utilities || uniform ) with soft gate
        U = E
        p = U / (U.sum(dim=1, keepdim=True) + 1e-8)
        K = U.size(1)
        kl_vals = (p * (p.add(1e-8).log())).sum(dim=1) + math.log(K)  # [B]

        u_max, _ = torch.max(E, dim=1)
        u_true = E.gather(1, targets.view(-1, 1)).squeeze(1)
        gate = u_max * (1.0 - u_true)
        kl = (kl_vals * gate).mean()

        kl_weight = self._kl_warmup_weight(epoch)
        # print(f"Base Loss: {base.mean().item():.4f}, KL Loss: {self.lmda*kl.item():.4f}")
        loss = (base + kl_weight * kl).mean()
        return loss

    def _kl_warmup_weight(self, epoch):
        if epoch is None or self.kl_warmup_epochs <= 0:
            return self.lmda
        if epoch <= 5:
            return 0.0
        if epoch >= self.kl_warmup_epochs:
            return self.lmda
        progress = float(epoch-5) / float(self.kl_warmup_epochs)
        return self.lmda * 0.5 * (1.0 - math.cos(math.pi * progress))

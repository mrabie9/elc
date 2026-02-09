from __future__ import print_function
import os
import sys
import logging
import argparse
import time
import yaml
import copy
from time import strftime
import torch
import torch.optim as optim
from torchvision import datasets, transforms

import models
import admm
from admm import GradualWarmupScheduler
from admm import CrossEntropyLossMaybeSmooth
from admm import mixup_data, mixup_criterion
from testers import *
from utils import *
from TrainValTest import CVTrainValTest, test_retrained_models, accuracy
np.set_printoptions(threshold=False)

DEFAULT_CONFIG_PATH = os.path.join("dronerc", "args.yaml")

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def check_and_create(dir_path):
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False
    
# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 admm training', fromfile_prefix_chars='@')
parser.add_argument('--config', type=str, default=None, help='Path to a YAML config file.')
parser.add_argument('--logger', type=str2bool, default=None, help='Enable or disable logger output.')
parser.add_argument('--arch', type=str, default=None, help='Network architecture identifier.')
parser.add_argument('--depth', type=int, default=None, help='Depth of the neural network.')
parser.add_argument('--s', type=float, default=None, help='Scale sparse rate.')
parser.add_argument('-j', '--workers', dest='workers', type=int, default=None, help='Number of data loading workers.')
parser.add_argument('--multi-gpu', dest='multi_gpu', type=str2bool, default=None, help='Enable multi-GPU training.')
parser.add_argument('--batch-size', dest='batch_size', type=int, default=None, help='Training batch size.')
parser.add_argument('--test-batch-size', dest='test_batch_size', type=int, default=None, help='Test batch size.')
parser.add_argument('--admm-epochs', dest='admm_epochs', type=int, default=None, help='Interval epochs to update ADMM.')
parser.add_argument('--optmzr', type=str, default=None, help='Optimizer to use.')
parser.add_argument('--lr', type=float, default=None, help='Learning rate.')
parser.add_argument('--lr-decay', dest='lr_decay', type=int, default=None, help='Epoch interval before LR decay.')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum.')
parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=None, help='Weight decay.')
parser.add_argument('--no-cuda', dest='no_cuda', type=str2bool, default=None, help='Disable CUDA.')
parser.add_argument('--seed', type=int, default=None, help='Random seed.')
parser.add_argument('--log-interval', dest='log_interval', type=int, default=None, help='Logging interval in batches.')
parser.add_argument('--save-model', dest='save_model', type=str, default=None, help='Path to save the model.')
parser.add_argument('--masked-retrain', dest='masked_retrain', type=str2bool, default=None, help='Enable masked retraining.')
parser.add_argument('--verbose', type=str2bool, default=None, help='Enable verbose logging.')
parser.add_argument('--admm', type=str2bool, default=None, help='Enable ADMM pruning.')
parser.add_argument('--rho', type=float, default=None, help='Initial rho value.')
parser.add_argument('--rho-num', dest='rho_num', type=int, default=None, help='Number of rho increases.')
parser.add_argument('--sparsity-type', dest='sparsity_type', type=str, default=None, help='Sparsity type to enforce.')
parser.add_argument('--combine-progressive', dest='combine_progressive', type=str2bool, default=None, help='Enable progressive combination pruning.')
parser.add_argument('--lr-scheduler', dest='lr_scheduler', type=str, default=None, help='Learning rate scheduler strategy.')
parser.add_argument('--warmup', type=str2bool, default=None, help='Enable LR warmup.')
parser.add_argument('--warmup-lr', dest='warmup_lr', type=float, default=None, help='Warmup learning rate.')
parser.add_argument('--warmup-epochs', dest='warmup_epochs', type=int, default=None, help='Number of warmup epochs.')
parser.add_argument('--mixup', type=str2bool, default=None, help='Enable mixup.')
parser.add_argument('--alpha', type=float, default=None, help='Alpha parameter for mixup.')
parser.add_argument('--smooth', type=str2bool, default=None, help='Enable label smoothing.')
parser.add_argument('--smooth-eps', dest='smooth_eps', type=float, default=None, help='Label smoothing epsilon.')
parser.add_argument('--no-tricks', dest='no_tricks', type=str2bool, default=None, help='Disable training tricks.')
parser.add_argument('--dataset', type=str, default=None, help='Dataset identifier.')
parser.add_argument('--task-folder', dest='task_folder', type=str, default=None, help='Relative task folder path.')
parser.add_argument('--exp-name', dest='exp_name', type=str, default=None, help='Experiment name.')
parser.add_argument('--base-path', dest='base_path', type=str, default=None, help='Base data directory.')
parser.add_argument('--save-path', dest='save_path', type=str, default=None, help='Directory for saving artifacts.')
parser.add_argument('--input-size', dest='input_size', type=int, default=None, help='Model input size.')
parser.add_argument('--classes', type=int, default=None, help='Number of classes.')
parser.add_argument('--tasks', type=int, default=None, help='Number of tasks.')
parser.add_argument('--disjoint_classifier', type=str2bool, default=None, help='Use disjoint classifier head.')
parser.add_argument('--mask', type=str, default=None, help='Mask checkpoint path.')
parser.add_argument('--config-file', dest='config_file', type=str, default=None, help='Additional config file.')
parser.add_argument('--config-setting', dest='config_setting', type=str, default=None, help='Manual prune ratios per task.')
parser.add_argument('--config-shrink', dest='config_shrink', type=float, default=None, help='Ratio of total model capacity to use.')
parser.add_argument('--heritage-weight', dest='heritage_weight', type=str2bool, default=None, help='Reuse previous task weights.')
parser.add_argument('--adaptive-mask', dest='adaptive_mask', type=str2bool, default=None, help='Enable adaptive mask learning.')
parser.add_argument('--admm-mask', dest='admm_mask', type=str2bool, default=None, help='Enable ADMM mask learning.')
parser.add_argument('--adaptive-ratio', dest='adaptive_ratio', type=float, default=None, help='Adaptive mask learning ratio.')
parser.add_argument('--epochs', type=int, default=None, help='Epochs to train the base model.')
parser.add_argument('--epochs-prune', dest='epochs_prune', type=int, default=None, help='Epochs to train during pruning.')
parser.add_argument('--epochs-mask-retrain', dest='epochs_mask_retrain', type=int, default=None, help='Epochs to retrain masks.')
parser.add_argument('--mask-admm-epochs', dest='mask_admm_epochs', type=int, default=None, help='Interval to update mask ADMM.')
parser.add_argument('--load-model', dest='load_model', type=str, default=None, help='Checkpoint to load pretrained model from.')
parser.add_argument('--load-model-pruned', dest='load_model_pruned', type=str, default=None, help='Checkpoint to load pruned model from.')
def original_args():
    pass
    # parser.add_argument('--logger', action='store_true', default=True,
    #                     help='whether to use logger')
    # parser.add_argument('--arch', type=str, default=None,
    #                     help='[vgg, resnet, convnet, alexnet]')
    # parser.add_argument('--depth', default=None, type=int,
    #                     help='depth of the neural network, 16,19 for vgg; 18, 50 for resnet')
    # parser.add_argument('--s', type=float, default=0.0001,
    #                     help='scale sparse rate (default: 0.0001)')
    # parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
    #                     help='number of data loading workers (default: 4)')
    # parser.add_argument('--multi-gpu', action='store_true', default=False,
    #                     help='for multi-gpu training')
    # parser.add_argument('--batch-size', type=int, default=128, metavar='N',
    #                     help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
    #                     help='input batch size for testing (default: 256)')
    # parser.add_argument('--admm-epochs', type=int, default=1, metavar='N',
    #                     help='number of interval epochs to update admm (default: 1)')
    # parser.add_argument('--optmzr', type=str, default='adam', metavar='OPTMZR',
    #                     help='optimizer used (default: adam)')
    # parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
    #                     help='learning rate (default: 0.1)')
    # parser.add_argument('--lr-decay', type=int, default=30, metavar='LR_decay',
    #                     help='how many every epoch before lr drop (default: 30)')
    # parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
    #                     help='SGD momentum (default: 0.9)')
    # parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
    #                     metavar='W', help='weight decay (default: 1e-4)')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=100, metavar='N',
    #                     help='how many batches to wait before logging training status')
    # parser.add_argument('--save-model', type=str, default="",
    #                     help='For Saving the current Model')
    # parser.add_argument('--masked-retrain', action='store_true', default=False,
    #                     help='for masked retrain')
    # parser.add_argument('--verbose', action='store_true', default=True,
    #                     help='whether to report admm convergence condition')
    # parser.add_argument('--admm', action='store_true', default=False,
    #                     help="for admm training")
    # parser.add_argument('--rho', type=float, default = 0.0001,
    #                     help ="define rho for ADMM")
    # parser.add_argument('--rho-num', type=int, default = 4,
    #                     help ="define how many rohs for ADMM training")
    # parser.add_argument('--sparsity-type', type=str, default='irregular',
    #                     help ="define sparsity_type: [irregular,column,filter,pattern,random-pattern]")
    # parser.add_argument('--combine-progressive', default=False, type=str2bool,
    #                     help="for filter pruning after column pruning")

    # parser.add_argument('--lr-scheduler', type=str, default='default',
    #                     help='define lr scheduler')
    # parser.add_argument('--warmup', action='store_true', default=False,
    #                     help='warm-up scheduler')
    # parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='M',
    #                     help='warmup-lr, smaller than original lr')
    # parser.add_argument('--warmup-epochs', type=int, default=0, metavar='M',
    #                     help='number of epochs for lr warmup')
    # parser.add_argument('--mixup', action='store_true', default=False,
    #                     help='ce mixup')
    # parser.add_argument('--alpha', type=float, default=0.0, metavar='M',
    #                     help='for mixup training, lambda = Beta(alpha, alpha) distribution. Set to 0.0 to disable')
    # parser.add_argument('--smooth', action='store_true', default=False,
    #                     help='lable smooth')
    # parser.add_argument('--smooth-eps', type=float, default=0.0, metavar='M',
    #                     help='smoothing rate [0.0, 1.0], set to 0.0 to disable')
    # parser.add_argument('--no-tricks', action='store_true', default=False,
    #                     help='disable all training tricks and restore original classic training process')

    # ########### For Dataset
    # parser.add_argument('--dataset', default='rfmls', type=str, help='Specify the dataset type [cifar;mnist]')
    # parser.add_argument('--exp-name', default='exprf6', type=str, help='Specify the experiment name')
    # parser.add_argument('--base-path', default='', type=str, help='Specify the data path')
    # parser.add_argument('--save-path', default='', type=str, help='Specify the save path')
    # parser.add_argument('--input-size', default=16382, type=int, help='Specify the input size')
    # parser.add_argument('--classes', default=10, type=int, help='Specify the number of classes')
    # ################## Lifelong Learning #####################
    # parser.add_argument('--tasks', type=int, default=10,help='number of tasks')
    # parser.add_argument('--mask', type=str, default="",help='loading cumulative Mask')
    # parser.add_argument('--config-file', type=str, default='', help ="config file name")
    # parser.add_argument('--config-setting', metavar='N', default='1', help ="If use manually setting, please set prune ratio for each task. Ex, for 5 tasks --config-setting 2,2,2,2,2")
    # parser.add_argument('--config-shrink', type=float, default=1, help ="set the ratio of total model capacity to use")

    # parser.add_argument('--heritage-weight', type=str2bool, default=False, help='use previous weights for current tasks')
    # parser.add_argument('--adaptive-mask', default=False, type=str2bool, help='adaptive learning the mask')
    # parser.add_argument('--admm-mask', default=False, type=str2bool, help='adaptive learning the mask')
    # parser.add_argument('--adaptive-ratio', default=1.0, type=float, help='adaptive learning the mask')

    # parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train base model (default: 160)')
    # parser.add_argument('--epochs-prune', type=int, default=30, metavar='N', help='number of epochs to train admm (default: 160)')
    # parser.add_argument('--epochs-mask-retrain', type=int, default=10, metavar='N', help='number of epochs to train mask (default: 160)')
    # parser.add_argument('--mask-admm-epochs', type=int, default=1, metavar='N', help='number of interval epochs to update mask admm ')

    # parser.add_argument('--load-model', type=str, default="", help='For loading exist pure trained Model')
    # parser.add_argument('--load-model-pruned', type=str, default="", help='For loading exist pruned Model')
    # parser.add_argument('@rfmls/more_data/2tasks/exprf6.yaml')

def load_yaml_config(config_path):
    """Load YAML configuration file into a dictionary."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file) or {}

def resolve_config_path(config_path):
    """Resolve a config path, accepting either a file or directory."""
    if not config_path:
        return None
    if str(config_path).lower() in {"none", "null"}:
        return None
    expanded = os.path.expanduser(config_path)
    if os.path.isdir(expanded):
        expanded = os.path.join(expanded, "args.yaml")
    return expanded

def apply_yaml_defaults(parser, yaml_config):
    """Apply YAML key/value pairs as argparse defaults."""
    if not yaml_config:
        return
    defaults = {}
    for key, value in yaml_config.items():
        defaults[key.replace('-', '_')] = value
    parser.set_defaults(**defaults)

def parse_args_with_yaml(parser, default_config_path=DEFAULT_CONFIG_PATH):
    """Parse CLI arguments, allowing YAML defaults and CLI overrides."""
    preliminary_args, _ = parser.parse_known_args()
    config_path = preliminary_args.config or default_config_path
    yaml_config = {}
    resolved_config_path = resolve_config_path(config_path)
    if resolved_config_path:
        if not os.path.exists(resolved_config_path):
            raise FileNotFoundError(f"Config file not found: {resolved_config_path}")
        yaml_config = load_yaml_config(resolved_config_path)
        apply_yaml_defaults(parser, yaml_config)
    args = parser.parse_args()
    args.config_path = resolved_config_path
    return args, yaml_config

def build_logs_dir(base_path, exp_name):
    components = ["logs"]
    if base_path:
        stripped = base_path.strip(os.sep)
        if stripped:
            components.extend(stripped.split(os.sep))
    components.append(exp_name)
    return os.path.join(*components)

args, yaml_config = parse_args_with_yaml(parser)
if not args.base_path:
    args.base_path = os.path.dirname(args.config_path) if args.config_path else ""
if not args.save_path:
    args.save_path = args.base_path
logs_dir = build_logs_dir(args.base_path, args.exp_name)
args.multi_head = True
classes_per_task = [5,5]#None #[5, 5, 5]

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.manual_seed(args.seed)
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
writer = None
print('Use Cuda:',use_cuda)
# ------------------ save path ----------------------------------------------
args.save_path_exp = os.path.join(args.save_path,args.exp_name)
check_and_create(args.save_path_exp)
setting_file = os.path.join(args.save_path_exp, args.exp_name+'.config')

# print("*************** Configuration ***************")
with open(setting_file, 'w') as f:
    args_dic = vars(args)
    for arg, value in args_dic.items():
        line = arg + ' : ' + str(value)
        # print(line)
        f.write(line+'\n')

# set up model archetecture
model = model_loader(args)
model.cuda()
# print(model)
""" disable all bag of tricks"""
if args.no_tricks:
    # disable all trick even if they are set to some value
    args.lr_scheduler = "default"
    args.warmup = False
    args.mixup = False
    args.smooth = False
    args.alpha = 0.0
    args.smooth_eps = 0.0


def prune(args, task, train_loader):
    
    """====================="""
    """ Initialize submask"""
    """====================="""
    
    if args.adaptive_mask:
        if task > 0:
            set_adaptive_mask(model, reset=True, requires_grad=True) # set initial as 1
            args.admm_mask = True
        else:
            set_adaptive_mask(model, reset=True, requires_grad=False)    
        
    """====================="""
    """ multi-rho admm train"""
    """====================="""
    
    args.admm, args.masked_retrain = True, False
    
    # Trigger for experiment [leave space for future learning]
    if args.admm_mask and task == args.tasks-1:
        args.admm = False
    admm_prune(args, args.mask, task, train_loader)

    """=============="""
    """masked retrain"""
    """=============="""
    args.admm, args.admm_mask, args.masked_retrain = False, False, True
    return masked_retrain(args, args.mask, task, train_loader)


def admm_prune(args, pre_mask, task, train_loader):
    
    """ 
    bag of tricks set-ups
    """
    initial_rho = args.rho
    # criterion = CrossEntropyLossMaybeSmooth(smooth_eps=args.smooth_eps).cuda()
    if args.multi_head:# "radar" in args.base_path.lower():
        num_classes_task = classes_per_task[args.current_task]
        criterion = torch.nn.CrossEntropyLoss().cuda() if args.arch != "evidential" else EvidentialLoss(num_classes=num_classes_task).cuda().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda() if args.arch != "evidential" else EvidentialLoss(num_classes=args.classes).cuda()
        # criterion = torch.nn.CrossEntropyLoss() # EvidentialLoss(num_classes=args.classes).cuda()

    args.smooth = args.smooth_eps > 0.0
    args.mixup = args.alpha > 0.0
    
    optimizer_init_lr = args.warmup_lr if args.warmup else args.lr
    optimizer = None
    if args.optmzr == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), optimizer_init_lr, momentum=0.9, weight_decay=1e-4)
    elif args.optmzr == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), optimizer_init_lr)
        
    '''
    Set learning rate
    '''
    scheduler = None
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_prune * len(train_loader), eta_min=4e-08)
    elif args.lr_scheduler == 'default':
        # my learning rate scheduler for cifar, following https://github.com/kuangliu/pytorch-cifar
        epoch_milestones = [65, 100, 130, 190, 220, 250, 280]

        """
        Set the learning rate of each parameter task to the initial lr decayed 
        by gamma once the number of epoch reaches one of the milestones
        """
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i * len(train_loader) for i in epoch_milestones], gamma=0.5)
    else:
        raise Exception("unknown lr scheduler")
        
    if args.warmup:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=args.lr / args.warmup_lr, total_iter=args.warmup_epochs * len(train_loader), after_scheduler=scheduler)
    
        
    # backup model weights
    if args.heritage_weight or args.adaptive_mask:
        model_backup = copy.deepcopy(model.state_dict())
    
    # get mask for training & set pre-trained (for previous tasks) weights to be zero   
    if pre_mask:
        pre_mask = mask_reverse(args, pre_mask)
        set_model_mask(model, pre_mask)
    

    '''
    if heritage or adaptive, copy weights back to model
    not for first task
    '''
    if args.heritage_weight or args.adaptive_mask:
        if args.mask: # second task onwards
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    if name in args.pruned_layer:
                        W.data += model_backup[name].data*args.mask[name].cuda()

    epoch_accuracies = []
    losses = []
    tr_acc = []
    '''
    Start Pruning...
    '''
    for i in range(args.rho_num):
        current_rho = initial_rho * 10 ** i
            
        if args.config_file:
            config = "./profile/" + args.config_file + ".yaml"
        elif args.config_setting:
            config = args.prune_ratios
        else:
            raise Exception("must provide a config setting.")
        ADMM = admm.ADMM(args, model, config=config, rho=current_rho)
        admm.admm_initialization(args, ADMM=ADMM, model=model)  # intialize Z variable

        # admm train
        best_prec1 = 0.

        for epoch in range(1, args.epochs_prune + 1):
            print("current rho: {}".format(current_rho))
            epoch_loss, epoch_acc = prune_train(args, pre_mask, ADMM, train_loader, criterion, optimizer, scheduler, epoch)
            losses.extend(epoch_loss)
            tr_acc.append(epoch_acc)
            
            prec1 = pipeline.test_model(args, model, mask=pre_mask, test=False)
            epoch_accuracies.append(prec1)
            best_prec1 = max(prec1, best_prec1)

            # if best_prec1 > 99:
            #     print(f"Early termination at epoch {epoch} with accuracy {best_prec1}")
            #     break   
            

        print("Best Acc: {:.4f}%".format(best_prec1))
        save_path = os.path.join(args.save_path_exp,'task'+str(task))
        torch.save(model.state_dict(), save_path+"/prunned_{}{}_{}_{}_{}_{}.pt".format(
            args.arch, args.depth, current_rho, args.config_file, args.optmzr, args.sparsity_type))
        
    losses = np.array([x.cpu().item() if torch.is_tensor(x) else x for x in losses])
    # print(epoch_accuracies)
    epoch_accuracies = np.array([x.cpu().item() if torch.is_tensor(x) else x for x in epoch_accuracies])
    # print(epoch_accuracies)
    tr_acc = np.array([x.detach().cpu().item() if torch.is_tensor(x) else x for sublist in tr_acc for x in sublist])
    os.makedirs(logs_dir, exist_ok=True)
    np.savez(os.path.join(logs_dir, "task" + str(task)+"-prune.npz"), losses=losses, epoch_acc=epoch_accuracies, tr_acc=tr_acc)     
        
        
                
def masked_retrain(args, pre_mask, task, train_loader):
    """ 
    bag of tricks set-ups
    """
    initial_rho = args.rho
    # criterion = CrossEntropyLossMaybeSmooth(smooth_eps=args.smooth_eps).cuda()
    if args.multi_head: #"radar" in args.base_path.lower():
        num_classes_task = classes_per_task[args.current_task]
        criterion = torch.nn.CrossEntropyLoss().cuda() if args.arch != "evidential" else EvidentialLoss(num_classes=num_classes_task).cuda().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda() if args.arch != "evidential" else EvidentialLoss(num_classes=args.classes).cuda()
        # criterion = torch.nn.CrossEntropyLoss() # EvidentialLoss(num_classes=args.classes).cuda()
    args.smooth = args.smooth_eps > 0.0
    args.mixup = args.alpha > 0.0

    optimizer_init_lr = args.warmup_lr if args.warmup else args.lr
    optimizer = None
    if args.optmzr == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), optimizer_init_lr, momentum=0.9, weight_decay=1e-4)
    elif args.optmzr == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), optimizer_init_lr)
        
    '''
    Set learning rate
    '''
    scheduler = None
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_mask_retrain * len(train_loader), eta_min=4e-08)
    elif args.lr_scheduler == 'default':
        # my learning rate scheduler for cifar, following https://github.com/kuangliu/pytorch-cifar
        epoch_milestones = [65, 100, 130, 190, 220, 250, 280]

        """
        Set the learning rate of each parameter task to the initial lr decayed 
        by gamma once the number of epoch reaches one of the milestones
        """
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i * len(train_loader) for i in epoch_milestones], gamma=0.5)
    else:
        raise Exception("unknown lr scheduler")
        
    if args.warmup:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=args.lr / args.warmup_lr, total_iter=args.warmup_epochs * len(train_loader), after_scheduler=scheduler)
    
    '''
    load admm trained model
    '''
    save_path = os.path.join(args.save_path_exp,'task'+str(task))
    print("Loading file: "+save_path+"/prunned_{}{}_{}_{}_{}_{}.pt".format(
        args.arch, args.depth, initial_rho * 10 ** (args.rho_num - 1), args.config_file, args.optmzr,
        args.sparsity_type))
    model.load_state_dict(torch.load(save_path+"/prunned_{}{}_{}_{}_{}_{}.pt".format(
        args.arch, args.depth, initial_rho * 10 ** (args.rho_num - 1), args.config_file, args.optmzr,
        args.sparsity_type)))
    
            
    if args.config_file:
            config = "./profile/" + args.config_file + ".yaml"
    elif args.config_setting:
        config = args.prune_ratios
    else:
        raise Exception("must provide a config setting.")
    ADMM = admm.ADMM(args, model, config=config, rho=initial_rho)
    best_prec1 = [0]
    best_mask = ''
    
    '''
    Deal with masks
    '''
    if args.heritage_weight or args.adaptive_mask:
        model_backup = copy.deepcopy(model.state_dict())
        
    if pre_mask:
        pre_mask = mask_reverse(args, pre_mask)
        #test_column_sparsity_mask(pre_mask)
        set_model_mask(model, pre_mask) # set mask by W x mask
    
    # Trigger for experiment [leave space for future learning]
    if task!=args.tasks-1:
        admm.hard_prune(args, ADMM, model) # prune weights
        
    if args.adaptive_mask and args.mask:
        admm.hard_prune_mask(args, ADMM, model) 
        
    current_trainable_mask = get_model_mask(model=model) # get new mask of non-zeros (i.e., weights to be trained for this task only)
    current_mask = copy.deepcopy(current_trainable_mask)
    submask = {}
                    
    # if heritage, restore weights
    if args.heritage_weight and args.mask:
        with torch.no_grad():
            for name, W in (model.named_parameters()):
                if name in args.pruned_layer:
                    W.data += model_backup[name].data*args.mask[name].cuda()
                    
    # if adaptive learning, copy selected weights back to model
    if args.adaptive_mask and args.mask:
        with torch.no_grad():
            
            # mask layer: previous tasks part {0,1}; remaining {0}
            for name, M in (model.named_parameters()):
                if 'mask' in name:
                    weight_name = name.replace('w_mask', 'weight')
                    submask[weight_name] = M.cpu().detach()
                    
            # copy selected weights back to model
            for name, W in (model.named_parameters()):
                if name in args.pruned_layer:
                    
                    '''
                    Reason why use args.mask instead of submask
                    1. easy to cumulate model weights, if use submask, then need to backup weights belong to args.mask-submask
                    2. weights 'selective' already achieved by mask layer (fixed during mask retrain)
                    '''
                    W.data += model_backup[name].data*args.mask[name].cuda()
            
            # combine submask and current trainable mask
            for name in submask:
                current_mask[name] += submask[name]
        
            # mask layer: previous tasks part {0,1}; remaining {1}
            for name, M in (model.named_parameters()):
                if 'mask' in name:
                    M.data = current_mask[name.replace('w_mask', 'weight')].cuda()
                        
        set_adaptive_mask(model, requires_grad=False)
    
    epoch_loss_dict = {}
    testAcc = []
    

    '''
    Start prunning
    '''
    # # Apply the hard mask to model weights before retraining
    # with torch.no_grad():
    #     for name, W in model.named_parameters():
    #         if name in current_trainable_mask:
    #             W.data *= current_trainable_mask[name].to(W.device)
    epoch_accuracies = []
    losses = []
    tr_acc = []
    for epoch in range(1, args.epochs_mask_retrain + 1):
        epoch_loss, epoch_acc = prune_train(args, pre_mask, ADMM, train_loader, criterion, optimizer, scheduler, epoch)
        losses.extend(epoch_loss)
        tr_acc.append(epoch_acc)
        prec1 = pipeline.test_model(args, model, mask=current_trainable_mask,test=False)
        epoch_accuracies.append(prec1)
        
        if prec1 >= max(best_prec1):
            #print("\n>_ Got better accuracy, saving model with accuracy {:.3f}% now...\n".format(prec1))
            torch.save(model.state_dict(), save_path+"/retrained.pt")                       
        
        testAcc.append(prec1)

        best_prec1.append(prec1)

        # if max(best_prec1) > 99:
        #     print(f"Early termination: Epoch: [{epoch}], Acc: {max(best_prec1)}%")
        #     break
        #print("current best acc is: {:.4f}".format(max(best_prec1)))

    print("Best Acc: {:.4f}%".format(max(best_prec1)))
    print('Pruned Mask sparsity')
    test_sparsity_mask(args, current_trainable_mask)

    losses = np.array([x.cpu().item() if torch.is_tensor(x) else x for x in losses])
    epoch_accuracies = np.array([x.cpu().item() if torch.is_tensor(x) else x for x in epoch_accuracies])
    tr_acc = np.array([x.detach().cpu().item() if torch.is_tensor(x) else x for sublist in tr_acc for x in sublist])
    os.makedirs(logs_dir, exist_ok=True)
    np.savez(os.path.join(logs_dir, "task" + str(task)+"-retrain.npz"), losses=losses, epoch_acc=epoch_accuracies, tr_acc=tr_acc)     
        
    return current_mask

def prune_train(args, pre_mask, ADMM, train_loader, criterion, optimizer, scheduler, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    idx_loss_dict = {}

    # switch to train mode
    model.train()
    
        
    end = time.time()
    losses_list = []
    epoch_acc = 0
    tr_acc = []
    for i, (input, target) in enumerate(train_loader):
        target = target.long().cuda()
        
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        if args.admm:
            admm.admm_adjust_learning_rate(optimizer, epoch, args)
        
        # optimizer.step()
            # scheduler.step()

        input=input.float().cuda()
        if torch.isnan(input).any():
            raise ValueError("Input to model contains NaNs")

        if args.mixup:
            input, target_a, target_b, lam = mixup_data(input, target, args.alpha)

        # compute output
        # output = model(input)
        if args.arch == 'bayes_rfnet':
            output, acc1 = model.observe(input, target, args.current_task, back_prop=False)
        elif args.arch == 'rfnet':
            if args.disjoint_classifier:
                output = model(input)
            else:
                output = model.forward(input, args.current_task)
        else:
            output, _, omega, beliefs = model(input, args.current_task, return_features = True) if args.multi_head else model(input, return_features = True) 
            output = output[:,:-1] # remove omega

        # if torch.isnan(output).any():
        #     raise ValueError("Model output contains NaNs")
        # if torch.isnan(beliefs).any():
        #     raise ValueError("Beliefs contain NaNs")


        if args.mixup:
            ce_loss = mixup_criterion(criterion, output, target_a, target_b, lam, args.smooth)
        
        elif 'rfnet' in args.arch:
            ce_loss = criterion(output, target)#, smooth=args.smooth)
        elif args.arch == 'evidential':
            ce_loss = criterion(output, target, beliefs)#, smooth=args.smooth)

        # if torch.isnan(ce_loss):
        #     print("Output:", torch.isnan(output).any())
        #     print("Target:", torch.isnan(target).any())
        #     print("Beliefs:", torch.isnan(beliefs).any())
        #     raise ValueError("ce_loss is NaN")

        mixed_loss = ce_loss
        
        if args.admm:
            admm.z_u_update(args, ADMM, model, device, train_loader, optimizer, epoch, input, i, writer)  # update Z and U
            ce_loss, admm_loss, mixed_loss = admm.append_admm_loss(args, ADMM, model, ce_loss)  # append admm loss
            # print("Z-U Total loss:", mixed_loss)
            # if ce_loss.abs().max() > 1e6:
            #     print(f"ln623 Large loss: {ce_loss}")
            # if torch.isnan(mixed_loss):
            #     raise ValueError("Total loss contains NaN")
            # if torch.isnan(ce_loss):
            #     raise ValueError("ce_loss contains NaN")
        if args.admm_mask:
            # print("admm_mask, before yk_update")
            admm.y_k_update(args, ADMM, model, device, train_loader, optimizer, epoch, input, i, writer) # update Y\K
            # print("admm_mask, after yk_update")
            ce_loss, admm_loss, mixed_loss = admm.append_mask_loss(args, ADMM, model, mixed_loss)
            # print("Y-K Total loss:", mixed_loss)
            # if ce_loss.abs().max() > 1e6:
            #     print(f"ln635 Large loss: {ce_loss}")
            # if torch.isnan(ce_loss):
            #     raise ValueError("ce_loss contains NaN")
    

        # compute gradient and do SGD step
        optimizer.zero_grad()

        with torch.autograd.set_detect_anomaly(True):
            if args.admm or args.admm_mask:
                mixed_loss.backward(retain_graph=True)
            else:
                ce_loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"NaN in gradient of {name}")
                if torch.isinf(param.grad).any():
                    print(f"Inf in gradient of {name}")

        if pre_mask: # W.grad is reverted with every optimizer step - reapply masks here
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    # shared layers
                    if name in args.fixed_layer:
                        W.grad *= 0
                        continue

                    if args.arch =="bayes_rfnet":
                        if name in args.output_layer and str(args.current_task) not in name:
                            # W.grad *= 0
                            continue
                        
                    # pruned weight layers: fix weight for previous task by setting grad to 0 if mask = 0    
                    if name in args.pruned_layer and name in pre_mask:
                        W.grad *= pre_mask[name].cuda()  
                        if torch.isnan(pre_mask[name]).any():
                            print(f"NaN in pre_mask {pre_mask[name]}") 
                    
                    if torch.isnan(W).any():
                        print(f"NaN in gradient of {name} after {pre_mask[name]}")
                    
                    # adaptively learn the mask: fix mask for trainable weight part
                    if args.adaptive_mask and 'mask' in name and args.admm:
                        W.grad *= args.mask[name.replace('w_mask', 'weight')].cuda()
                        if torch.isnan(args.mask[name.replace('w_mask', 'weight')]).any():
                            print(f"NaN in adaptive mask {args.mask[name.replace('w_mask', 'weight')]}") 
                    
                    if torch.isnan(W).any():
                        print(f"NaN in gradient of {name} after {args.mask[name.replace('w_mask', 'weight')]}")
                        
                        #W.grad *= 100    
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
                    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0:
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            # print('({0}) lr:[{1:.5f}]  '
            #       'Epoch: [{2}][{3}/{4}]\t'
            #       'Status: admm-[{5}] retrain-[{6}]\t'
            #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #       'Acc@1 {top1.val:.3f}% ({top1.avg:.3f}%)\t'
            #       .format(args.optmzr, current_lr,
            #        epoch, i, len(train_loader), args.admm, args.masked_retrain, batch_time=data_time, loss=losses, top1=top1))
        if i % 100 == 0:
            idx_loss_dict[i] = losses.avg

            # measure accuracy and record loss
        acc1, _ = accuracy_k(output, target, topk=(1,5)) if args.arch != "bayes_rfnet" else (acc1, None)
        epoch_acc += acc1.detach().cpu()/input.size(0) if torch.is_tensor(acc1) else acc1/input.size(0)

        losses.update(ce_loss.detach(), input.size(0))
        losses_list.append(ce_loss.detach().cpu())

    tr_acc.append(epoch_acc.detach().cpu() if torch.is_tensor(epoch_acc) else epoch_acc)
    top1.update(acc1.detach().cpu() if torch.is_tensor(acc1) else acc1, input.size(0))
        
    print('({0}) lr:[{1:.5f}]  '
                    'Epoch: [{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f}% ({top1.avg:.3f}%)\t'
                    .format('adam', current_lr,
                    epoch, len(train_loader), loss=losses, top1=top1))
    return losses_list, tr_acc #masks, idx_loss_dict

def train(args, pipeline, task, train_loader):
    print('*************** Training Model ***************')
    optimizer_init_lr = args.lr
    best_acc = 0
    
    # adaptive mask should be fixed during pure training
    if args.adaptive_mask:
        set_adaptive_mask(model, reset=True, requires_grad=False)
        
    optimizer = torch.optim.Adam(model.parameters(), optimizer_init_lr)
    # criterion = torch.nn.CrossEntropyLoss()
    if args.multi_head: #"radar" in args.base_path.lower():
        num_classes_task = classes_per_task[args.current_task]
        criterion = torch.nn.CrossEntropyLoss().cuda() if args.arch != "evidential" else EvidentialLoss(num_classes=num_classes_task).cuda().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda() if args.arch != "evidential" else EvidentialLoss(num_classes=args.classes).cuda()
    
    '''
    Set learning rate
    '''
    scheduler = None
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=4e-08)
    elif args.lr_scheduler == 'default':
        # my learning rate scheduler for cifar, following https://github.com/kuangliu/pytorch-cifar
        epoch_milestones = [65, 100, 130, 190, 220, 250, 280]

        """
        Set the learning rate of each parameter task to the initial lr decayed 
        by gamma once the number of epoch reaches one of the milestones
        """
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i * len(train_loader) for i in epoch_milestones], gamma=0.5)
    else:
        pass
        #adjust learning rate
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    scheduler = None
    # lr = optimizer_init_lr * (0.1 ** (epoch // 25))
    for param_group in optimizer.param_groups:
        param_group['lr'] = optimizer_init_lr
    
    train_losses = []
    tr_acc = []
    epoch_accuracies = []
    # start training
    for epoch in range(0, args.epochs):
        start = time.time()
        
        #check availability of the current mask
        _, epoch_loss, acc = pipeline.train_model(args, model, args.mask, train_loader, criterion, optimizer, scheduler, epoch)
        train_losses.append(epoch_loss)
        end_train = time.time()
        prec1 = pipeline.test_model(args, model, test=False)
        tr_acc.append(acc)
        epoch_accuracies.append(prec1)
        end_test = time.time()
        print("Training time: {:.3f}; Testing time: {:.3f}.".format(end_train-start, end_test-end_train))

        if prec1 >= best_acc:
            best_acc = prec1
            save_path = os.path.join(args.save_path_exp,'task'+str(task))
            check_and_create(save_path)
            torch.save(model.state_dict(), save_path+"/{}{}.pt".format(args.arch, args.depth))

        # if best_acc > 99:
        #     print(f"Early termination at epoch {epoch} with accuracy {best_acc}")
        #     break
    
    train_losses = np.array(train_losses)
    epoch_accuracies = np.array([x.cpu().item() if torch.is_tensor(x) else x for x in epoch_accuracies])
    tr_acc = np.array([x.detach().cpu().item() if torch.is_tensor(x) else x for sublist in tr_acc for x in sublist])
    os.makedirs(logs_dir, exist_ok=True)
    np.savez(os.path.join(logs_dir, "task" + str(task)+"-train.npz"), losses=train_losses, epoch_acc=epoch_accuracies, tr_acc=tr_acc)     
    print("Best Acc: {:.4f}%".format(best_acc))    
    
if __name__ == '__main__':

    # test_retrained_models(args, num_tasks=args.tasks, model_loader=model_loader, CVTrainValTest=CVTrainValTest)
    # print("Testing Done")
    
    '''
    Consecutively train a model with tasks of data.
    '''
    start_time = time.time()
    num_tasks = args.tasks

    # start_task = 1
    classes_offset = 0
    # for t in range(args.tasks):
    #     task_dir = os.path.join(args.save_path_exp, f'task{t}')
    #     if os.path.exists(os.path.join(task_dir, 'cumu_model.pt')):
    #         start_task = t + 1
    #     else:
    #         break

    # if start_task>0:
    #     # Now resume from start_task onward
    #     args.mask = pickle.load(open(os.path.join(args.save_path_exp, f'task{start_task - 1}', 'cumu_mask.pkl'), 'rb'))
    #     args.load_model = os.path.join(args.save_path_exp, f'task{start_task - 1}', 'cumu_model.pt')

    for task in range(num_tasks):
        print("\n\n*************** Training task {} ***************".format(task))
        ''' 
        load config (pruning) setting
        '''
        args = load_layer_config(args, model, task)
        args.current_task = task
        ''' 
        load-data 
        '''
        data_components = [args.base_path, args.task_folder, f'task{task}']
        base_path = os.path.join(*[component for component in data_components if component])
        save_path = os.path.join(args.save_path_exp, f'task{task}')

        pipeline = CVTrainValTest(base_path=base_path, save_path=save_path, num_tasks=args.tasks)
        if args.dataset == 'cifar':
            train_loader = pipeline.load_data_cifar(args.batch_size)
        elif args.dataset == 'mnist':
            train_loader = pipeline.load_data_mnist(args.batch_size)
        elif args.dataset == 'mixture':
            args, train_loader = pipeline.load_data_mixture(args)
        # elif args.dataset == 'rfmls':
        #     train_loader = pipeline.load_data_rfmls(args.input_size, args.batch_size, args.classes, task)
        else: #args.dataset == 'dronerc':
            train_loader, num_task_classes = pipeline.load_data_dronerc(args.batch_size, offset=classes_offset, args=args)
        

        """
        Pure train
        """

        if task == 0 and args.load_model_pruned:
            print('Loading pre-pruned model from: ', args.load_model_pruned)
        else:
            if task == 0 and args.load_model:
                print('Loading pretrained model from: ', args.load_model)    
            else:
                train(args, pipeline ,task, train_loader)
                args.load_model = save_path+"/{}{}.pt".format(args.arch, args.depth)
            load_state_dict(args, model, torch.load(args.load_model))


        '''
        Prune
        '''
        # Trigger for experiment [leave space for future learning]
        if task != num_tasks - 1:
            '''
            admm prunning based on basic model
            mask_for_current_task: pruned mask for current task i
            if adaptive_mask: mask_for_current_task = pruned + subset of cumulative mask
            '''
            if task == 0 and args.load_model_pruned:
                load_state_dict(args, model, torch.load(args.load_model_pruned)) # this will be saved as retrained
                mask_for_current_task = get_model_mask(model=model)
                torch.save(model.state_dict(), save_path+"/retrained.pt")
            else:
                mask_for_current_task = prune(args, task, train_loader)

            '''
            Get mask for this specific task
            '''
            print('Total Mask sparsity for task ', str(task))
            test_sparsity_mask(args,mask_for_current_task)

            cumulative_mask = mask_joint(args, mask_for_current_task, args.mask)
            args.mask = copy.deepcopy(cumulative_mask)
            #test_column_sparsity_mask(args.mask)


            with open(os.path.join(save_path,'mask.pkl'), 'wb') as handle:
                pickle.dump(mask_for_current_task, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(save_path,'cumu_mask.pkl'), 'wb') as handle:
                pickle.dump(cumulative_mask, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Trigger for experiment [leave space for future learning]
        else:
            print('Final task — skipping pruning and retraining.')
            # if args.adaptive_mask:
            #     print('Total Mask sparsity for task ', str(task))
            #     mask_for_current_task = prune(args, task, train_loader)
            #     test_sparsity_mask(args,mask_for_current_task)
            #     with open(os.path.join(save_path,'mask.pkl'), 'wb') as handle:
            #         pickle.dump(mask_for_current_task, handle, protocol=pickle.HIGHEST_PROTOCOL)

        '''
        Combine & Save Model
        '''

        '''
        save the best model to be cumulated model 
        for the last task, there is no pruning requirement, then save the best purely trained model
        '''
        if args.heritage_weight or args.adaptive_mask:
            if task != num_tasks-1: # Trigger for experiment [leave space for future learning]
                torch.save(torch.load(save_path + "/retrained.pt"), save_path+"/cumu_model.pt")
            else: # Trigger for experiment [leave space for future learning]
                torch.save(torch.load(save_path + "/{}{}.pt".format(args.arch, args.depth)), save_path+"/cumu_model.pt")
        else:
            cumulate_model(args, task) #cumulate pruned layers

        for name, W in model.named_parameters():
            if "head" in name:
                print(f"{name} mean: {W.data.mean()}, std: {W.data.std()}, grad: {W.grad.data.mean() if W.grad is not None else 'None'}")

        '''
        Test
        '''
        print("*************** Testing ***************")
        for i in range(task+1):

            classes_offset = i * args.classes/args.tasks
            args.current_task = i
            '''
            Load Data
            '''
            data_components = [args.base_path, args.task_folder, f'task{i}']
            base_path = os.path.join(*[component for component in data_components if component])
            save_path = os.path.join(args.save_path_exp,'task'+str(i))

            pipeline = CVTrainValTest(base_path=base_path, save_path=save_path, num_tasks=args.tasks)
            if args.dataset == 'cifar':
                train_loader = pipeline.load_data_cifar(args.batch_size)
            elif args.dataset == 'mnist':
                train_loader = pipeline.load_data_mnist(args.batch_size)
            elif args.dataset == 'mixture':
                args, _ = pipeline.load_data_mixture(args)
            # elif args.dataset == 'rfmls':
            #     train_loader = pipeline.load_data_rfmls(args.input_size, args.batch_size, args.classes, task)
            else: #args.dataset == 'dronerc':
                train_loader, _ = pipeline.load_data_dronerc(args.batch_size, offset=classes_offset, args=args)



            '''
            Load Model & Mask & Test
            '''

            # Trigger for experiment [leave space for future learning]
            if i != num_tasks - 1:
                if args.heritage_weight: # use previous weights for current tasks
                    trained_mask = pickle.load(open(save_path + "/cumu_mask.pkl",'rb'))
                else:
                    trained_mask = pickle.load(open(save_path + "/mask.pkl",'rb'))
                test_sparsity_mask(args,trained_mask)
            # else: # Trigger for experiment [leave space for future learning]
                # if args.adaptive_mask: # adaptive learning the mask
                #     trained_mask = pickle.load(open(save_path + "/mask.pkl",'rb'))
                #     test_sparsity_mask(args,trained_mask)   

            state_dict = torch.load(save_path + "/cumu_model.pt")
            model.load_state_dict(state_dict)



            if args.adaptive_mask:
                set_adaptive_mask(model, reset=True, requires_grad=False)


            print("Task{}: ".format(str(i)))
            # Trigger for experiment [leave space for future learning]\
            if i != num_tasks - 1:
                state_dict = torch.load(save_path + "/retrained.pt")
                load_state_dict(args, model, state_dict, target_keys=args.output_layer)
                prec1 = pipeline.test_model(args,model,trained_mask)
            else: # Trigger for experiment [leave space for future learning]
                if args.heritage_weight or args.adaptive_mask:
                    prec1 = pipeline.test_model(args,model)
                elif args.adaptive_mask:
                    prec1 = pipeline.test_model(args,model,trained_mask)
                else:
                    prec1 = pipeline.test_model(args,model,mask_reverse(args, args.mask))

            print(f"Testing: Task {i} precision = {prec1}")
        args.load_model = save_path+"/cumu_model.pt"
        model = model_loader(args)
        model.cuda()
        if args.fixed_layer:
            load_state_dict(args, model, state_dict, target_keys=args.fixed_layer)

        if args.heritage_weight or args.adaptive_mask:
            load_state_dict(args, model, torch.load(args.load_model), masks=mask_reverse(args, args.mask))
        else: # from zero
            set_model_mask(model, mask_reverse(args, args.mask))

        classes_offset += num_task_classes

    duration = time.time() - start_time
    need_hour, need_mins, need_secs = convert_secs2time(duration)
    print('total runtime: {:02d}:{:02d}:{:02d}'.format(need_hour, need_mins, need_secs))

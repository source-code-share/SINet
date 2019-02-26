import argparse
import json
import os
import sys
from pprint import pprint
import torch
import numpy as np
from easydict import EasyDict as edict


def parse_args():
    """
    Parse the arguments of the program
    :return: (config_args)
    :rtype: tuple
    """
    # Create a parser
    parser = argparse.ArgumentParser(description="MobileNet-V2 PyTorch Implementation")
    parser.add_argument('--version', action='version', version='%(prog)s 0.0.1')
    parser.add_argument('--config', default="../config/cifar100.json", type=str, help='Configuration file')

    # Parse the arguments
    args = parser.parse_args()

    # Parse the configurations from the config json file provided
    try:
        if args.config is not None:
            with open(args.config, 'r') as config_file:
                config_args_dict = json.load(config_file)
        else:
            print("Add a config file using \'--config file_name.json\'", file=sys.stderr)
            exit(1)

    except FileNotFoundError:
        print("ERROR: Config file not found: {}".format(args.config), file=sys.stderr)
        exit(1)
    except json.decoder.JSONDecodeError:
        print("ERROR: Config file is not a proper JSON file!", file=sys.stderr)
        exit(1)

    config_args = edict(config_args_dict)

    # pprint(config_args)
    # print("\n")

    return config_args


def create_experiment_dirs(exp_dir):
    """
    Create Directories of a regular tensorflow experiment directory
    :param exp_dir:
    :return summary_dir, checkpoint_dir:
    """
    experiment_dir = os.path.realpath(
        os.path.join(os.path.dirname(__file__))) + "/experiments/" + exp_dir + "/"
    summary_dir = experiment_dir + 'summaries/'
    checkpoint_dir = experiment_dir + 'checkpoints/'

    dirs = [summary_dir, checkpoint_dir]
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        print("Experiment directories created!")
        # return experiment_dir, summary_dir, checkpoint_dir
        return experiment_dir, summary_dir, checkpoint_dir
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)

def calc_dataset_stats(dataset, axis=0, ep=1e-7):
    return (np.mean(dataset, axis=axis) / 255.0).tolist(), (
            np.std(dataset + ep, axis=axis) / 255.0).tolist()

def saveModel(net,epoch,best_acc,model_path):
    print('saving.....')
    state={
        'net':net.state_dict(),
        'acc':best_acc,
        'epoch':epoch,
    }
    torch.save(state,model_path)

def loadModel(modelPath,net):
    print('==>Resuming from checkpoint...')
    checkpoint=torch.load(modelPath)
    net.load_state_dict(checkpoint['net'])
    best_acc=checkpoint['acc']
    start_epoch=checkpoint['epoch']
    return net,best_acc,start_epoch

def adjust_learning_rate(args,optimizer,epoch=0):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate * (args.lr_decay ** (epoch // args.step))            # args.lr = 0.1 , 即每30步，lr = lr /10
    for param_group in optimizer.param_groups:       # 将更新的lr 送入优化器 optimizer 中，进行下一次优化
        param_group['lr'] = lr

def adjust_batchsize(args,step, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    args.batchsize = int(args.batchsize * (0.5 ** (epoch // step)))            # args.lr = 0.1 , 即每30步，lr = lr /10
def adjust_batch(args,epoch):
    if args.train_batch>=64:
        args.train_batch = int(args.train_batch * (0.5 ** (epoch // 60)))

class AverageTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def count(Convds,Linears):
    MAdds1 = 0
    MAdds2 =0
    Params1=0
    Params2 = 0
    for k in Convds:
        conv=Convds.get(k)
        # print(conv)
        in_planes=conv['in_planes']
        out_planes=conv['out_planes']
        kernel_size=conv['kernel_size']
        groups=conv['groups']
        in_featuremap=conv['in_featuremap']
        out_featuremap=conv['out_featuremap']
        param=in_planes*kernel_size*kernel_size*out_planes//groups
        # print(k,'的参数量：',param)
        Params1+=param
        madd=in_featuremap*in_featuremap*\
             in_planes*kernel_size*kernel_size*out_planes\
             //groups
        # print(k, '的计算量：', madd)
        MAdds1+=madd

    for k in Linears:
        linear=Linears.get(k)
        in_planes = linear['in_planes']
        out_planes = linear['out_planes']
        param = in_planes * out_planes
        Params2 += param
        madd=param
        MAdds2 += madd
    print('总的参数量：', Params1 + Params2)
    print('Conv的计算量：', MAdds1)
    print('Linear的计算量：', MAdds2)
    print('总的计算量：', MAdds1 + MAdds2)

def get_valofconv(Convds,conv_name,in_planes,out_planes
                  ,kernel_size=3,stride=1,groups=1,padding=1,
                  in_featuremap=32,out_featuremap=32):
    conv = {
        # 'conv_name':'conv_',
        'in_planes': 0,
        'out_planes': 0,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'groups': 1,
        'in_featuremap': 0,
        'out_featuremap': 0,
    }
    # conv_name= 'conv_' + str(conv_num)+'_'+str(conv_num_num)
    # conv['conv_name']= 'conv_' + str(conv_num)+'_'+str(conv_num_num)
    conv['in_planes']=in_planes
    conv['out_planes'] = out_planes
    conv['kernel_size'] = kernel_size
    conv['stride'] = stride
    conv['groups'] = groups
    conv['padding'] = padding
    conv['in_featuremap'] = in_featuremap
    conv['out_featuremap'] = out_featuremap
    Convds[conv_name]=conv

def get_outfeature(in_feature,kernel,padding,stride):
    out_feature = (in_feature - kernel + 2 * padding) // stride + 1
    return out_feature

def get_valoflinear(Linears,linear_name,in_planes,out_planes):
    linear = {
        'in_planes': 0,
        'out_planes': 0,
    }
    linear['in_planes']=in_planes
    linear['out_planes'] = out_planes
    Linears[linear_name]=linear


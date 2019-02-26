import argparse
import shutil 
import time
import datetime
from tqdm import tqdm, trange
import random
from data.imagenet_data import get_loaders
from models.sinet import SINet
from opt.flops_benchmark import *
from models.mobile2_inverse import *
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from tensorboardX import SummaryWriter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
from opt.logger import CsvLogger
import sys
ROOT_PATH='../data/ILSVRC2012_img/'
def get_args():
    parser = argparse.ArgumentParser(description='Imagenet training with PyTorch')
    parser.add_argument('--dataroot', default=ROOT_PATH, metavar='PATH',
                        help='Path to ImageNet train and val folders, preprocessed as described in ')
    parser.add_argument('--gpus', default='0,1,2,3', help='List of GPUs used for training - e.g 0,1,3')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='Number of data loading workers (default: 4)')

    parser.add_argument('--epochs', default=400, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--train_batch', default=256, type=int, metavar='N',help='mini-batch size (default: 256)')
    parser.add_argument('-vb', '--test_batch', default=128, type=int, metavar='N',
                        help='mini-batch size (default: 256)')

    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The learning rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--weight_decay', '-d', type=float, default=4e-5, help='Weight decay (L2 penalty).')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma at scheduled epochs.')
    parser.add_argument('--schedule', type=int, nargs='+', default=[200, 300],help='Decrease learning rate at these epochs.')

    # Checkpoints
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='Just evaluate model')
    parser.add_argument('--save', '-s', type=str, default='', help='Folder to save checkpoints.')
    parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='Directory to store results')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',help='manual epoch number (useful on restarts)')
    parser.add_argument('--print_freq', '-p', type=int, default=100, metavar='N',help='Number of batches between log messages')
    parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: 1)')

    # Architecture
    parser.add_argument('--scaling', type=float, default=1, metavar='SC', help='Scaling of MobileNet (default x1).')
    parser.add_argument('--input_size', type=int, default=224, metavar='I',
                        help='Input size, multiple of 32 (default 224).')
    args = parser.parse_args()
    return args
claimed_acc_top1 = {224: {1.4: 0.75, 1.3: 0.744, 1.0: 0.718, 0.75: 0.698, 0.5: 0.654, 0.35: 0.603},
                    192: {1.0: 0.707, 0.75: 0.687, 0.5: 0.639, 0.35: 0.582},
                    160: {1.0: 0.688, 0.75: 0.664, 0.5: 0.610, 0.35: 0.557},
                    128: {1.0: 0.653, 0.75: 0.632, 0.5: 0.577, 0.35: 0.508},
                    96: {1.0: 0.603, 0.75: 0.588, 0.5: 0.512, 0.35: 0.455},
                    }
claimed_acc_top5 = {224: {1.4: 0.925, 1.3: 0.921, 1.0: 0.910, 0.75: 0.896, 0.5: 0.864, 0.35: 0.829},
                    192: {1.0: 0.901, 0.75: 0.889, 0.5: 0.854, 0.35: 0.812},
                    160: {1.0: 0.890, 0.75: 0.873, 0.5: 0.832, 0.35: 0.791},
                    128: {1.0: 0.869, 0.75: 0.855, 0.5: 0.808, 0.35: 0.750},
                    96: {1.0: 0.832, 0.75: 0.816, 0.5: 0.758, 0.35: 0.704},
                    }
def main():
    best_prec1 = 0.0
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpus:
        torch.cuda.manual_seed_all(args.seed)

    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = time_stamp
    save_path = os.path.join(args.results_dir, args.save)
    summary_path = os.path.join(save_path, 'log')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    writer = SummaryWriter(log_dir=summary_path)

    if args.gpus is not None:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        cudnn.benchmark = True
    # model = MobileNet2(input_size=args.input_size, scale=args.scaling)
    model = SINet(num_classes=1000, input_size=args.input_size, scale=args.scaling,add=True)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    print('number of parameters: {}'.format(num_parameters))
    # print('FLOPs: {}'.format(
    #     count_flops_test(SINet,use_cuda=True,input_size=args.input_size)))
    if args.gpus is not None:
        model = torch.nn.DataParallel(model, args.gpus)
        model=model.cuda()

    ##### step2: define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    ####step3：optionally resume from a checkpoint
    if args.resume:
        resume = os.path.join(args.results_dir, args.resume)
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        elif os.path.isdir(resume):
            checkpoint_path = os.path.join(resume, 'checkpoint.pth.tar')
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
    cudnn.benchmark = True

    #train_loader,train_sampler, val_loader,=load_data()
    print('----------------loading training and val data-------------------')
    train_loader, val_loader = get_loaders(args.dataroot, args.test_batch, args.train_batch,args.input_size,
                                           args.workers)
    # optionally resume from a checkpoint
    data = None
    csv_logger = CsvLogger(filepath=save_path, data=data)
    csv_logger.save_params(sys.argv, args)

    claimed_acc1 = None
    claimed_acc5 = None
    if args.input_size in claimed_acc_top1:
        if args.scaling in claimed_acc_top1[args.input_size]:
            claimed_acc1 = claimed_acc_top1[args.input_size][args.scaling]
            claimed_acc5 = claimed_acc_top5[args.input_size][args.scaling]
            csv_logger.write_text(
                'Claimed accuracies are: {:.2f}% top-1, {:.2f}% top-5'.format(claimed_acc1 * 100., claimed_acc5 * 100.))

    ### step5: 
    if args.evaluate:
        print('----------------validate model-------------------')
        validate(args,val_loader, model, criterion,0,writer) 
        return

    ##### step6:
    else:
        print('----------------training model-------------------')
        for epoch in range(args.start_epoch, args.epochs):
            start_time=time.time()
            adjust_learning_rate(args,optimizer,80,epoch)  

            # train for one epoch
            train_loss, train_prec1, train_prec5=train(args,train_loader, model, criterion, optimizer, epoch,writer)

            test_loss,prec1,prec5 = validate(args,val_loader, model, criterion,epoch,writer)

            csv_logger.write({'epoch': epoch + 1, 'val_error1': 1 - prec1, 'val_error5': 1 - prec5,
                              'val_loss': test_loss, 'train_error1': 1 - train_prec1,
                              'train_error5': 1 - train_prec5, 'train_loss': train_loss})

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({'epoch': epoch + 1,'arch': args.save, 'state_dict': model.state_dict(),
                'best_prec1': best_prec1,'optimizer': optimizer.state_dict(),
            }, is_best,save_path)
            print('save one time')
            one_epoch=time.time()-start_time
            print('one epoch training use time :', one_epoch)
            csv_logger.plot_progress(claimed_acc1=claimed_acc1, claimed_acc5=claimed_acc5,title=args.save)
        csv_logger.write_text('Best accuracy is {:.2f}% top-1'.format(best_prec1 * 100.))
        writer.close()

def train(args,train_loader, model, criterion, optimizer, epoch,writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:     # default=100
            tqdm.write('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
        writer.add_scalar('/train_loss',losses.avg,(epoch+1)*(i+1))
        writer.add_scalar('/train_top1', top1.avg, (epoch + 1) * (i + 1))
        writer.add_scalar('/train_top5', top5.avg, (epoch + 1) * (i + 1))
    return losses.avg,top1.avg,top5.avg

def validate(args,val_loader, model, criterion,epoch,writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        batch_time.update(time.time() - end)
    tqdm.write('Test: [{0}/{1}]\t'
               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
               'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
               'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        epoch, len(val_loader), batch_time=batch_time, loss=losses,
        top1=top1, top5=top5))
    if not args.evaluate:
        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/top1_acc', top1.avg, epoch)
        writer.add_scalar('val/top5_acc', top5.avg, epoch)

    return losses.avg,top1.avg,top5.avg

def save_checkpoint(state, is_best, filepath='./', filename='checkpoint.pth.tar'):
    save_path = os.path.join(filepath, filename)
    best_path = os.path.join(filepath, 'model_best.pth.tar')
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, best_path)

def adjust_learning_rate(args,optimizer,step, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // step))        
    for param_group in optimizer.param_groups:       
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    # Computes and stores the average and current value
    """
       batch_time = AverageMeter()
       即 self = batch_time
       则 batch_time 具有__init__，reset，update三个属性，
       直接使用batch_time.update()调用
       功能为：batch_time.update(time.time() - end)
               仅一个参数，则直接保存参数值
        对应定义：def update(self, val, n=1)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        这些有两个参数则求参数val的均值，保存在avg中##不确定##

    """
    def __init__(self):
        self.reset()       # __init__():reset parameters

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

if __name__ == '__main__':
    args = get_args()
    main()


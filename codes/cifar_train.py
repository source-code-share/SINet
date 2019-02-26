'''Train CIFAR with PyTorch.'''

import torch.optim as optim
import time
import torch.backends.cudnn as cudnn
from opt.utils import *


from data.load import *
from models import *
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--learning_rate','-lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--scale',default=1.0, type=float, help='scale')
parser.add_argument('-b', '--batchsize', default=128, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--c_tag','-c',default=1.0, type=float, help='c_tag')
parser.add_argument('--input_size',default=32, type=int, help='input_size')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--test', '-t', action='store_true', help='test')
parser.add_argument('--data', '-d',default='cifar100',type=str, help='load_data')
parser.add_argument('--save', '-s',default='mobilenetv2',type=str, help='save_model')
parser.add_argument('--gpus', default='0', help='List of GPUs used for training - e.g 0,1,3')
parser.add_argument('--lr_decay', type=float, default=0.1, help='Weight decay  for learning rate.')
parser.add_argument('--step', type=int, default=60, help='lr decrease after step')
parser.add_argument('--type', default='float32', help='Type of tensor: float32, float16, float64. Default: float32')
args = parser.parse_args()

def train(model, epoch, checkPoint, savePoint, modelPath,  curEpoch=0, best_acc = 0, useCuda=True,
          adjustLR = True, earlyStop=True, tolearnce=4):
    tolerance_cnt = 0
    tolerance_loss = 0
    step = 0
    if useCuda:
        model = model.cuda()
    ceriation = nn.CrossEntropyLoss()
    # ceriation=Weighted_LOSS()
    # optimizer = optim.Adam(net.parameters(), lr=args.lr)
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)

    for i in range(curEpoch, curEpoch+epoch):
        best_epoch=curEpoch
        model.train()
        # trainning
        sum_loss = 0
        # adjust_batchsize(args, 60, epoch=i)
        trainLoader, testLoader = load_data(args, batchSize=args.batchsize, device=6)
        if args.test:
            test(net,testLoader,True)
        else:
            for batch_idx, (x, target) in enumerate(trainLoader):
                optimizer.zero_grad()
                if adjustLR:
                    adjust_learning_rate(args,optimizer,i)
                if useCuda:
                    x, target = x.cuda(), target.cuda()
                x, target = Variable(x), Variable(target)
                out = model(x)
                loss = ceriation(out, target)
                sum_loss += loss.item()
                loss.backward()
                optimizer.step()
                step += 1
                writer.add_scalar('model/train_loss', loss.item(), (i + 1) * (batch_idx + 1))
                if (batch_idx + 1) % checkPoint == 0 or (batch_idx + 1) == len(trainLoader):
                    print('==>>> epoch: {}, batch index: {}, step: {}, train loss: {:.6f}'.format
                          (i, batch_idx + 1, step, sum_loss/(batch_idx+1)))

        acc = test(net,dataloade=testLoader,useCuda=True)
        # train_acc=test(net,trainLoader,useCuda=True)
        # writer.add_scalar('model/train_acc', train_acc, (i + 1))
        writer.add_scalar('model/test_acc', acc, (i + 1))

        # early stopping
        if earlyStop:
            if acc < best_acc:
                tolerance_cnt += 1
            else:
                best_acc = acc
                best_epoch=i
                tolerance_cnt = 0
                saveModel(model, best_epoch, best_acc, modelPath)

            if tolerance_cnt >= tolearnce:
                print("early stopping training....")
                saveModel(model, best_epoch, best_acc, modelPath)
                return
        else:
            if best_acc < acc:
                best_epoch=i
                saveModel(model, best_epoch, acc, modelPath)
                best_acc = acc
    writer.close()
    # saveModel(model, epoch, best_acc, modelPath)

def test(model,dataloade,useCuda=True):
    correct_cnt, sum_loss = 0, 0
    total_cnt = 0
    model.eval()

    if useCuda:
        model=model.cuda()
    # start=time.time()
    for batch_idx, (x, target) in enumerate(dataloade):
        with torch.no_grad():
            x, target = Variable(x), Variable(target)
        # x, target = Variable(x, volatile=True), Variable(target, volatile=True)
        if useCuda:
            x, target = x.cuda(), target.cuda()
        out = model(x)

        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()
        correct_cnt = correct_cnt.item()
    # duration=time.time()-start
    # print('test time is :',duration)

    acc = (correct_cnt * 1.0 / float(total_cnt))
    print("acc:", acc)
    return acc

def load_model(args):
    if args.data == 'cifar10':
        # net=VGG('VGG16')
        # net = ResNeXt29_2x64d()
        # net = MobileNet()
        # net = MobileNetV2(num_classes=10,scale=1.6)
        # net=ResNet50(num_classes=100)
        # net = DPN92()
        # net = ShuffleNetG2(num_classes=10)
        # net = SENet18()
        # net=ResNet101(num_classes=10)
        # net = NewModel24(num_classes=10)
        # net=ResNeXt29_2x64d()
        # net=inverse(num_classes=10,input_size=args.input_size)
        # net=pre_t(num_classes=10,input_size=args.input_size)
        net=pre_inverse(num_classes=10,input_size=args.input_size)
        # net=PreActResNet34()
        # net=DenseNet121(num_classes=10)
    elif args.data == 'cifar100':
        # net = VGG('VGG16')
        # net = ResNeXt29_2x64d()
        net = MobileNet(num_classes=100)
        # net = MobileNetV2(num_classes=100,scale=1.0)
        # net=SENet18(num_classes=100)
        # net=Inverse_M(num_classes=100,scale=1.0)
        # net = DPN92()
        # net = ShuffleNetG2(num_classes=100)
        # net = SENet18()
        # net=inverse(num_classes=100,input_size=args.input_size)
        # net = pre_t(num_classes=100, input_size=args.input_size)
        # net = pre_inverse(num_classes=100,input_size=args.input_size)
        # net = ResNet34(num_classes=100)
        # net = NewModel24(num_classes=100)
        # net = DenseNet121(num_classes=100)
        # net=SINet(num_classes=100,add=False)
    return net

if __name__ == '__main__':
    summary_dir = './result/log/' + args.save + '/'
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    print('log dir: ' + summary_dir)
    writer = SummaryWriter(log_dir=summary_dir)

    # Model
    save_path_root="/home/yaoyang/result/"
    # log_sum_dir=save_path_root+args.save_training_log+'.json'
    use_cuda = torch.cuda.is_available()

    model_path=save_path_root+args.save

    print('model_save_path is:',model_path)

    # net=load_model(args)
    net=M2(1.0,32,6,3,100)
    if args.gpus is not None:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        cudnn.benchmark = True


    if args.resume:
        print('==> loading model..')
        # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        net = torch.nn.DataParallel(net, device_ids=args.gpus)
        net,best_acc,curEpoch=loadModel(model_path,net)
    else:
        best_acc=0
        curEpoch=0
        print('==> Building model..')
        # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        net = torch.nn.DataParallel(net, device_ids=args.gpus)

    print('current epoch: ', curEpoch)
    print('current best acc: ', best_acc)
    use_cuda=torch.cuda.is_available()
    train(net,epoch=200,checkPoint=10,savePoint=500,modelPath=model_path,
      useCuda=use_cuda,best_acc=best_acc,adjustLR=True,curEpoch=curEpoch,earlyStop=False)


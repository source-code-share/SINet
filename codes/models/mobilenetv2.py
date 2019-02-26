'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from opt.utils import *

def split(x):
    n=int(x.size()[1])
    n1 = round(n*0.5)
    x1 = x[:, :n1, :, :].contiguous()
    x2 = x[:, n1:, :, :].contiguous()
    return x1, x2

def merge(x1,x2):
    return torch.cat((x1,x2),1)

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=False),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class BasicUnit(nn.Module):
    exchange=True
    def __init__(self, inplanes,outplanes,expansion=6,stride=1,c_tag=0.5):
        super(BasicUnit, self).__init__()
        self.stride=stride

        self.left_part_in = round(c_tag *inplanes)
        self.left_part_out = round(c_tag*outplanes)
        self.right_part_in = inplanes - self.left_part_in
        self.right_part_out= outplanes - self.left_part_out
        self.lout1 = round(c_tag * self.left_part_out)
        self.lout2 = self.left_part_out - self.lout1
        self.rout1 = round(c_tag * self.right_part_out)
        self.rout2 = self.right_part_out - self.rout1
        if stride==1 :
            self.conv_r=Block(self.right_part_in,self.rout1,expansion=expansion,stride=self.stride)
            self.conv_l=Block(self.left_part_in,self.lout1,expansion=expansion,stride=self.stride)
        else:
            self.conv_r = Block(self.right_part_in, self.right_part_out, expansion=expansion,stride=self.stride)
            self.conv_l = Block(self.left_part_in, self.left_part_out,expansion=expansion, stride=self.stride)

        self.shortcut_l=nn.Sequential()
        if stride==1 and self.left_part_in !=self.rout2:
            self.shortcut_l = nn.Sequential(
                nn.Conv2d(self.left_part_in,self.rout2,kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.rout2)
            )
        self.shortcut_r = nn.Sequential()
        if stride == 1 and self.right_part_in != self.lout2:
            self.shortcut_r = nn.Sequential(
                nn.Conv2d(self.right_part_in, self.lout2, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.lout2)
            )

    def forward(self, x):
        left,right=split(x)
        out_l = self.conv_l(left)
        out_r=self.conv_r(right)
        if self.stride==1:
            left = self.shortcut_l(left)
            right = self.shortcut_r(right)
            if self.exchange:
                out_r=torch.cat([out_r,left],dim=1)
                out_l=torch.cat([out_l,right],dim=1)
            else:
                out_r = torch.cat([out_r, right], dim=1)
                out_l = torch.cat([out_l, left], dim=1)
            out=merge(out_r,out_l)
        else:
            out = merge(out_r, out_l)
        return out

class Decision(nn.Module):
    def __init__(self, inplanes,outplanes,expansion,stride,add=[]):
        super(Decision, self).__init__()
        self.stride=stride
        self.add = add
        self.decision=nn.Sequential()
        if stride==1:
            self.block=Block(inplanes,outplanes,expansion,1)
        else:
            self.decision = nn.Sequential(
                SELayer(inplanes),
                nn.AdaptiveAvgPool2d(1)
            )
            self.block=Block(inplanes,outplanes,expansion,2)

    def forward(self, x):
        out=self.block(x)
        if self.stride==2:
            out1=self.decision(x)
            self.add.append(out1)
        return out

    def clean(self):
        while self.add:
            self.add.pop()
        return self.add

class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]
    add=[]
    def __init__(self, num_classes=10,input_size=224,scale=1.0):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.scale=scale
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.block=Decision

        self.dropout=nn.Dropout(p=0.2,inplace=True)
        self.layers,sum_planes = self._make_layers(self.block,in_planes=32)

        self.conv2 = nn.Conv2d(_make_divisible(320 * self.scale, 8), 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280+sum_planes, num_classes)
        self.init_params()

    def _make_layers(self, block,in_planes):
        layers = []
        sum_planes=0
        for expansion, out_planes, num_blocks, stride in self.cfg:
            out_planes = _make_divisible(out_planes * self.scale, 8)
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                if block==Decision:
                    if stride==2:
                        sum_planes+=in_planes
                    layers.append(block(in_planes, out_planes, expansion, stride,self.add))
                else:
                    layers.append(block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers),sum_planes

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)

        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        if self.block==Decision:
            # add = torch.cat([self.add[0], self.add[1], self.add[2], self.add[3]], dim=1)
            add = torch.cat([self.add[0], self.add[1], self.add[2]], dim=1)
            Decision.clean(self)
            out = torch.cat([add, out], dim=1)
        out=self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

from opt.flops_benchmark import *
def test():
    net = MobileNetV2(num_classes=100,input_size=32,scale=1.0)
    x = torch.randn(1, 3, 32, 32)
    y = net(Variable(x))
    print(y.size())
    # print_model_parm_nums(net)
    # count_flops_test(net, True, 224)
# test()


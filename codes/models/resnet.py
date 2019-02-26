'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
def split(x):
    n=int(x.size()[1])
    n1 = round(n*0.5)
    x1 = x[:, :n1, :, :].contiguous()
    x2 = x[:, n1:, :, :].contiguous()
    return x1, x2

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        out=x.view(N,g,int(C/g),H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)
        return out

def merge(x1,x2):
    return torch.cat((x1,x2),1)

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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicUnit(nn.Module):
    expansion = 1
    exchange=True
    def __init__(self, inplanes,planes,stride=1,c_tag=0.5):
        super(BasicUnit, self).__init__()
        self.stride=stride
        self.out_planes=self.expansion*planes
        self.left_part_in = round(c_tag *inplanes)
        self.left_part = round(c_tag*planes)

        self.right_part_in = inplanes - self.left_part_in
        self.right_part= planes - self.left_part
        if stride==1 :
            self.conv_r=Bottleneck(self.right_part_in,self.right_part_in,stride=self.stride)
            self.conv_l=Bottleneck(self.left_part_in,self.left_part_in,stride=self.stride)
        else:
            self.conv_r = Bottleneck(self.right_part_in, self.right_part, stride=self.stride)
            self.conv_l = Bottleneck(self.left_part_in, self.left_part, stride=self.stride)

        self.shortcut=nn.Sequential()
        if stride==1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes+inplanes,planes,kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        left,right=split(x)
        out_l = self.conv_l(left)
        out_r=self.conv_r(right)
        if self.stride==1:
            if self.exchange:
                out_r=torch.cat([out_r,left],dim=1)
                out_l=torch.cat([out_l,right],dim=1)
            else:
                out_r = torch.cat([out_r, right], dim=1)
                out_l = torch.cat([out_l, left], dim=1)
            out=merge(out_r,out_l)
            out=self.shortcut(out)
        else:
            out = merge(out_r, out_l)
        return out

class ResNet(nn.Module):
    add=True
    def __init__(self, block, num_blocks, num_classes=10,input_size=224):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.gb=nn.AdaptiveAvgPool2d(1)
        self.SE1 = SELayer(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.SE2 = SELayer(64)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.SE3 = SELayer(128)
        self.layer3= self._make_layer(block, 256, num_blocks[2], stride=2)
        self.SE4 = SELayer(256)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        #使用stride降采样来凝结会使得效果变差
        if self.add and num_classes==1000:
            sum_planes=64+64+128+256+512
        elif self.add and num_classes==100:
            sum_planes=64+128+256+512
        self.linear = nn.Linear(sum_planes, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out1 = self.gb(self.SE1(out))
        out = self.layer1(out)
        out2 = self.gb(self.SE2(out))

        out = self.layer2(out)

        out3 = self.gb(self.SE3(out))
        out = self.layer3(out)

        out4 = self.gb(self.SE4(out))
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        # out = torch.cat([out1,out2, out3, out4, out], dim=1)
        out = torch.cat([out2,out3, out4, out], dim=1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2],num_classes=num_classes)

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3],num_classes=num_classes)

def ResNet50(num_classes=10):
    return ResNet(BasicUnit, [3,4,6,3],num_classes=num_classes)

def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3,4,23,3],num_classes=num_classes)

def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3,8,36,3],num_classes=num_classes)


from opt.flops_benchmark import *
def test():
    net = ResNet34(num_classes=100)
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())
    # gpus = '0,1,2,3'
    # if gpus is not None:
    #     gpus = [int(i) for i in gpus.split(',')]
    #     device = 'cuda:' + str(gpus[0])
    # else:
    #     device = 'cpu'
    # dtype = torch.float32
    #
    # num_parameters = sum([l.nelement() for l in net.parameters()])
    # print('number of parameters: {:.2f}M'.format(num_parameters / 1000000))
    # count_flops_test(net, device, dtype, 224)


# test()

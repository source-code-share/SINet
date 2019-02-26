'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
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

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, stride):
        super(Block, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        return out

class BasicUnit(nn.Module):
    exchange=True
    def __init__(self, inplanes,outplanes,stride=1,c_tag=0.5):
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
            self.conv_r=Block(self.right_part_in,self.rout1,stride=self.stride)
            self.conv_l=Block(self.left_part_in,self.lout1,stride=self.stride)
        else:
            self.conv_r = Block(self.right_part_in, self.right_part_out, stride=self.stride)
            self.conv_l = Block(self.left_part_in, self.left_part_out, stride=self.stride)

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
    def __init__(self, inplanes,outplanes,stride,add=[]):
        super(Decision, self).__init__()
        self.stride=stride
        self.add = add
        self.decision=nn.Sequential()
        if stride==1:
            self.block=Block(inplanes,outplanes,1)
        else:
            self.decision = nn.Sequential(
                SELayer(inplanes),
                nn.AdaptiveAvgPool2d(1)
            )
            self.block=Block(inplanes,outplanes,2)

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

class MobileNet1(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,1), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    def __init__(self, num_classes=10,input_size=224):
        super(MobileNet1, self).__init__()
        self.gb=nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers,self.adds,self.add_index,sum_planes =self._make_layers(Block,32)
        self.linear = nn.Linear(1024+sum_planes, num_classes)

    def _make_layers(self,block, in_planes):
        layers = []
        adds=[]
        add_indx=[]
        sum_planes=0
        for i,out in enumerate(self.cfg):
            out_planes = out if isinstance(out, int) else out[0]
            stride = 1 if isinstance(out, int) else out[1]
            if stride==2:
                add_indx.append(i)
                sum_planes+=in_planes
                adds.append(nn.Sequential(
                        SELayer(in_planes),
                        nn.AdaptiveAvgPool2d(1)
                    ))
            layers.append(block(in_planes, out_planes,stride))
            in_planes = out_planes
        return layers,adds,add_indx,sum_planes

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        index=0
        add=[]
        for i in range(len(self.cfg)):
            if i in self.add_index:
                add.append(self.adds[index](out))
                index+=1
            out=self.layers[i](out)
        add=torch.cat([add[0],add[1],add[2]],dim=1)
        # add = torch.cat([add[0], add[1], add[2],add[3]], dim=1)
        out = F.avg_pool2d(out, 4)
        out=torch.cat([out,add],dim=1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,1), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    add=[]
    decision=True
    def __init__(self, num_classes=10,input_size=224):
        super(MobileNet, self).__init__()
        self.gb=nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        sum_planes=0
        # self.layers,sum_planes= self._make_layers(block=self.block, in_planes=32)
        self.gb=nn.AdaptiveAvgPool2d(1)
        self.layers1=Block(32, 64, 1)
        # if self.decision:
            # self.SE1 = SELayer(64)
            # sum_planes+=64
        self.layers2 = Block(64, 128, 1)#2
        self.layers3 = Block(128, 128, 1)
        if self.decision:
            self.SE2=SELayer(128)
            sum_planes+=128
        self.layers4 = Block(128, 256, 2)
        self.layers5 = Block(256, 256, 1)
        if self.decision:
            self.SE3 = SELayer(256)
            sum_planes+=256
        self.layers6 = Block(256, 512,2)
        self.layers7 = Block(512, 512, 1)
        self.layers8 = Block(512, 512, 1)
        self.layers9 = Block(512, 512, 1)
        self.layers10 = Block(512, 512, 1)
        self.layers11 = Block(512, 512, 1)
        if self.decision:
            self.SE4 = SELayer(512)
            sum_planes+=512
        self.layers12 = Block(512, 1024, 2)
        self.layers13 = Block(1024, 1024, 1)

        self.linear = nn.Linear(1024+sum_planes, num_classes)

    def _make_layers(self,block, in_planes):
        layers = []
        sum_planes=0
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            if block==Decision:
                if stride==2:
                    sum_planes+=in_planes
                layers.append(block(in_planes, out_planes, stride,self.add))
            else:
                layers.append(block(in_planes, out_planes,stride))
            in_planes = out_planes
        return layers,sum_planes

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = self.layers(out)
        out = self.layers1(out)
        # if self.decision:
            # add1=self.gb(self.SE1(out))
        out=self.layers2(out)
        out = self.layers3(out)
        if self.decision:
            add2 = self.gb(self.SE2(out))
        out = self.layers4(out)
        out = self.layers5(out)
        if self.decision:
            add3 = self.gb(self.SE3(out))
        out = self.layers6(out)
        out = self.layers7(out)
        out = self.layers8(out)
        out = self.layers9(out)
        out = self.layers10(out)
        out = self.layers11(out)
        if self.decision:
            add4 = self.gb(self.SE4(out))
        out = self.layers12(out)
        out = self.layers13(out)
        out = F.avg_pool2d(out, 4)
        if self.decision:
            add = torch.cat([add2, add3,add4], dim=1)
            out=torch.cat([add,out],dim=1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

from opt.flops_benchmark import *
def test():
    net = MobileNet(num_classes=1000)
    x = torch.randn(1,3,32,32)
    y = net(Variable(x))
    print(y.size())


# test()

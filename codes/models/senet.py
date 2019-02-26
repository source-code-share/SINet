'''SENet in PyTorch.

SENet is the winner of ImageNet-2017. The paper is not released yet.
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
            self.conv_r=PreActBlock(self.right_part_in,self.right_part_in,stride=self.stride)
            self.conv_l=PreActBlock(self.left_part_in,self.left_part_in,stride=self.stride)
        else:
            self.conv_r = PreActBlock(self.right_part_in, self.right_part, stride=self.stride)
            self.conv_l = PreActBlock(self.left_part_in, self.left_part, stride=self.stride)

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

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
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

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        out += self.shortcut(x)
        out = F.relu(out)
        return out

class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out

class SENet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SENet, self).__init__()
        self.in_planes = 64
        self.gb=nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.SE1=SELayer(64)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.SE2 = SELayer(128)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.SE3 = SELayer(256)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512+256+128+64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        add1=self.gb(self.SE1(out))
        out = self.layer2(out)
        add2 = self.gb(self.SE2(out))
        out = self.layer3(out)
        add3 = self.gb(self.SE3(out))
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out=torch.cat([add1,add2,add3,out],dim=1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def SENet18(num_classes=10):
    return SENet(BasicUnit, [2,2,2,2],num_classes)


def test():
    net = SENet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()

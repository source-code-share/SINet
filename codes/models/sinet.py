from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
from opt.flops_benchmark import *
import math

def split(x):
    n = int(x.size()[1])
    n1 = round(n * 0.5)
    x1 = x[:, :n1, :, :].contiguous()
    x2 = x[:, n1:, :, :].contiguous()
    return x1, x2


def merge(x1, x2):
    return torch.cat((x1, x2), 1)

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

def Conv_3x3(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def Conv_1x1(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def SepConv_3x3(inp, oup):  # input=32, output=16
    return nn.Sequential(
        # dw
        nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, kernel):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel, stride, kernel // 2, groups=inp * expand_ratio,
                      bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class LinearBottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, t=6,k=3, activation=nn.ReLU6):
        super(LinearBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes * t, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes * t)
        self.conv2 = nn.Conv2d(inplanes * t, inplanes * t, kernel_size=k, stride=stride, padding=1, bias=False,
                               groups=inplanes * t)
        self.bn2 = nn.BatchNorm2d(inplanes * t)
        self.conv3 = nn.Conv2d(inplanes * t, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.activation = activation(inplace=True)
        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride == 1 and self.inplanes == self.outplanes:
            out += residual

        return out

class BasicUnit(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, t=6,k=3,  activation=nn.ReLU6,growth_rate=32, c_tag=0.5):
        super(BasicUnit, self).__init__()
        self.stride = stride
        self.expansion = t
        self.activation=activation
        self.left_part_in = round(c_tag * inplanes)
        self.left_part_out = round(c_tag * outplanes)

        self.right_part_in = inplanes - self.left_part_in
        self.right_part_out = outplanes - self.left_part_out

        self.lout1 = round(c_tag * self.left_part_out)
        self.lout2 = self.left_part_out - self.lout1
        self.rout1 = round(c_tag * self.right_part_out)
        self.rout2 = self.right_part_out - self.rout1

        if stride == 1:
            self.conv_r = InvertedResidual(self.right_part_in, self.rout2, self.stride, self.expansion,k)
            self.conv_l = InvertedResidual(self.left_part_in, self.lout2, self.stride, self.expansion,k)
            self.shortcut_l = nn.Sequential()
            if self.left_part_in != self.lout1:
                self.shortcut_l = nn.Sequential(
                    nn.Conv2d(self.left_part_in, self.lout1, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(self.lout1)
                )
            self.shortcut_r = nn.Sequential()
            if self.right_part_in != self.rout1:
                self.shortcut_r = nn.Sequential(
                    nn.Conv2d(self.right_part_in, self.rout1, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(self.rout1)
                )
        else:
            self.conv_r = InvertedResidual(self.right_part_in, self.right_part_out, self.stride, self.expansion,k)
            self.conv_l = InvertedResidual(self.left_part_in, self.left_part_out, self.stride, self.expansion,k)

    def forward(self, x):
        left, right = split(x)
        out_l = self.conv_l(left)
        out_r = self.conv_r(right)
        out_r = torch.cat([out_r, self.shortcut_l(left)], dim=1) if self.stride == 1 else out_r
        out_l = torch.cat([out_l, self.shortcut_r(right)], dim=1) if self.stride == 1 else out_l
        out = torch.cat([out_r, out_l], dim=1)
        return out

class M2(nn.Module):
    def __init__(self, scale=1.0, input_size=224, t=6, in_channels=3, num_classes=1000, activation=nn.ReLU6):
        super(M2, self).__init__()

        self.scale = scale
        self.t = t
        self.activation_type = activation
        self.activation = activation(inplace=True)
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels, self.c[0], kernel_size=3, bias=False, stride=self.s[0], padding=1)
        self.bn1 = nn.BatchNorm2d(self.c[0])
        self.bottlenecks = self._make_bottlenecks(BasicUnit)

        # Last convolution has 1280 output channels for scale <= 1
        self.last_conv_out_ch = 1280 if self.scale <= 1 else _make_divisible(1280 * self.scale, 8)
        self.conv_last = nn.Conv2d(self.c[-1], self.last_conv_out_ch, kernel_size=1, bias=False)
        self.bn_last = nn.BatchNorm2d(self.last_conv_out_ch)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2, inplace=True)  # confirmed by paper authors
        self.fc = nn.Linear(self.last_conv_out_ch, self.num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_stage(self, block,inplanes, outplanes, n, stride, t, stage):
        modules = OrderedDict()
        stage_name = "LinearBottleneck{}".format(stage)

        # First module is the only one utilizing stride
        first_module = block(inplanes=inplanes, outplanes=outplanes, stride=stride, t=t,
                                        activation=self.activation_type)
        modules[stage_name + "_0"] = first_module

        # add more LinearBottleneck depending on number of repeats
        for i in range(n - 1):
            name = stage_name + "_{}".format(i + 1)
            module = block(inplanes=outplanes, outplanes=outplanes, stride=1, t=6,
                                      activation=self.activation_type)
            modules[name] = module

        return nn.Sequential(modules)

    def _make_bottlenecks(self,block):
        modules = OrderedDict()
        stage_name = "Bottlenecks"

        # First module is the only one with t=1
        bottleneck1 = self._make_stage(LinearBottleneck,inplanes=self.c[0], outplanes=self.c[1], n=self.n[1], stride=self.s[1], t=1,
                                       stage=0)
        modules[stage_name + "_0"] = bottleneck1

        # add more LinearBottleneck depending on number of repeats
        for i in range(1, len(self.c) - 1):
            name = stage_name + "_{}".format(i)
            if i<2:
                block=LinearBottleneck
            else:
                block=BasicUnit
            module = self._make_stage(block,inplanes=self.c[i], outplanes=self.c[i + 1], n=self.n[i + 1],
                                      stride=self.s[i + 1],
                                      t=self.t, stage=i)
            modules[name] = module

        return nn.Sequential(modules)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.bottlenecks(x)
        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.activation(x)

        # average pooling layer
        x = self.avgpool(x)
        x = self.dropout(x)

        # flatten for input to fully-connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)  # TODO not needed(?)



class M1(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(M1, self).__init__()
        self.gb=nn.AdaptiveAvgPool2d(1)
        self.width_mult=width_mult
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s, k
            [3, 24, 4, 2, 3],  # -> 56x56
            [3, 40, 4, 2, 5],  # -> 28x28
            [6, 80, 4, 2, 5],  # -> 14x14
            [6, 96, 4, 1, 3],  # -> 14x14
            [6, 192, 4, 2, 5],  # -> 7x7
            [6, 320, 1, 1, 3],  # -> 7x7
        ]

        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280

        # building first two layer
        self.conv1=Conv_3x3(3, input_channel, 2)
        self.conv2=SepConv_3x3(input_channel, 16)
        input_channel = 16

        self.layers,planes=self._make_layers(input_channel,BasicUnit)

        # building last several layers
        self.conv_last=nn.Conv2d(planes,self.last_channel,1,1,bias=False)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def _make_layers(self,input_channel,block=BasicUnit):
        layers=[]
        for t, c, n, s, k in self.interverted_residual_setting:
            output_channel = int(c * self.width_mult)
            for i in range(n):
                if i == 0:
                    layers.append(block(input_channel, output_channel, s, t, k))
                else:
                    layers.append(block(input_channel, output_channel, 1, t, k))
                input_channel = output_channel
        return nn.Sequential(*layers),input_channel

    def forward(self, x):
        out=self.conv1(x)
        out=self.conv2(out)
        out = self.layers(out)
        out=self.conv_last(out)
        out=self.gb(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def test():
    from torch.autograd import Variable
    # net = M2(scale=1.0)
    net=M1()
    x = Variable(torch.randn(1, 3, 224, 224))
    y = net(x)
    print(y.size())
    gpus = '0,1'
    if gpus is not None:
        gpus = [int(i) for i in gpus.split(',')]
        device = 'cuda:' + str(gpus[0])
    else:
        device = 'cpu'
    dtype = torch.float32

    num_parameters = sum([l.nelement() for l in net.parameters()])
    print('number of parameters: {:.2f}M'.format(num_parameters / 1000000))
    count_flops_test(net, device, dtype, 2)

# test()


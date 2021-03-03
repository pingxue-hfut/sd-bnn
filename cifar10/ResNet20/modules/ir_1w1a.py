import torch.nn as nn
import torch.nn.functional as F
from . import binaryfunction
import torch
from . import se_module

'''
Reference:
[23] Haotong Qin, Ruihao Gong, Xianglong Liu, Mingzhu Shen,Ziran Wei, Fengwei Yu, and Jingkuan Song. Forward and
backward information retention for accurate binary neural networks. 
In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA,
USA, June 13-19, 2020, pages 2247–2256, 2020.
'''

class IRConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(IRConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()
        self.alpha = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        # self.beta = nn.Parameter(torch.zeros(in_channels), requires_grad=True)
        self.dybeta = se_module.SELayer(in_channels)

    def forward(self, input):
        w = self.weight
        a = input
        mw = 0.01 * w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1).detach()
        w = w + torch.sigmoid(self.alpha.view(self.alpha.size(0), 1, 1, 1)) * mw
        # a = a + torch.sigmoid(self.beta.view(1, self.beta.size(0), 1, 1))
        a = a + self.dybeta(a)
        bw = binaryfunction.BinaryQuantize().apply(w, self.k, self.t)
        ba = binaryfunction.BinaryQuantize().apply(a, self.k, self.t)

        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        
        return output

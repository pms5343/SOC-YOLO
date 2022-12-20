from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import torch
from torch.nn.modules.utils import _triple, _pair, _single
from torch.utils.cpp_extension import load

#import softpool_cuda 
class SoftPool2d(torch.nn.Module):
    def __init__(self,kernel_size,stride=None,padding=0,ceil_mode = False,count_include_pad = True,divisor_override = None):
        super(SoftPool2d, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size,stride,padding,ceil_mode,count_include_pad,divisor_override)
    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool
    
class SoftPooling2D(torch.nn.Module):
    def __init__(self,kernel_size,stride=1,padding=5//2,ceil_mode = False,count_include_pad = True,divisor_override = None):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size,stride,padding,ceil_mode,count_include_pad,divisor_override)
        
    def forward(self, x):
        
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool
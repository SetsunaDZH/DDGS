import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, n_vertex = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, n_vertex]).to(x)], dim=1)
        else:
            x = x
        
        return x

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        if enable_padding == True:
            self.__padding = (kernel_size - 1) * dilation
        else:
            self.__padding = 0
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=self.__padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[: , : , : -self.__padding]
        
        return result

class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        padding = (dilation[0] * (kernel_size[0] - 1), dilation[1] * (kernel_size[1] - 1))  # 此处为 causal padding
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        self.padding = (self.padding[0] // 2, self.padding[1] // 2)
        x = F.pad(x, (self.padding[1], 0, self.padding[0], 0))
        return super(CausalConv2d, self).forward(x)

class TemporalConvLayer(nn.Module):
    def __init__(self, Kt, c_in, c_out, n_vertex, act_func):
        super(TemporalConvLayer, self).__init__()
        padding = (Kt - 1, 0)  # Adjust for causal convolution
        print("c_out:", c_out)
        calculated_out_channels = c_out * 2 if act_func in ['glu', 'gtu'] else c_out
        print("calculated_out_channels:", calculated_out_channels)

        self.conv = nn.Conv2d(c_in, c_out * 2 if act_func in ['glu', 'gtu'] else c_out, (Kt, 1), padding=padding)
        self.gate = act_func in ['glu', 'gtu']
        self.act_func = act_func
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x_conv = self.conv(x)
        if self.gate:
            x = x_conv[:, :x_conv.size(1)//2] * self.sigmoid(x_conv[:, x_conv.size(1)//2:])
        else:
            x = self.relu(x_conv)
        return x


        
class ChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out, Ks, gso, bias=True):
        super().__init__()
        self.gso = gso
        self.Ks = Ks
        self.weights = nn.Parameter(torch.Tensor(Ks, c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weights)
        if self.bias is not None:
            fan_in = self.weights.size(1)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x_out = 0
        x_l = x
        for l in range(self.Ks):
            if l > 0:
                x_l = torch.einsum('nchw,wv->nchv', (x_l, self.gso))  # L^l * x
            x_out += torch.einsum('nchw,kio->nkhw', (x_l, self.weights[l]))
        if self.bias is not None:
            x_out += self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return x_out

class GraphConv(nn.Module):
    def __init__(self, c_in, c_out, gso, bias=True):
        super().__init__()
        self.gso = gso
        self.conv = CausalConv2d(c_in, c_out, kernel_size=(1, 1), bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = torch.einsum('nchw,wv->nchv', (x, self.gso))
        return x


class GraphConvLayer(nn.Module):
    def __init__(self, c_in, c_out, Ks, gso, bias=True):
        super(GraphConvLayer, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, (1, Ks), bias=bias)

    def forward(self, x, gso):
        return self.conv(torch.matmul(x, gso))

class STConvBlock(nn.Module):
    def __init__(self, Kt, Ks, n_vertex, c_in, c_out, act_func, graph_conv_type, gso, enable_bias, droprate):
        super(STConvBlock, self).__init__()
        self.tconv1 = TemporalConvLayer(Kt, c_in, c_out, n_vertex, act_func)
        self.gconv = GraphConvLayer(c_out, c_out, Ks, gso, enable_bias)
        self.tconv2 = TemporalConvLayer(Kt, c_out, c_out, n_vertex, act_func)
        self.dropout = nn.Dropout(droprate)

    def forward(self, x):
        x = self.tconv1(x)
        x = self.gconv(x)
        x = self.tconv2(x)
        x = self.dropout(x)
        return x


class OutputBlock(nn.Module):
    def __init__(self, Ko, c_in, c_mid, c_out, n_vertex, act_func='relu', enable_bias=True, droprate=0.0):
        super(OutputBlock, self).__init__()
        self.Ko = Ko
        self.act_func = act_func
        self.enable_bias = enable_bias
        self.dropout = nn.Dropout(droprate)

        # Gated Temporal Convolution Layer
        self.temporal_conv = TemporalConvLayer(Ko, c_in, c_mid, n_vertex, act_func)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(c_mid)

        # Fully-Connected Layers
        self.fc1 = nn.Linear(c_mid * n_vertex, c_out) # Adjusted for flattened input
        self.fc2 = nn.Linear(c_out, c_out)

        # Activation and Dropout
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Temporal convolution
        x = self.temporal_conv(x)
        
        # Normalize and activate
        x = self.layer_norm(x.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1)) # Reshape for LayerNorm
        x = self.relu(x)
        
        # First fully-connected layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Second fully-connected layer
        x = self.fc2(x)
        
        return x


import torch.nn as nn
import torch
import torch.nn.functional as F


##############################################################
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3 , padding=1, stride=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels // 2
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding,
                      stride=1, padding_mode='replicate'),
            # Conv1DPadded(in_channels, mid_channels, kernel_size=kernel_size, padding=padding,
            #           stride=1, padding_mode='replicate'),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding,
                      stride=stride, padding_mode='replicate'),
            # Conv1DPadded(mid_channels, out_channels, kernel_size=kernel_size, padding=padding,
            #           stride=stride, padding_mode='replicate'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x):
        if self.bilinear:
            x = nn.functional.interpolate(input=x.unsqueeze(-1),scale_factor=[2,1],
                                          mode='bilinear', align_corners=True).squeeze(-1)
        else:
            x = self.up(x)
        return self.conv(x)

##############################################################
#############
##############################################################
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class OutDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1, padding=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels // 2
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=padding, stride=1, padding_mode='replicate'),
            # Conv1DPadded(in_channels, mid_channels, kernel_size=3, padding=padding, stride=1, padding_mode='replicate'),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.double_conv(x)



##############################################################
#############
##############################################################
class Conv1dTBlock(nn.Module):
    '''
    simplified version if D too hard
    '''
    def __init__(self, input_dim, hidden_dims=[128],  ks=2, activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'leak':
            self.activation = nn.LeakyReLU()

        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Conv1d(last_dim, nh, kernel_size=ks))
            ks = 1  # only for first layer.
            last_dim = nh

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x

class Conv1dBlock(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128],  ks=2, activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'leak':
            self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Conv1d(last_dim, nh, kernel_size=ks))
            self.affine_layers.append(nn.Conv1d(nh, nh, kernel_size=1))
            last_dim = nh

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x

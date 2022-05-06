import torch
import numpy as np
import torch.nn as nn
from functions import PLU


class StateEncoder(nn.Module):
    def __init__(self, in_dim = 128, hidden_dim = 512, out_dim = 256):
        super(StateEncoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        self.fc0 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc1 = nn.Linear(hidden_dim, out_dim, bias=True)
    
    def forward(self, x):
        x = self.fc0(x)
        x = PLU(x)
        x = self.fc1(x)
        x = PLU(x)
        return x

class OffsetEncoder(nn.Module):
    def __init__(self, in_dim = 128, hidden_dim = 512, out_dim = 256):
        super(OffsetEncoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        self.fc0 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc1 = nn.Linear(hidden_dim, out_dim, bias=True)
    
    def forward(self, x):
        x = self.fc0(x)
        x = PLU(x)
        x = self.fc1(x)
        x = PLU(x)
        return x

class TargetEncoder(nn.Module):
    def __init__(self, in_dim = 128, hidden_dim = 512, out_dim = 256):
        super(TargetEncoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        self.fc0 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc1 = nn.Linear(hidden_dim, out_dim, bias=True)
    
    def forward(self, x):
        x = self.fc0(x)
        x = PLU(x)
        x = self.fc1(x)
        x = PLU(x)
        return x

class LSTM(nn.Module):
    def __init__(self, in_dim = 128, hidden_dim = 768, num_layer = 1):
        super(LSTM, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.rnn = nn.LSTM(self.in_dim, self.hidden_dim, self.num_layer)
    
    def init_hidden(self, batch_size):
        self.h = torch.zeros((self.num_layer, batch_size, self.hidden_dim)).cuda()
        self.c = torch.zeros((self.num_layer, batch_size, self.hidden_dim)).cuda()
    
    def forward(self, x):
        x, (self.h, self.c) = self.rnn(x, (self.h, self.c))
        return x
        

class Decoder(nn.Module):
    def __init__(self, in_dim = 128, hidden_dim = 512, out_dim = 256):
        super(Decoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        self.fc0 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2, bias=True)
        self.fc2 = nn.Linear(hidden_dim // 2, out_dim - 4, bias=True)
        self.fc_conct = nn.Linear(hidden_dim // 2, 4, bias=True)
        self.ac_sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc0(x)
        x = PLU(x)
        x = self.fc1(x)
        x = PLU(x)
        o1 = self.fc2(x)
        o2 = self.ac_sig(self.fc_conct(x))
        return o1, o2

class ShortMotionDiscriminator(nn.Module):
    def __init__(self, length = 3, in_dim = 128, hidden_dim = 512, out_dim = 1):
        super(ShortMotionDiscriminator, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.length = length

        self.fc0 = nn.Conv1d(in_dim, hidden_dim, kernel_size = self.length, bias=True)
        self.fc1 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size = 1, bias=True)
        self.fc2 = nn.Conv1d(hidden_dim // 2, out_dim, kernel_size = 1, bias=True)

    def forward(self, x):
        x = self.fc0(x)
        x = PLU(x)
        x = self.fc1(x)
        x = PLU(x)
        x = self.fc2(x)
        return x

class LongMotionDiscriminator(nn.Module):
    def __init__(self, length = 10, in_dim = 128, hidden_dim = 512, out_dim = 1):
        super(LongMotionDiscriminator, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.length = length
        
        self.fc0 = nn.Conv1d(in_dim, hidden_dim, kernel_size = self.length, bias=True)
        self.fc1 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size = 1, bias=True)
        self.fc2 = nn.Conv1d(hidden_dim // 2, out_dim, kernel_size = 1, bias=True)

    def forward(self, x):
        x = self.fc0(x)
        x = PLU(x)
        x = self.fc1(x)
        x = PLU(x)
        x = self.fc2(x)
        return x

if __name__=="__main__":
    state_encoder = StateEncoder()
    x = torch.zeros((32, 128))
    print(state_encoder(x).size())

    offset_encoder = OffsetEncoder()
    x = torch.zeros((32, 128))
    print(offset_encoder(x).size())

    target_encoder = TargetEncoder()
    x = torch.zeros((32, 128))
    print(target_encoder(x).size())

    lstm = LSTM(32)
    x = torch.zeros((10, 32, 128))
    print(lstm(x).size())

    decoder = Decoder()
    x = torch.zeros((32, 128))
    print(decoder(x)[0].size())

    short_dis = ShortMotionDiscriminator()
    x = torch.zeros((32, 128, 50))
    print(short_dis(x).size())

    long_dis = LongMotionDiscriminator()
    x = torch.zeros((32, 128, 50))
    print(long_dis(x).size())


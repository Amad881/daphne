import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


class rnn_block(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.rnn = nn.GRU(hidden_size,hidden_size,bidirectional=True,batch_first=True) # Change Bidrectional to False for unidirectional
        self.linear = nn.Linear(hidden_size*3,hidden_size) # Change 3 to 2 for unidirectional

    def forward(self,x):
        ln1x = self.ln1(x)
        rnnx = self.rnn(ln1x)[0]
        ln2x = self.ln2(x)
        out = torch.cat([rnnx,ln2x],dim=-1)
        out = self.linear(out)
        return out

    
class attention_block(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.q_norm = nn.LayerNorm(hidden_size)
        self.kv_norm = nn.LayerNorm(hidden_size)

        self.register_parameter("q_prime", nn.Parameter(torch.rand((1,hidden_size),requires_grad=True)-.5))
        self.register_parameter("k_prime", nn.Parameter(torch.rand((1,hidden_size),requires_grad=True)-.5))
        self.register_parameter("v_prime", nn.Parameter(torch.rand((1,hidden_size),requires_grad=True)-.5))
        self.q_lin = nn.Linear(hidden_size,hidden_size)
        self.v1_lin = nn.Linear(hidden_size,hidden_size)
        self.v2_lin = nn.Linear(hidden_size,hidden_size)
        self.divisor = np.sqrt(hidden_size)
        
    def forward(self,x):
        xqn = self.q_norm(x)
        xkvn = self.kv_norm(x)
        kr = xkvn * torch.sigmoid(self.k_prime)
        qr = self.q_lin(xqn) * torch.sigmoid(self.q_prime)
        tmp_v_prime = torch.sigmoid(self.v_prime)
        vr = xkvn * (torch.sigmoid(self.v1_lin(tmp_v_prime)) * torch.tanh(self.v2_lin(tmp_v_prime)))
        tmp_shape = (*qr.shape[:-1],kr.shape[-2])
        attended = torch.matmul(F.softmax(torch.matmul(qr,kr.transpose(-1,-2)).flatten(-2)/self.divisor,dim=-1).reshape(tmp_shape),vr)
        return attended + xqn

    
class feedforward_block(nn.Module):
    def __init__(self,hidden_size,dropout=.5,mult=4):
        super().__init__()
        self.path = nn.Sequential(nn.LayerNorm(hidden_size),
                                  nn.Linear(hidden_size,hidden_size*mult),
                                  nn.GELU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(hidden_size*mult,hidden_size))
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self,x):
        out = self.path(x) + self.ln(x)
        return out


class arn_block(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.rnn_block = rnn_block(hidden_size)
        self.attn_block = attention_block(hidden_size)
        self.ff_block = feedforward_block(hidden_size)

    def forward(self,x):
        out = self.rnn_block(x)
        out = self.attn_block(out)
        out = self.ff_block(out)
        return out

class Classifier(nn.Module):
    def __init__(self, n_classes, hidden_size=768):
        super(Classifier, self).__init__()
        self.n_classes = n_classes

        self.ar_block = arn_block(hidden_size)
        self.linear = nn.Linear(hidden_size,self.n_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        out = self.ar_block(x)
        out = out.mean(-2)
        out = self.linear(out)
        y_pred = self.softmax(out)
        return y_pred
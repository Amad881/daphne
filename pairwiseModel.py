import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from pdb import set_trace as bp

''' 
Original implementation of self-attentive RNN by Adam Stiff taken from (https://arxiv.org/pdf/1703.03130.pdf)
'''

# Bidirectional GRU block
class RNN(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.rnn = nn.GRU(hidden_size,hidden_size,bidirectional=True,batch_first=True)

    def forward(self,x):
        return self.rnn(x)[0]


# Dot product attention block on output of RNN block
class Attention(nn.Module):
    def __init__(self, inp_size, attention_nhid=350, attention_heads=8):
        super().__init__()
        self.ws1 = nn.Linear(inp_size, attention_nhid, bias=False)
        self.ws2 = nn.Linear(attention_nhid, attention_heads, bias=False)
        self.dropout = nn.Dropout(0.5)

    def forward(self, H):
        # Create attention weights
        A = F.softmax(self.ws2(torch.tanh(self.ws1(self.dropout(H)))),dim=1)
        
        # Apply attention weights to input
        M = torch.matmul(A.transpose(-1,-2),H)

        # Mean over attention heads
        M = torch.mean(M,dim=1)
        return M


class Classifier(nn.Module):
    def __init__(self, n_classes, hidden_size=768):
        super().__init__()
        self.rnn = RNN(hidden_size)
        self.attention = Attention(hidden_size*2)
        self.ff1 = nn.Linear(hidden_size*2, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.ff2 = nn.Linear(hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        out = self.rnn(x)
        out = self.attention(out)
        out = self.ff1(out)
        out = self.ln(out)
        out = self.ff2(out)
        out = self.softmax(out)
        return out


if __name__ == '__main__':
    batchSize = 10
    seqLen = 20
    embeddingSize = 30
    testTensor = torch.empty((batchSize, seqLen, embeddingSize))
    classifier = Classifier(embeddingSize, 20)
    pred = classifier(testTensor)
    bp()
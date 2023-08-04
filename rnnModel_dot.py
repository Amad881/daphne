import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from pdb import set_trace as bp
from transformers import AutoModel

''' 
Original implementation of self-attentive RNN by Adam Stiff taken from (https://arxiv.org/pdf/1703.03130.pdf)
'''

class Embed(nn.Module):
    def __init__(self):
        super().__init__()
        bert = AutoModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.embd = bert.embeddings
        self.multiLayer = False
        if self.multiLayer:
            self.bert = bert
            self.layers = [-1, -2, -3, -4]
    
    def forward(self, x):
        if self.multiLayer:
            output = self.bert(**x)
            states = output.hidden_states
            output = torch.stack([states[i] for i in self.layers]).sum(0).squeeze()
            if len(output.shape) == 2:
                output = torch.Tensor([output])
        else:
            output = self.embd(x['input_ids'])
        return output

# Bidirectional GRU block
class RNN(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.rnn = nn.GRU(hidden_size,hidden_size,bidirectional=True,batch_first=True)

    def forward(self,x):
        return self.rnn(x)[0]


# Dot product attention block on output of RNN block
class Attention(nn.Module):
    def __init__(self, inp_size, attention_nhid=350, attention_heads=8, dropout=0.3):
        super().__init__()
        self.ws1 = nn.Linear(inp_size, attention_nhid, bias=False)
        self.ws2 = nn.Linear(attention_nhid, attention_heads, bias=False)
        self.downsize = nn.Linear(inp_size*attention_heads, inp_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H, masks):
        # Create attention weights
        A = self.ws2(self.dropout(torch.tanh(self.ws1(self.dropout(H)))))

        # Applied masked softmax
        masks = torch.ones_like(masks,device=masks.device) - masks
        # Expands masks to match shape of A and then applies
        masks = masks.unsqueeze(2)
        A = F.softmax(A + masks*(-1e9), dim=1)

        # Apply attention weights to input
        M = torch.matmul(A.transpose(-1,-2),H)

        # Mean over attention heads (this or downsize)
        # M = torch.mean(M,dim=1)

        # Downsize to over all attention heads
        M = M.view(M.shape[0],-1)
        M = self.downsize(self.dropout(M))
        return M


class Classifier(nn.Module):
    def __init__(self, n_classes, hidden_size=768, dropout=0.3):
        super().__init__()
        self.embed = Embed()
        self.rnn = RNN(hidden_size)
        self.attention = Attention(hidden_size*2, dropout=dropout)
        self.ff1 = nn.Linear(hidden_size*2, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.ff2 = nn.Linear(hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x):
        out = self.embed(x)
        out = self.rnn(self.dropout(out))
        out = self.attention(out, x['attention_mask'])
        return out

    def classify(self, x):
        out = self.ff1(self.dropout(x))
        out = self.ln(self.dropout(out))
        preSoft = self.ff2(self.dropout(out))
        postSoft = self.softmax(preSoft)
        return postSoft, preSoft

    def forward(self, x):
        out = self.encode(x)
        out, _ = self.classify(out)
        return out


if __name__ == '__main__':
    model = Classifier(2)
    x = torch.rand(1, 512)
    y = torch.rand(1, 512)
    out = model(x, y)
    print(out.shape)
'''
implement epoch based sampling
implement the staggered loss
implement dataset cartography
'''
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

meta_dim = {'latlong':4, 'month':2, 'time':2}

class source_encoding(nn.Module):
    def __init__(self,sources=4,fc_dim=512):
        super().__init__()
        self.source_embed = nn.Embedding(sources,fc_dim)

    def forward(self,x):
        x = self.source_embed(x)
        return x

class metaNet(nn.Module):
    def __init__(self,metadata_type='latlong',fc_dim = 512):
        super().__init__()
        self.fc1 = nn.Linear(meta_dim[metadata_type], 64)
        self.fc2 = nn.Linear(64,fc_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.fc1(x))
        output = self.fc2(x)
        return output


#original implementation" https://github.com/naver-ai/pcmepp/blob/main/pcmepp/modules/mlp.py
class pcmeppMLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=1024, num_layers=1):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])
        self.last_layer = self.layers.pop(-1)
        self.bns.pop(-1)

    def forward(self, x):
        B, N, D = x.size()
        x = x.reshape(B*N, D)
        for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            x = F.relu(bn(layer(x))) #if i < self.num_layers - 1 else layer(x)
        x = self.last_layer(x)
        x = x.view(B, N, self.output_dim)
        return x
    

if __name__ == '__main__':
    metadata_type = 'latlong'
    metanet = metaNet(metadata_type=metadata_type,fc_dim=1024)
    x = torch.randn(5,meta_dim[metadata_type])
    print(metanet(x).shape) #torch.Size([5, 1024])
    
    sources = torch.tensor([0,1,3,2,3])
    sourceNet = source_encoding()
    print(sourceNet(sources).shape) #torch.Size([5, 1024])

    pcmeppMLP = pcmeppMLP(input_dim=1024, hidden_dim=512, output_dim=1024, num_layers=1)
    x = torch.randn(5,196,1024)
    print(pcmeppMLP(x).shape) #torch.Size([5, 196, 1024])
    





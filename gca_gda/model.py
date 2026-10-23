"""Two-layer GAT feature encoder and cosine classifier."""
import torch
from torch import nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv as PyGGAT
except ImportError:
    PyGGAT=None

class ScatterGAT(nn.Module):
    """Single-head graph attention with destination-wise edge softmax.
    Uses sparse edge indexing instead of a dense N x N adjacency matrix.
    """
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.linear=nn.Linear(in_channels,out_channels,bias=False)
        self.a_src=nn.Parameter(torch.empty(out_channels))
        self.a_dst=nn.Parameter(torch.empty(out_channels))
        self.bias=nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.a_src.view(1,-1))
        nn.init.xavier_uniform_(self.a_dst.view(1,-1))

    def forward(self,x,edge):
        n=x.size(0); device=x.device
        loops=torch.arange(n,device=device)
        edge=edge[:,edge[0]!=edge[1]]
        src=torch.cat([edge[0],loops]); dst=torch.cat([edge[1],loops])
        z=self.linear(x)
        scores=F.leaky_relu((z[src]*self.a_src).sum(-1)+
                             (z[dst]*self.a_dst).sum(-1),negative_slope=.2)
        max_per=z.new_full((n,),float('-inf'))
        max_per.scatter_reduce_(0,dst,scores,reduce='amax',include_self=True)
        weights=(scores-max_per[dst]).exp()
        denom=z.new_zeros(n).index_add_(0,dst,weights).clamp_min(1e-12)
        alpha=weights/denom[dst]
        out=z.new_zeros(z.shape).index_add_(0,dst,z[src]*alpha.unsqueeze(-1))
        return out+self.bias

class Encoder(nn.Module):
    def __init__(self,in_channels,hidden_dim=128,dropout=.5,require_pyg=False):
        super().__init__()
        if require_pyg and PyGGAT is None:
            raise ImportError('torch-geometric not installed; install or omit --require-pyg')
        layer = (lambda a,b:PyGGAT(a,b,heads=1,concat=False,dropout=0.)) if PyGGAT else ScatterGAT
        self.layers=nn.ModuleList([layer(in_channels,2*hidden_dim),
                                   layer(2*hidden_dim,hidden_dim)])
        self.dropout=dropout
    def forward(self,x,edge):
        for layer in self.layers:
            x=F.relu(layer(x,edge))
            x=F.dropout(x,p=self.dropout,training=self.training)
        return F.normalize(x,p=2,dim=-1,eps=1e-8)

class CosineModel(nn.Module):
    def __init__(self,in_channels,n_classes,hidden_dim=128,dropout=.5,scale=16.,require_pyg=False):
        super().__init__()
        self.encoder=Encoder(in_channels,hidden_dim,dropout,require_pyg=require_pyg)
        self.classifier=nn.Parameter(torch.empty(n_classes,hidden_dim))
        nn.init.xavier_uniform_(self.classifier)
        self.scale=float(scale)
    def logits(self,h):
        return self.scale*F.normalize(h,dim=-1) @ F.normalize(self.classifier,dim=-1).T
    def forward(self,x,edge):
        h=self.encoder(x,edge)
        return self.logits(h),h


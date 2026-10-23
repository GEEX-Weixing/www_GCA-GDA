"""Load the five supported graph domain-adaptation benchmarks.

All available labeled source nodes are supervised; unlabeled target nodes
are used for adaptation. Target labels are accessed only at final evaluation.
Dataset files are not included in the anonymous code release.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import torch

DOMAINS = {
    'citation': ['acmv9', 'citationv1', 'dblpv7'],
    'blog': ['Blog1', 'Blog2'],
    'twitch': ['DE', 'ENGB', 'FR'],
    'mag': ['FR', 'JP', 'RU'],
    'elliptic': ['S10', 'M10', 'E10'],
}
ALIASES = {'ACM': 'acmv9', 'C': 'citationv1', 'DBLP': 'dblpv7',
           'EN': 'ENGB', 'B1': 'Blog1', 'B2': 'Blog2',
           'D': 'dblpv7', 'A': 'acmv9'}

@dataclass
class Graph:
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    eval_mask: torch.Tensor

    def to(self, device):
        return Graph(*(getattr(self,k).to(device) for k in self.__dataclass_fields__))


def _edge_index(adj):
    if sp.issparse(adj):
        adj = adj.tocoo()
        return torch.from_numpy(np.stack((adj.row, adj.col)).astype('int64'))
    return torch.from_numpy(np.stack(np.nonzero(adj)).astype('int64'))


def _labels(y):
    if sp.issparse(y): y=y.toarray()
    y = np.asarray(y)
    if y.ndim > 1 and y.shape[1] > 1:
        y = y.argmax(1)
    return np.asarray(y).reshape(-1).astype('int64')


def _mask(indices, n):
    mask = torch.zeros(n, dtype=torch.bool)
    mask[torch.as_tensor(indices, dtype=torch.long)] = True
    return mask


def _source_split(y, seed=None, family=None, label_rate=None):
    """Full-source supervision: all nodes with available source labels.

    No source validation split is created, and no target evaluation labels are
    used to select checkpoints. Unlabeled source nodes remain graph context.
    The legacy arguments are accepted for loader compatibility only.
    """
    known = np.flatnonzero(np.asarray(y) >= 0)
    if len(known) == 0:
        raise ValueError('Source graph has no labeled nodes')
    train = _mask(known, len(y))
    val = torch.zeros(len(y), dtype=torch.bool)
    return train, val


def _citation_or_blog(root, name, family):
    path = root / (name + '.mat')
    if not path.exists(): raise FileNotFoundError(f'Missing {path}')
    net = sio.loadmat(path)
    x,a,y = net['attrb'],net['network'],net['group']
    # Preserve the benchmark-specific uint8/CSC feature conversion for Blog.
    if family == 'blog':
        x=sp.csc_matrix(x.astype(np.uint8)).astype(np.float32)
    if sp.issparse(x): x=x.toarray()
    if sp.issparse(y): y=y.toarray()
    # As in the original loader, attrb is NOT row-normalized.
    return np.asarray(x,dtype='float32'),_edge_index(a),_labels(y)


def _twitch(root,name):
    import networkx as nx
    edges = pd.read_csv(root/'edge'/f'{name}_edges.csv').iloc[:,:2].to_numpy()
    graph = nx.from_edgelist([(int(a),int(b)) for a,b in edges])
    graph.remove_edges_from(nx.selfloop_edges(graph))
    nodes = list(graph.nodes()); idx = {node:i for i,node in enumerate(nodes)}
    x = np.zeros((len(nodes),3170),dtype='float32')
    with open(root/'features'/f'{name}.json', encoding='utf-8') as f: features=json.load(f)
    for key, cols in features.items():
        node=int(key)
        if node in idx: x[idx[node],np.asarray(cols,dtype=int)] = 1.
    labels = pd.read_csv(root/'target'/f'{name}_target.csv').drop_duplicates('new_id')
    y = np.full(len(nodes),-1,dtype=np.int64)
    for _, row in labels.iterrows():
        node=int(row['new_id'])
        if node in idx:
            status=str(row['mature']).strip().lower()
            if status not in ('true','false','1','0'):
                raise ValueError(f'Unexpected Twitch mature label: {row["mature"]!r}')
            y[idx[node]] = int(status in ('true','1'))
    adj=nx.to_scipy_sparse_array(graph,nodelist=nodes,format='coo')
    return x,_edge_index(adj),y


def _mag(root,name):
    path = root/f'{name}_labels_20.pt'
    if not path.exists(): raise FileNotFoundError(f'Missing {path}')
    # Graphs serialized as PyG Data require torch-geometric installed at load time.
    try: graph=torch.load(path,map_location='cpu',weights_only=False)
    except (ModuleNotFoundError,AttributeError) as exc:
        raise ImportError('MAG .pt contains PyG objects: install compatible torch-geometric to unpickle') from exc
    x = graph.x.cpu().numpy().astype('float32')
    edge_index = graph.edge_index.cpu().long()
    # Symmetrize observed MAG edges without building dense adjacency.
    edge_index=torch.cat([edge_index,edge_index.flip(0)],dim=1).unique(dim=1)
    return x,edge_index,graph.y.cpu().numpy().reshape(-1).astype('int64')


def _elliptic(root,name):
    candidates = [root/'Elliptic'/name/'raw'/f'{name}.pkl',
        root/'Elliptic'/name/f'{name}.pkl',root/'Elliptic'/f'{name}.pkl']
    path=next((p for p in candidates if p.exists()),None)
    if path is None: raise FileNotFoundError(f'{name} not found; tried {candidates}')
    with path.open('rb') as f: adj,y,x=pickle.load(f)
    # Preserve raw Elliptic labels; do not treat unlabeled nodes as a class.
    #   -1 = unlabeled: still included in x/edge_index and self-supervised loss
    #    0, 1 = the two classes used for supervised training and evaluation.
    y=np.asarray(y).reshape(-1).astype('int64')
    unique=np.unique(y)
    if not np.isin(unique,[-1,0,1]).all():
        raise ValueError(f'{path}: expected raw Elliptic labels in [-1,0,1], got {unique}')
    if not np.any(y>=0):
        raise ValueError(f'{path}: no labeled Elliptic nodes (expected labels 0/1)')
    return np.asarray(x,dtype='float32'),_edge_index(adj),y


def load_graph(family,name,data_dir='data',seed=42,role='source',label_rate=1.0):
    if family not in DOMAINS or name not in DOMAINS[family]:
        raise ValueError(f'Unknown family/domain: {family}/{name}, valid={DOMAINS}')
    root=Path(data_dir)
    x,edge,y=({'citation':_citation_or_blog,'blog':_citation_or_blog}.get(family) or
        {'twitch':_twitch,'mag':_mag,'elliptic':_elliptic}[family])(root,name,*([family] if family in ('citation','blog') else []))
    if x.shape[0] != len(y): raise ValueError(f'{name}: features/labels count mismatch')
    if x.ndim!=2: raise ValueError(f'{name}: x must be a matrix')
    if int(edge.max())>=len(y) or int(edge.min())<0: raise ValueError(f'{name}: invalid edge indices')
    if role=='source':
        train,val=_source_split(y,seed,family,label_rate)
        ev=torch.zeros(len(y),dtype=torch.bool)
    else:
        train=val=torch.zeros(len(y),dtype=torch.bool)
        # Target labels are used at final evaluation only, never for selection.
        ev=torch.from_numpy(y>=0)
    return Graph(torch.from_numpy(x.copy()).float(),edge.long(),torch.from_numpy(y).long(),train,val,ev)

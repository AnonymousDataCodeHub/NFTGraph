from typing import Iterator
import torch
from torch.nn import Module, ModuleList, Linear, Parameter, BatchNorm1d, LayerNorm
import torch.nn.functional as F
import torch.nn as nn

from dgl.nn.pytorch import GraphConv
from dgl import DropEdge
from layers import SAGEConv, GATConv, AGDNConv, MemAGDNConv
from dgl.sampling import node2vec_random_walk
from torch_geometric.nn import Node2Vec
from torch.utils.data import DataLoader


class MLP(Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers,
                 dropout, bn=False):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        if bn and n_layers > 1:
            self.bns = torch.nn.ModuleList()
        else:
            self.bns = None
        for i in range(n_layers):
            in_feats_ = in_feats if i == 0 else n_hidden
            out_feats_ = out_feats if i == n_layers - 1 else n_hidden
            self.lins.append(torch.nn.Linear(in_feats_, out_feats_))
            if bn and i < n_layers - 1:
                self.bns.append(torch.nn.BatchNorm1d(out_feats_))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, graph,feat,edge_feat):
        x = feat
        # x = torch.cat([x_i, x_j], dim=-1)
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                if self.bns is not None:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class Node2vec(nn.Module):
    """Node2vec model from paper node2vec: Scalable Feature Learning for Networks <https://arxiv.org/abs/1607.00653>
    Attributes
    ----------
    g: DGLGraph
        The graph.
    embedding_dim: int
        Dimension of node embedding.
    walk_length: int
        Length of each trace.
    p: float
        Likelihood of immediately revisiting a node in the walk.  Same notation as in the paper.
    q: float
        Control parameter to interpolate between breadth-first strategy and depth-first strategy.
        Same notation as in the paper.
    num_walks: int
        Number of random walks for each node. Default: 10.
    window_size: int
        Maximum distance between the center node and predicted node. Default: 5.
    num_negatives: int
        The number of negative samples for each positive sample.  Default: 5.
    use_sparse: bool
        If set to True, use PyTorch's sparse embedding and optimizer. Default: ``True``.
    weight_name : str, optional
        The name of the edge feature tensor on the graph storing the (unnormalized)
        probabilities associated with each edge for choosing the next node.

        The feature tensor must be non-negative and the sum of the probabilities
        must be positive for the outbound edges of all nodes (although they don't have
        to sum up to one).  The result will be undefined otherwise.

        If omitted, DGL assumes that the neighbors are picked uniformly.
    """

    def __init__(
        self,
        g,
        embedding_dim,
        walk_length,
        p,
        q,
        num_walks=10,
        window_size=5,
        num_negatives=1,
        use_sparse=True,
        weight_name=None,
    ):
        super(Node2vec, self).__init__()

        assert walk_length >= window_size

        self.g = g
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.window_size = window_size
        self.num_negatives = num_negatives
        self.N = self.g.num_nodes()
        if weight_name is not None:
            self.prob = weight_name
        else:
            self.prob = None
        self.use_sparse = use_sparse
        self.embedding = nn.Embedding(self.N, embedding_dim, sparse=use_sparse)

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def sample(self, batch):
        """
        Generate positive and negative samples.
        Positive samples are generated from random walk
        Negative samples are generated from random sampling
        """
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch = batch.repeat(self.num_walks)
        # positive
        pos_traces = node2vec_random_walk(
            self.g, batch, self.p, self.q, self.walk_length, self.prob
        )
        pos_traces = pos_traces.unfold(1, self.window_size, 1)  # rolling window
        pos_traces = pos_traces.contiguous().view(-1, self.window_size)

        # negative
        neg_batch = batch.repeat(self.num_negatives)
        neg_traces = torch.randint(
            self.N, (neg_batch.size(0), self.walk_length)
        )
        neg_traces = torch.cat([neg_batch.view(-1, 1), neg_traces], dim=-1)
        neg_traces = neg_traces.unfold(1, self.window_size, 1)  # rolling window
        neg_traces = neg_traces.contiguous().view(-1, self.window_size)

        return pos_traces, neg_traces

    def loss(self, pos_trace, neg_trace):
        """
        Computes the loss given positive and negative random walks.
        Parameters
        ----------
        pos_trace: Tensor
            positive random walk trace
        neg_trace: Tensor
            negative random walk trace

        """
        e = 1e-15

        # Positive
        pos_start, pos_rest = (
            pos_trace[:, 0],
            pos_trace[:, 1:].contiguous(),
        )  # start node and following trace
        w_start = self.embedding(pos_start).unsqueeze(dim=1)
        w_rest = self.embedding(pos_rest)
        pos_out = (w_start * w_rest).sum(dim=-1).view(-1)

        # Negative
        neg_start, neg_rest = neg_trace[:, 0], neg_trace[:, 1:].contiguous()

        w_start = self.embedding(neg_start).unsqueeze(dim=1)
        w_rest = self.embedding(neg_rest)
        neg_out = (w_start * w_rest).sum(dim=-1).view(-1)

        # compute loss
        pos_loss = -torch.log(torch.sigmoid(pos_out) + e).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(neg_out) + e).mean()

        return pos_loss + neg_loss

    def loader(self, batch_size):
        """

        Parameters
        ----------
        batch_size: int
            batch size

        Returns
        -------
        DataLoader
            Node2vec training data loader

        """
        return DataLoader(
            torch.arange(self.N),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.sample,
        )

    def forward(self, graph,feat,nodes=None):
        """
        Returns the embeddings of the input nodes
        Parameters
        ----------
        nodes: Tensor, optional
            Input nodes, if set `None`, will return all the node embedding.

        Returns
        -------
        Tensor
            Node embedding

        """
        emb = self.embedding.weight
        if nodes is None:
            return emb
        else:
            return emb[nodes]


class MF(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        use_sparse=True,
    ):
        super(MF, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, sparse=use_sparse)

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def forward(self, graph,feat,edge_feat):
        """
        Returns the embeddings of the input nodes
        Parameters
        ----------
        nodes: Tensor, optional
            Input nodes, if set `None`, will return all the node embedding.

        Returns
        -------
        Tensor
            Node embedding

        """
        emb = self.embedding.weight
        return emb
        

class GCN(Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers,
                 dropout, input_drop, bn=True, residual=True):
        super(GCN, self).__init__()
        self.residual = residual
        self.input_drop = input_drop
        self.convs = ModuleList()
        if n_layers == 1:
            self.convs.append(GraphConv(in_feats, out_feats, norm='both', allow_zero_in_degree=True))
            self.norms = None
        else:
            self.convs.append(GraphConv(in_feats, n_hidden, norm="both", allow_zero_in_degree=True))
            if bn:
                self.norms = ModuleList()
                self.norms.append(BatchNorm1d(n_hidden))
            else:
                self.norms = None

        for _ in range(n_layers - 2):
            self.convs.append(
                GraphConv(n_hidden, n_hidden, norm="both", allow_zero_in_degree=True))
            if bn:
                self.norms.append(BatchNorm1d(n_hidden))

        if n_layers > 1:
            self.convs.append(GraphConv(n_hidden, out_feats, norm="both", allow_zero_in_degree=True))


        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        
        if self.norms is not None:
            for norm in self.norms:
                norm.reset_parameters()

    def forward(self, graph, feat, edge_feat=None):
        h = F.dropout(feat, self.input_drop, training=self.training)
        h_last = h
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(graph, h, edge_weight=edge_feat)
            if self.residual:
                if h_last.shape[1] == h.shape[1]:
                    h = h + h_last
            if self.norms is not None:
                h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h_last = h
        h = self.convs[-1](graph, h, edge_weight=edge_feat)
        if self.residual:
            if h_last.shape[1] == h.shape[1]:
                h = h + h_last
        if len(self.convs) == 1:
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class SAGE(torch.nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers,
                 dropout, input_drop, edge_feats=0, bn=True, residual=True):
        super(SAGE, self).__init__()
        self.residual = residual
        self.input_drop = input_drop
        self.convs = torch.nn.ModuleList()
        if n_layers == 1:
            self.convs.append(SAGEConv(in_feats, out_feats, 'mean'))
            self.norms = None
        else:
            self.convs.append(SAGEConv(in_feats, n_hidden, 'mean'))
            if bn:
                self.norms = torch.nn.ModuleList()
                self.norms.append(BatchNorm1d(n_hidden))
            else:
                self.norms = None
        for _ in range(n_layers - 2):
            self.convs.append(SAGEConv(n_hidden, n_hidden, 'mean'))
            if bn:
                self.norms.append(BatchNorm1d(n_hidden))
        if n_layers > 1:
            self.convs.append(SAGEConv(n_hidden, out_feats, 'mean'))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        if self.norms is not None:
            for norm in self.norms:
                norm.reset_parameters()

    def forward(self, graph, feat, edge_feat=None):
        h = F.dropout(feat, self.input_drop, training=self.training)
        h_last = h
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(graph, h, edge_weight=edge_feat)
            if self.residual:
                if h_last.shape[1] == h.shape[1]:
                    h = h + h_last
            if self.norms is not None:
                h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h_last = h
        h = self.convs[-1](graph, h, edge_weight=edge_feat)
        if self.residual:
            if h_last.shape[1] == h.shape[1]:
                h = h + h_last
        if len(self.convs) == 1:
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class GAT(Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers,
                 num_heads,
                 dropout, input_drop, attn_drop, bn=True, residual=False):
        super(GAT, self).__init__()
        self.residual = residual
        self.input_drop = input_drop
        self.convs = ModuleList()
        
        self.convs.append(GATConv(in_feats, n_hidden if n_layers > 1 else out_feats, num_heads, attn_drop=attn_drop, residual=True))
        if bn and n_layers > 1:
            self.norms = ModuleList()
            self.norms.append(BatchNorm1d(num_heads * n_hidden))
        else:
            self.norms = None


        for _ in range(n_layers - 2):
            self.convs.append(
                GATConv(n_hidden * num_heads, n_hidden, num_heads, attn_drop=attn_drop, residual=True))
            if bn:
                self.norms.append(
                    BatchNorm1d(num_heads * n_hidden)
                )

        if n_layers > 1:
            self.convs.append(GATConv(n_hidden * num_heads, out_feats, num_heads, attn_drop=attn_drop, residual=True))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        if self.norms is not None:
            for norm in self.norms:
                norm.reset_parameters()

    def forward(self, graph, feat, edge_feat=None):
        h = F.dropout(feat, self.input_drop, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(graph, h, edge_feat=edge_feat).flatten(1)
            if self.norms is not None:
                h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.convs[-1](graph, h, edge_feat=edge_feat).mean(1)
        if len(self.convs) == 1:
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class AGDN(Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers,
                 num_heads, K,
                 dropout, input_drop, attn_drop, edge_drop, diffusion_drop,
                 transition_matrix='gat',
                 no_dst_attn=False,
                 weight_style="HA", bn=True, output_bn=False, hop_norm=False,
                 pos_emb=True, residual=False, share_weights=True, pre_act=False):
        super(AGDN, self).__init__()
        self.residual = residual
        self.input_drop = input_drop

        self.convs = ModuleList()
        self.convs.append(AGDNConv(in_feats, n_hidden if n_layers > 1 else out_feats, num_heads, K, 
            attn_drop=attn_drop, edge_drop=edge_drop, diffusion_drop=diffusion_drop, 
            transition_matrix=transition_matrix, weight_style=weight_style, 
            no_dst_attn=no_dst_attn, hop_norm=hop_norm, pos_emb=pos_emb, share_weights=share_weights, pre_act=pre_act, residual=True))
        if bn:
            self.norms = ModuleList()
            self.norms.append(BatchNorm1d(num_heads * n_hidden))
        else:
            self.norms = None

        for _ in range(n_layers - 2):
            self.convs.append(
                AGDNConv(n_hidden * num_heads, n_hidden, num_heads, K, 
                    attn_drop=attn_drop, edge_drop=edge_drop, diffusion_drop=diffusion_drop,
                    transition_matrix=transition_matrix, weight_style=weight_style, 
                    no_dst_attn=no_dst_attn, hop_norm=hop_norm, pos_emb=pos_emb, share_weights=share_weights, pre_act=pre_act, residual=True))
            if bn:
                self.norms.append(BatchNorm1d(num_heads * n_hidden))

        if n_layers > 1:
            self.convs.append(AGDNConv(n_hidden * num_heads, out_feats, num_heads, K, 
                attn_drop=attn_drop, edge_drop=edge_drop, diffusion_drop=diffusion_drop,
                transition_matrix=transition_matrix, weight_style=weight_style, 
                no_dst_attn=no_dst_attn, hop_norm=hop_norm, pos_emb=pos_emb, share_weights=share_weights, pre_act=pre_act, residual=True))
            if bn and output_bn:
                self.norms.append(BatchNorm1d(n_hidden))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        if self.norms is not None:
            for norm in self.norms:
                norm.reset_parameters()

    def forward(self, graph, feat, edge_feat=None):
        h = F.dropout(feat, self.input_drop, training=self.training)
        if self.residual:
            h_last = h
        for i, conv in enumerate(self.convs[:-1]):
            
            h = conv(graph, h, edge_feat=edge_feat).flatten(1)
            if self.residual:
                if h_last.shape[1] == h.shape[1]:
                    h = h + h_last
            if self.norms is not None:
                h = self.norms[i](h)
            
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.residual:
                h_last = h

        h = self.convs[-1](graph, h, edge_feat=edge_feat).mean(1)
        
        if self.norms is not None and len(self.norms) == len(self.convs):
            h = self.norms[-1](h)

        if len(self.convs) == 1:
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        if self.residual:
            if h_last.shape[1] == h.shape[1]:
                h = h + h_last
        return h


class MemAGDN(Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers,
                 num_heads, K,
                 dropout, input_drop, attn_drop, in_edge_feats=0, n_edge_hidden=1, weight_style="HA", residual=False):
        super(MemAGDN, self).__init__()
        self.residual = residual
        self.input_drop = input_drop
        if in_edge_feats > 0:
            self.edge_encoders = torch.nn.ModuleList()
        else:
            self.edge_encoders = None
        self.convs = ModuleList()
        self.norms = ModuleList()
        self.convs.append(MemAGDNConv(in_feats, n_hidden, num_heads, K, attn_drop=attn_drop, edge_feats=n_edge_hidden, weight_style=weight_style, residual=True, bias=False))
        self.norms.append(BatchNorm1d(num_heads * n_hidden))
        if in_edge_feats > 0:
            self.edge_encoders.append(Linear(in_edge_feats, n_edge_hidden))

        for _ in range(n_layers - 2):
            self.convs.append(
                MemAGDNConv(n_hidden * num_heads, n_hidden, num_heads, K, attn_drop=attn_drop, edge_feats=n_edge_hidden, weight_style=weight_style, residual=True, bias=False))
            self.norms.append(BatchNorm1d(num_heads * n_hidden))
            if in_edge_feats > 0:
                self.edge_encoders.append(Linear(in_edge_feats, n_edge_hidden))
        self.convs.append(MemAGDNConv(n_hidden * num_heads, out_feats, num_heads, K, attn_drop=attn_drop, edge_feats=n_edge_hidden, weight_style=weight_style, residual=True, bias=False))
        if in_edge_feats > 0:
            self.edge_encoders.append(Linear(in_edge_feats, n_edge_hidden))
        self.bias = Parameter(torch.FloatTensor(size=(1, out_feats)))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for norm in self.norms:
            norm.reset_parameters()
        
        if self.edge_encoders is not None:
            for encoder in self.edge_encoders:
                encoder.reset_parameters()
        torch.nn.init.zeros_(self.bias)

    def forward(self, graph, feat, edge_feat=None):
        h = F.dropout(feat, self.input_drop, training=self.training)
        
        # h_last = h
        for i, conv in enumerate(self.convs[:-1]):

            h = conv(graph, h, edge_feat=edge_feat).flatten(1)
            # if self.residual:
            #     if h_last.shape[-1] == h.shape[-1]:
            #         h += h_last
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            # h_last = h

        h = self.convs[-1](graph, h, edge_feat=edge_feat).mean(1)
        # if self.residual:
        #     if h_last.shape[1] == h.shape[1]:
        #         h = h + h_last
        h += self.bias
        return h


class DotPredictor(Module):
    def __init__(self):
        super(DotPredictor, self).__init__()

    def reset_parameters(self):
        return

    def forward(self, x_i, x_j):
        x = torch.sum(x_i * x_j, dim=-1)
        return x


class CosPredictor(Module):
    def __init__(self):
        super(CosPredictor, self).__init__()

    def reset_parameters(self):
        return
    
    def forward(self, x_i, x_j):
        x = torch.sum(x_i * x_j, dim=-1) / \
            torch.sqrt(torch.sum(x_i * x_i, dim=-1) * torch.sum(x_j * x_j, dim=-1)).clamp(min=1-9)
        return x
    
        
class LinkPredictor(Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers,
                 dropout, bn=False):
        super(LinkPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        if bn and n_layers > 1:
            self.bns = torch.nn.ModuleList()
        else:
            self.bns = None
        for i in range(n_layers):
            in_feats_ = in_feats if i == 0 else n_hidden
            out_feats_ = out_feats if i == n_layers - 1 else n_hidden
            self.lins.append(torch.nn.Linear(in_feats_, out_feats_))
            if bn and i < n_layers - 1:
                self.bns.append(torch.nn.BatchNorm1d(out_feats_))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        # x = torch.cat([x_i, x_j], dim=-1)
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                if self.bns is not None:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
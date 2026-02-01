from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F
from logging import getLogger
import numpy as np
from Global_Local_Attention import LocalChannelAttention, GlobalChannelAttention
from laet import LTAE


class NConv(nn.Module):
    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x, adj):
        """
        A * X

        Args:
            x(torch.tensor):  (B, input_channels, N, T)
            adj(torch.tensor):  N * N

        Returns:
            torch.tensor: (B, input_channels, N, T)
        """
        x = torch.einsum('ncwl,vw->ncvl', (x, adj))
        return x.contiguous()


class DyNconv(nn.Module):
    def __init__(self):
        super(DyNconv, self).__init__()

    def forward(self, x, adj):
        x = torch.einsum('ncvl,nvwl->ncwl', (x, adj))
        return x.contiguous()

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class ChannelSpecific(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=1, mid_reduction=1, loop_times=4, fuse_alpha=0.15):
        super(ChannelSpecific, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rel_reduction = rel_reduction
        # if in_channels == 3 or in_channels == 9:
        #     self.rel_channels = 8
        #     self.mid_channels = 16
        # else:
        self.rel_channels = in_channels // rel_reduction
        self.mid_channels = in_channels // mid_reduction
        # self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        # self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

        self.loop_times = loop_times
        self.fuse_alpha = fuse_alpha
        self.alpha = nn.Parameter(torch.ones(self.rel_reduction))

    def forward(self, x):
        #n, c, t, v = x.shape
        n, c, v, t = x.shape
        x1, x2 = self.conv3(x.mean(-1, keepdim=True)), self.conv3(x.mean(-1, keepdim=True))
        N, C, V, T = x1.shape
        x1 = x1.reshape(n, self.rel_reduction, int(C // self.rel_reduction), v, -1)
        x2 = x2.reshape(n, self.rel_reduction, int(C // self.rel_reduction), v, -1)
        graph = self.sigmoid(x1 * x2.transpose(-1,-2)) # torch.Size([16, 1, 32, 134, 1, 1])
        #graph = torch.einsum('nkctuv,k->nkctuv', graph, self.alpha)  # 64, 8, 10, 1, 25  N, Grp, Ch, T, V, V
        graph = graph.mean(0, keepdim=True).squeeze()
        #Grp, V_, _ = graph.shape
        #graph = graph.view(Grp * Ch, V, _)
        graph = graph.mean(0, keepdim=True).squeeze()
        return graph


class Linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(Linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class Prop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(Prop, self).__init__()
        self.nconv = NConv()
        self.mlp = Linear(c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        """

        Args:
            x(torch.tensor):  (B, c_in, N, T)
            adj(torch.tensor):  N * N

        Returns:
            torch.tensor: (B, c_out, N, T)
        """
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h, a)
        ho = self.mlp(h)
        return ho


class MixProp(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        """
        MixProp GCN

        Args:
            c_in: input
            c_out: output
            gdep: GCN layers
            dropout: dropout
            alpha: beta in paper
        """
        super(MixProp, self).__init__()
        self.nconv = NConv()
        #self.mlp = Linear((gdep+1)*c_in, c_out)
        self.mlp = Linear(96, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        """
        MixProp GCN

        Args:
            x(torch.tensor):  (B, c_in, N, T)
            adj(torch.tensor):  N * N

        Returns:
            torch.tensor: (B, c_out, N, T)
        """
        #check_adj = torch.eye(adj.size(0)).to(x.device)
        #print(check_adj.shape)
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]  # h(0) = h_in = x
        a = adj / d.view(-1, 1)  # A' = A * D^-1
        for i in range(self.gdep):
            # h(k) = alpha * h_in + (1 - alpha) * A' * H(k-1)
            # h: shape = (B, c_in, N, T)
            h = self.alpha*x + (1-self.alpha)*self.nconv(h, a)
            out.append(h)
        # ho: (B, c_in * (gdep + 1), N, T)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho  # (B, c_out, N, T)


class DyMixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(DyMixprop, self).__init__()
        self.nconv = DyNconv()
        self.mlp1 = Linear((gdep+1)*c_in, c_out)
        self.mlp2 = Linear((gdep+1)*c_in, c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = Linear(c_in, c_in)
        self.lin2 = Linear(c_in, c_in)

    def forward(self, x):
        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        adj = self.nconv(x1.transpose(2, 1), x2)
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2, 1), dim=2)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h, adj0)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho1 = self.mlp1(ho)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)
        return ho1+ho2


class Dilated1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(Dilated1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        self.tconv = nn.Conv2d(cin, cout, (1, 7), dilation=(1, dilation_factor))

    def forward(self, inputs):
        x = self.tconv(inputs)
        return x


class DilatedInception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(DilatedInception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        """

        Args:
            inputs: (B, C_in, N, T)

        Returns:
            torch.tensor: (B, C_out, N, T)

        """
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x


class GraphConstructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(GraphConstructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))-torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj*mask
        return adj

    def fulla(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))-torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj


class GraphGlobal(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(GraphGlobal, self).__init__()
        self.nnodes = nnodes
        self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)

    def forward(self, idx):
        return F.relu(self.A)


class GraphUndirected(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(GraphUndirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin1(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj*mask
        return adj


class GraphDirected(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(GraphDirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj*mask
        return adj


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, inputs, idx):
        if self.elementwise_affine:
            return F.layer_norm(inputs, tuple(inputs.shape[1:]),
                                self.weight[:, idx, :], self.bias[:, idx, :], self.eps)
        else:
            return F.layer_norm(inputs, tuple(inputs.shape[1:]),
                                self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class MLAGCN(nn.Module):
    def __init__(self, config, adj_mx=None):
        super().__init__()
        self.adj_mx = adj_mx
        self.feature_dim = config["var_len"]
        self.input_window = config["input_len"]
        self.output_window = config["output_len"]
        self.num_nodes = config["capacity"]
        self.data_diff = config["data_diff"]
        self.output_dim = 1
        self.device = config['device']

        self.gcn_true = config.get('gcn_true', True)
        self.add_apt = config.get('add_apt', True)
        self.gcn_depth = config.get('gcn_depth', 2)
        self.dropout = config.get('dropout', 0.3)
        self.subgraph_size = config.get('subgraph_size', 20)
        self.node_dim = config.get('node_dim', 40)
        self.dilation_exponential = config.get('dilation_exponential', 2)
        self.conv_channels = config.get('conv_channels', 16)
        self.residual_channels = config.get('residual_channels', 32)
        self.skip_channels = config.get('skip_channels', 64)
        self.end_channels = config.get('end_channels', 128)
        self.layers = config.get('layers', 3)
        self.propalpha = config.get('propalpha', 0.05)
        self.tanhalpha = config.get('tanhalpha', 3)
        self.layer_norm_affline = config.get('layer_norm_affline', True)

        self.static_feat = None
        self.idx = torch.arange(self.num_nodes).to(self.device)
        
        self._logger = getLogger()

        self.predefined_A = torch.tensor(self.adj_mx.astype(np.float32)) - torch.eye(self.num_nodes)
        self.predefined_A = self.predefined_A.to(self.device)
        self.static_feat = None

        self.global_attention = GlobalChannelAttention(self.num_nodes, 3)
        self.local_attention = LocalChannelAttention(self.num_nodes, 3)

        self.identity_matrix = nn.Parameter(torch.eye(self.num_nodes))
        #self.late = LTAE(self.feature_dim)
        self.Channelcorrelation = ChannelSpecific(self.residual_channels, self.residual_channels)

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.skip_convs_W = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.norm_W = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=self.feature_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))
        if self.add_apt:
            self.gc = GraphConstructor(self.num_nodes, self.subgraph_size, self.node_dim,
                                       self.device, alpha=self.tanhalpha, static_feat=self.static_feat)

        kernel_size = 7
        if self.dilation_exponential > 1:
            self.receptive_field = int(self.output_dim + (kernel_size-1) * (self.dilation_exponential**self.layers-1)
                                       / (self.dilation_exponential - 1))
        else:
            self.receptive_field = self.layers * (kernel_size-1) + self.output_dim

        for i in range(1):
            if self.dilation_exponential > 1:
                rf_size_i = int(1 + i * (kernel_size-1) * (self.dilation_exponential**self.layers-1)
                                / (self.dilation_exponential - 1))
            else:
                rf_size_i = i * self.layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, self.layers+1):
                if self.dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1) * (self.dilation_exponential**j - 1)
                                    / (self.dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(LTAE(self.residual_channels))
                #self.gate_convs.append(DilatedInception(self.residual_channels, self.conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                     out_channels=self.residual_channels, kernel_size=(1, 1)))
                if self.input_window > self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels, out_channels=32,
                                                     kernel_size=1))
                    self.skip_convs_W.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels, out_channels=self.skip_channels,
                                                     kernel_size=1))

                if self.gcn_true:
                    self.gconv1.append(MixProp(self.conv_channels, self.residual_channels,
                                               self.gcn_depth, self.dropout, self.propalpha))
                    self.gconv2.append(MixProp(self.conv_channels, self.residual_channels,
                                               self.gcn_depth, self.dropout, self.propalpha))

                if self.input_window > self.receptive_field:
                    self.norm.append(LayerNorm((self.residual_channels, self.num_nodes,
                                                self.input_window),
                                               elementwise_affine=self.layer_norm_affline))
                    self.norm_W.append(LayerNorm((self.residual_channels, self.num_nodes,
                                                7),
                                               elementwise_affine=self.layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((self.residual_channels, self.num_nodes,
                                                self.receptive_field),
                                               elementwise_affine=self.layer_norm_affline))

                new_dilation *= self.dilation_exponential

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                    out_channels=self.end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.output_window, kernel_size=(1, 1), bias=True)
        if self.input_window > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=self.feature_dim,
                                   out_channels=self.skip_channels,
                                   kernel_size=(1, self.input_window), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels,
                                   out_channels=self.skip_channels,
                                   kernel_size=1, bias=True)
        else:
            self.skip0 = nn.Conv2d(in_channels=self.feature_dim,
                                   out_channels=self.skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels,
                                   out_channels=self.skip_channels, kernel_size=1, bias=True)

        self._logger.info('receptive_field: ' + str(self.receptive_field))

    def forward(self, batch_x, batch_y, data_mean, data_scale):
        inputs = batch_x  # (batch_size, num_nodes, input_window, feature_dim)
        bz, id_len, input_len, var_len = inputs.shape
        channel_attention_input = batch_x

        global_channel = self.global_attention(channel_attention_input) + self.identity_matrix
        local_channel = self.local_attention(channel_attention_input) + self.identity_matrix


        inputs = inputs[:, :, :, 2:]
        inputs = (inputs - data_mean) / data_scale

        if self.data_diff:
            # add Data Differential Features
            inputs_diff = inputs[:, :, 1:, -1:] - inputs[:, :, :-1, -1:]
            inputs_diff = torch.cat((torch.zeros(bz, id_len, 1, 1).to(inputs.device), inputs_diff), 2)
            inputs = torch.cat((inputs, inputs_diff), 3)

        inputs = inputs.permute((0, 3, 1, 2))  # (batch_size, feature_dim, num_nodes, input_window) torch.Size([32, 5, 134, 144])

        assert inputs.size(3) == self.input_window, 'input sequence length not equal to preset sequence length'

        if self.input_window < self.receptive_field:
            inputs = nn.functional.pad(inputs, (self.receptive_field-self.input_window, 0, 0, 0))

        x = self.start_conv(inputs)
        A_correlation = self.Channelcorrelation(x)

        if self.gcn_true:
            if self.add_apt:
                adp = self.gc(self.idx) + global_channel + local_channel + A_correlation
            else:
                adp = self.predefined_A + global_channel + local_channel + A_correlation


        skip = self.skip0(F.dropout(inputs, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            #filters = self.filter_convs[i](x)
            #filters = torch.tanh(filters)
            #gate = self.gate_convs[i](x)
            #gate = torch.sigmoid(gate)
            #x = filters[1]
            #x = F.dropout(x, self.dropout, training=self.training)
            #s = x
            #s = self.skip_convs[i](s)
            #skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1, 0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            b_, c_, h_, w_ = x.shape
            if w_ != 7:
                x = self.norm[i](x, self.idx) # torch.Size([32, 32, 134, 144])
            else:
                x = self.norm_W[i](x, self.idx)

            filters = self.filter_convs[i](x)
            x = filters[1]
            x = F.dropout(x, self.dropout, training=self.training)
            sd, _, _ = x.shape
            x = x.view(bz, sd, id_len, w_)
            if sd == 16:
                x = self.skip_convs[i](x)
            else:
                x = self.skip_convs_W[i](x)

        x = self.skipE(x)
        skip = x + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)  # (B,T,N,F)
        return x[..., 0].permute((0, 2, 1))  # (B,N,T)

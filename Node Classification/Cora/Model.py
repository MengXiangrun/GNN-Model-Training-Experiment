import torch
import torch.nn.functional as F
from absl.testing.parameterized import parameters
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, Linear
import torch_geometric
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, Linear
import torch.nn as nn
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops


class Linear(torch.nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.linear = torch_geometric.nn.Linear(in_channels=-1,
                                                out_channels=self.out_dim,
                                                weight_initializer='kaiming_uniform',
                                                bias=True,
                                                bias_initializer=None)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        return self.linear(x)


class GCN(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_layer=3,
                 dropout=0.5,
                 num_head=1,
                 is_InLinear=False,
                 is_ResidualConnection=False,
                 is_LayerNorm=False,
                 is_BatchNorm1d=False,
                 is_Sum=False,
                 is_SelfLoop=False):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layer = num_layer
        self.num_head = num_head
        self.head_dim = self.hidden_dim // self.num_head
        self.dropout = dropout
        self.model_name = 'GCN'

        self.is_InLinear = is_InLinear
        self.is_ResidualConnection = is_ResidualConnection
        self.is_LayerNorm = is_LayerNorm
        self.is_BatchNorm1d = is_BatchNorm1d
        self.is_Sum = is_Sum
        self.is_SelfLoop = is_SelfLoop

        self.Gnn_list = torch.nn.ModuleList()
        self.Residual_list = torch.nn.ModuleList()
        self.LayerNorm_list = torch.nn.ModuleList()
        self.BatchNorm1d_list = torch.nn.ModuleList()

        self.in_linear = Linear(hidden_dim)

        for layer_index in range(0, self.num_layer):
            gnn = GCNConv(in_channels=-1, out_channels=hidden_dim, cached=False, normalize=True)
            residual = Linear(hidden_dim)
            layernorm = torch.nn.LayerNorm(hidden_dim)
            batchnorm1d = torch.nn.BatchNorm1d(hidden_dim)
            self.Gnn_list.append(gnn)
            self.Residual_list.append(residual)
            self.LayerNorm_list.append(layernorm)
            self.BatchNorm1d_list.append(batchnorm1d)

        self.out_linear = torch.nn.Sequential(
            Linear(hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            Linear(out_dim)
        )

    def forward(self, x, edge_index, edge_attribute=None):
        node_feature = x.clone()

        if self.is_SelfLoop:
            edge_index, edge_attribute = self.add_self_loops(x=x,
                                                             edge_index=edge_index,
                                                             edge_attr=edge_attribute)

        if self.is_InLinear:
            node_emb = self.in_linear.forward(x=node_feature)
            node_emb = torch.nn.functional.dropout(input=node_emb,
                                                   p=self.dropout,
                                                   training=self.training)
        else:
            node_emb = node_feature

        added_node_emb = None
        for layer_index in range(self.num_layer):
            node_emb = self.Gnn_list[layer_index].forward(x=node_emb,
                                                          edge_index=edge_index,
                                                          edge_weight=edge_attribute)

            if self.is_ResidualConnection:
                residual_emb = self.Residual_list[layer_index].forward(x=node_emb)
                node_emb = node_emb + residual_emb

            if self.is_LayerNorm:
                node_emb = self.LayerNorm_list[layer_index].forward(input=node_emb)

            elif self.is_BatchNorm1d:
                node_emb = self.BatchNorm1d_list[layer_index].forward(input=node_emb)

            node_emb = torch.nn.functional.relu(input=node_emb)
            node_emb = torch.nn.functional.dropout(input=node_emb,
                                                   p=self.dropout,
                                                   training=self.training)

            if self.is_Sum:
                if added_node_emb is not None:
                    node_emb = node_emb + added_node_emb
                added_node_emb = node_emb.clone()

        node_emb = self.out_linear(node_emb)

        return node_emb

    def add_self_loops(self, x, edge_index, edge_attr=None):
        num_edge_index = edge_index.shape[1]

        mask = edge_index[0] == edge_index[1]
        exist_self_loops = edge_index[:, mask]
        exist_self_loops_node = exist_self_loops[0]

        num_node = x.shape[0]
        need_self_loops_node = torch.arange(start=0,
                                            end=num_node,
                                            step=1,
                                            device=edge_index.device,
                                            dtype=edge_index.dtype)

        mask = torch.isin(elements=need_self_loops_node,
                          test_elements=exist_self_loops_node)
        mask = ~ mask
        need_self_loops_node = need_self_loops_node[mask]

        new_self_loops = torch.stack([need_self_loops_node, need_self_loops_node], dim=0)

        num_new_self_loops = new_self_loops.shape[1]
        new_edge_index = torch.cat([edge_index, new_self_loops], dim=1)

        if edge_attr is not None:
            num_edge_attr = edge_attr.shape[0]
            edge_attr_dim = edge_attr.shape[1]

            assert num_edge_index >= num_edge_attr

            new_edge_attr = torch.ones(size=(num_new_self_loops, edge_attr_dim),
                                       device=edge_index.device,
                                       dtype=edge_index.dtype)
            new_edge_attr = torch.cat([edge_attr, new_edge_attr], dim=1)
        else:
            new_edge_attr = None

        return new_edge_index, new_edge_attr

    def output_parameter(self):
        # 获取所有 self 变量
        all_variables = vars(self)
        parameters = {}
        for variable_name, variable_value in all_variables.items():
            if isinstance(variable_value, (int, float, str)) and not isinstance(variable_value, bool):
                parameters[variable_name] = variable_value

        return parameters


class GAT(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_layer=3,
                 dropout=0.5,
                 num_head=1,
                 is_InLinear=False,
                 is_ResidualConnection=False,
                 is_LayerNorm=False,
                 is_BatchNorm1d=False,
                 is_Sum=False,
                 is_SelfLoop=False):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layer = num_layer
        self.num_head = num_head
        self.head_dim = self.hidden_dim // self.num_head
        self.dropout = dropout
        self.model_name = 'GAT'

        self.is_InLinear = is_InLinear
        self.is_ResidualConnection = is_ResidualConnection
        self.is_LayerNorm = is_LayerNorm
        self.is_BatchNorm1d = is_BatchNorm1d
        self.is_Sum = is_Sum
        self.is_SelfLoop = is_SelfLoop

        self.Gnn_list = torch.nn.ModuleList()
        self.Residual_list = torch.nn.ModuleList()
        self.LayerNorm_list = torch.nn.ModuleList()
        self.BatchNorm1d_list = torch.nn.ModuleList()

        self.in_linear = Linear(hidden_dim)

        for layer_index in range(0, self.num_layer):
            gnn = GATConv(in_channels=-1, out_channels=self.head_dim, heads=self.num_head,
                          concat=True, add_self_loops=False, bias=False)
            residual = Linear(hidden_dim)
            layernorm = torch.nn.LayerNorm(hidden_dim)
            batchnorm1d = torch.nn.BatchNorm1d(hidden_dim)
            self.Gnn_list.append(gnn)
            self.Residual_list.append(residual)
            self.LayerNorm_list.append(layernorm)
            self.BatchNorm1d_list.append(batchnorm1d)

        self.out_linear = torch.nn.Sequential(
            Linear(hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            Linear(out_dim)
        )

    def forward(self, x, edge_index, edge_attribute=None):
        node_feature = x.clone()

        if self.is_SelfLoop:
            edge_index, edge_attribute = self.add_self_loops(x=x,
                                                             edge_index=edge_index,
                                                             edge_attr=edge_attribute)

        if self.is_InLinear:
            node_emb = self.in_linear.forward(x=node_feature)
            node_emb = torch.nn.functional.dropout(input=node_emb,
                                                   p=self.dropout,
                                                   training=self.training)
        else:
            node_emb = node_feature

        added_node_emb = None
        for layer_index in range(self.num_layer):
            node_emb = self.Gnn_list[layer_index].forward(x=node_emb,
                                                          edge_index=edge_index,
                                                          edge_weight=edge_attribute)

            if self.is_ResidualConnection:
                residual_emb = self.Residual_list[layer_index].forward(x=node_emb)
                node_emb = node_emb + residual_emb

            if self.is_LayerNorm:
                node_emb = self.LayerNorm_list[layer_index].forward(input=node_emb)

            elif self.is_BatchNorm1d:
                node_emb = self.BatchNorm1d_list[layer_index].forward(input=node_emb)

            node_emb = torch.nn.functional.relu(input=node_emb)
            node_emb = torch.nn.functional.dropout(input=node_emb,
                                                   p=self.dropout,
                                                   training=self.training)

            if self.is_Sum:
                if added_node_emb is not None:
                    node_emb = node_emb + added_node_emb
                added_node_emb = node_emb.clone()

        node_emb = self.out_linear(node_emb)

        return node_emb

    def add_self_loops(self, x, edge_index, edge_attr=None):
        num_edge_index = edge_index.shape[1]

        mask = edge_index[0] == edge_index[1]
        exist_self_loops = edge_index[:, mask]
        exist_self_loops_node = exist_self_loops[0]

        num_node = x.shape[0]
        need_self_loops_node = torch.arange(start=0,
                                            end=num_node,
                                            step=1,
                                            device=edge_index.device,
                                            dtype=edge_index.dtype)

        mask = torch.isin(elements=need_self_loops_node,
                          test_elements=exist_self_loops_node)
        mask = ~ mask
        need_self_loops_node = need_self_loops_node[mask]

        new_self_loops = torch.stack([need_self_loops_node, need_self_loops_node], dim=0)

        num_new_self_loops = new_self_loops.shape[1]
        new_edge_index = torch.cat([edge_index, new_self_loops], dim=1)

        if edge_attr is not None:
            num_edge_attr = edge_attr.shape[0]
            edge_attr_dim = edge_attr.shape[1]

            assert num_edge_index >= num_edge_attr

            new_edge_attr = torch.ones(size=(num_new_self_loops, edge_attr_dim),
                                       device=edge_index.device,
                                       dtype=edge_index.dtype)
            new_edge_attr = torch.cat([edge_attr, new_edge_attr], dim=1)
        else:
            new_edge_attr = None

        return new_edge_index, new_edge_attr

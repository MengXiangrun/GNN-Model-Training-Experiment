import random
import numpy as np
import pandas as pd
import torch_geometric
import torch


class HomogeneousGraph():
    def __init__(self):
        self.node_feature = None
        self.node_label = None

        self.edge_feature = None
        self.edge_label = None

        self.message_edge = None

        self.predict_node = None
        self.predict_edge = None

        self.graph_label = None


class Cora(torch.nn.Module):
    def __init__(self, split=False, feature_normalization=False):
        super().__init__()
        self.split = split
        if self.split not in ['public', 'full', 'geom-gcn']:
            print("split not in ['public', 'full', 'geom-gcn']")
            assert False

        self.feature_normalization = feature_normalization
        data = torch_geometric.datasets.Planetoid(root=f'./', name='cora', split=self.split, transform=None)[0]

        if self.feature_normalization in ['l1_L1']:
            data.x = self.L1Normalize(node_feature=data.x)
        if self.feature_normalization in ['l2_L2']:
            data.x = self.L2Normalize(node_feature=data.x)
        if self.feature_normalization in ['mm_minmax']:
            data.x = self.MinMaxNormalize(node_feature=data.x)
        if self.feature_normalization in ['zs_zscore']:
            data.x = self.ZScoreNormalize(node_feature=data.x)
        if self.feature_normalization in ['max']:
            data.x = self.MaxNormalize(node_feature=data.x)
        if self.feature_normalization in ['lg_log']:
            data.x = self.LogNormalize(node_feature=data.x)
        if self.feature_normalization in ['rb_robust']:
            data.x = self.RobustNormalize(node_feature=data.x)
        if self.feature_normalization in ['uv_unitvector']:
            data.x = self.UnitVectorNormalize(node_feature=data.x)
        if self.feature_normalization in ['ds_decimalscaling']:
            data.x = self.DecimalScalingNormalize(node_feature=data.x)

        assert data.x.shape[0] == data.num_nodes

        self.train_graph = HomogeneousGraph()
        self.train_graph.message_edge = data.edge_index
        self.train_graph.node_feature = data.x
        self.train_graph.node_label = data.y
        self.train_graph.predict_node = torch.where(data.train_mask)[0]

        self.valid_graph = HomogeneousGraph()
        self.valid_graph.message_edge = data.edge_index
        self.valid_graph.node_feature = data.x
        self.valid_graph.node_label = data.y
        self.valid_graph.predict_node = torch.where(data.val_mask)[0]

        self.test_graph = HomogeneousGraph()
        self.test_graph.message_edge = data.edge_index
        self.test_graph.node_feature = data.x
        self.test_graph.node_label = data.y
        self.test_graph.predict_node = torch.where(data.test_mask)[0]

        # self.distribution(data=data)

        print('Done')

    def L1Normalize(self, node_feature: torch.Tensor):
        l1_norm = torch.norm(node_feature, p=1, dim=-1, keepdim=True)
        l1_norm = l1_norm.clamp(min=1.0)
        normalized_feature = node_feature / l1_norm
        return normalized_feature

    def L2Normalize(self, node_feature: torch.Tensor):
        l2_norm = torch.norm(node_feature, p=2, dim=-1, keepdim=True)
        l2_norm = l2_norm.clamp(min=1.0)
        normalized_feature = node_feature / l2_norm
        return normalized_feature

    def MinMaxNormalize(self, node_feature: torch.Tensor):
        min_val = node_feature.min(dim=-1, keepdim=True).values
        max_val = node_feature.max(dim=-1, keepdim=True).values
        normalized_feature = (node_feature - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero
        return normalized_feature

    def ZScoreNormalize(self, node_feature: torch.Tensor):
        mean = node_feature.mean(dim=-1, keepdim=True)
        std = node_feature.std(dim=-1, keepdim=True)
        normalized_feature = (node_feature - mean) / (std + 1e-8)  # Avoid division by zero
        return normalized_feature

    def MaxNormalize(self, node_feature: torch.Tensor):
        max_val = node_feature.max(dim=-1, keepdim=True).values
        normalized_feature = node_feature / (max_val + 1e-8)  # Avoid division by zero
        return normalized_feature

    def LogNormalize(self, node_feature: torch.Tensor):
        normalized_feature = torch.log(1 + node_feature)
        return normalized_feature

    def RobustNormalize(self, node_feature: torch.Tensor):
        median = node_feature.median(dim=-1, keepdim=True).values
        q1 = node_feature.quantile(0.25, dim=-1, keepdim=True)
        q3 = node_feature.quantile(0.75, dim=-1, keepdim=True)
        iqr = q3 - q1
        normalized_feature = (node_feature - median) / (iqr + 1e-8)  # Avoid division by zero
        return normalized_feature

    def UnitVectorNormalize(self, node_feature: torch.Tensor):
        l2_norm = torch.norm(node_feature, p=2, dim=-1, keepdim=True)
        normalized_feature = node_feature / (l2_norm + 1e-8)  # Avoid division by zero
        return normalized_feature

    def DecimalScalingNormalize(self, node_feature: torch.Tensor):
        max_val = node_feature.abs().max()
        j = torch.ceil(torch.log10(max_val + 1e-8))  # Avoid log(0)
        normalized_feature = node_feature / (10 ** j)
        return normalized_feature

    def distribution(self, data):
        train_node = torch.where(data.train_mask)[0].detach().cpu().numpy().tolist()
        train_node_label = data.y[train_node].detach().cpu().numpy().tolist()
        train_node_label = pd.DataFrame([train_node, train_node_label])
        train_node_label = train_node_label.T
        train_node_label.columns = ['node', 'label']

        valid_node = torch.where(data.val_mask)[0].detach().cpu().numpy().tolist()
        valid_node_label = data.y[valid_node].detach().cpu().numpy().tolist()
        valid_node_label = pd.DataFrame([valid_node, valid_node_label])
        valid_node_label = valid_node_label.T
        valid_node_label.columns = ['node', 'label']

        test_node = torch.where(data.test_mask)[0].detach().cpu().numpy().tolist()
        test_node_label = data.y[test_node].detach().cpu().numpy().tolist()
        test_node_label = pd.DataFrame([test_node, test_node_label])
        test_node_label = test_node_label.T
        test_node_label.columns = ['node', 'label']

        unique_label = sorted(set(data.y.detach().cpu().numpy().tolist()))
        train_data_statistics = ['train']
        valid_data_statistics = ['valid']
        test_data_statistics = ['test']
        node_class = ['datatype']
        for label in unique_label:
            node_class.append(str(label))

            num_node = train_node_label[train_node_label['label'] == label].shape[0]
            train_data_statistics.append(num_node)

            num_node = valid_node_label[valid_node_label['label'] == label].shape[0]
            valid_data_statistics.append(num_node)

            num_node = test_node_label[test_node_label['label'] == label].shape[0]
            test_data_statistics.append(num_node)

        data_statistics = [train_data_statistics, valid_data_statistics, test_data_statistics]
        self.data_distribution = pd.DataFrame(data_statistics, columns=node_class)

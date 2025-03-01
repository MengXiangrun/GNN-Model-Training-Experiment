import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from Model import GCN
from Dataset import Cora
import pandas as pd


class EarlyStopping():
    def __init__(self, patience, threshold):
        self.val_loss_list = list()
        self.best_model_parameter = 0
        self.patience = patience
        self.threshold = threshold
        self.count = 0
        self.stop = False

    def save(self, model, val_loss: float):
        self.val_loss_list.append(val_loss)
        print()

        min_val_loss = min(self.val_loss_list)
        min_val_loss_epoch = self.val_loss_list.index(min_val_loss)
        now_val_loss = self.val_loss_list[-1]
        now_val_loss_epoch = self.val_loss_list.index(now_val_loss)

        if now_val_loss <= min_val_loss:
            self.best_model_parameter = model.state_dict()

        if (now_val_loss_epoch - min_val_loss_epoch) >= self.patience:
            self.stop = True


seed = 121
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda")

result = {}
mean_test_accuracy = []
for split in ['public', 'full', 'geom-gcn']:
    result[split] = {}

    data = Cora(split=split, feature_normalization=0)
    num_class = torch.unique(data.test_graph.node_label).shape[0]

    model = GCN(in_dim=-1,
                hidden_dim=128,
                out_dim=num_class,
                num_layer=3,
                dropout=0.7,
                num_head=1,
                is_InLinear=1,
                is_ResidualConnection=0,
                is_LayerNorm=1,
                is_BatchNorm1d=0,
                is_Sum=0,
                is_SelfLoop=0)

    CrossEntropyLoss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0005, lr=0.0005)
    stop = EarlyStopping(patience=30, threshold=0.0001)

    data.train_graph.to_device(device=device)
    data.valid_graph.to_device(device=device)
    model = model.to(device=device)

    for epoch in range(10000):
        if stop.stop:
            break
        model.train()
        optimizer.zero_grad()

        # forward
        node_emb = model.forward(x=data.train_graph.node_feature, edge_index=data.train_graph.message_edge)

        # prediction
        true_label = torch.nn.functional.one_hot(input=data.train_graph.node_label, num_classes=num_class)
        true_label = true_label.to(node_emb.dtype)
        true_label = true_label[data.train_graph.predict_node]

        pred_label = node_emb
        pred_label = pred_label[data.train_graph.predict_node]

        # loss
        loss = CrossEntropyLoss(input=pred_label, target=true_label)
        loss.backward()
        optimizer.step()

        # metrics
        true_label = true_label.max(1)[1]
        pred_label = pred_label.max(1)[1]
        true_label = true_label.detach().cpu().numpy()
        pred_label = pred_label.detach().cpu().numpy()
        accuracy = accuracy_score(y_true=true_label, y_pred=pred_label)

        print(f'epoch {epoch} train loss {loss.item():.4f} accuracy {accuracy:.4f}')

        with torch.no_grad():
            model.eval()

            # forward
            node_emb = model.forward(x=data.valid_graph.node_feature, edge_index=data.valid_graph.message_edge)

            # prediction
            true_label = torch.nn.functional.one_hot(input=data.valid_graph.node_label, num_classes=num_class)
            true_label = true_label.to(node_emb.dtype)
            true_label = true_label[data.valid_graph.predict_node]

            pred_label = node_emb
            pred_label = pred_label[data.valid_graph.predict_node]

            # loss
            loss = CrossEntropyLoss(input=pred_label, target=true_label)

            # metrics
            true_label = true_label.max(1)[1]
            pred_label = pred_label.max(1)[1]
            true_label = true_label.detach().cpu().numpy()
            pred_label = pred_label.detach().cpu().numpy()
            accuracy = accuracy_score(y_true=true_label, y_pred=pred_label)

            print(f'epoch {epoch} valid loss {loss.item():.4f} accuracy {accuracy:.4f}')
            stop.save(model=model, val_loss=-accuracy * 10000 + loss.item())

    # test
    data.test_graph.to_device(device=device)
    with torch.no_grad():
        model.eval()
        model.load_state_dict(stop.best_model_parameter)

        # forward
        node_emb = model.forward(x=data.test_graph.node_feature, edge_index=data.test_graph.message_edge)

        # prediction
        true_label = torch.nn.functional.one_hot(input=data.test_graph.node_label, num_classes=num_class)
        true_label = true_label.to(node_emb.dtype)
        true_label = true_label[data.test_graph.predict_node]

        pred_label = node_emb
        pred_label = pred_label[data.test_graph.predict_node]

        # loss
        loss = CrossEntropyLoss(input=pred_label, target=true_label)

        # metrics
        true_label = true_label.max(1)[1]
        pred_label = pred_label.max(1)[1]
        true_label = true_label.detach().cpu().numpy()
        pred_label = pred_label.detach().cpu().numpy()
        accuracy = accuracy_score(y_true=true_label, y_pred=pred_label)

        print(f'test loss {loss.item():.4f} accuracy {accuracy:.4f}')
        print('Done')
        print()
        mean_test_accuracy.append(accuracy)

    result[split] = model.output_parameter()
    result[split]['seed'] = seed
    result[split]['test_accuracy'] = accuracy
    result[split]['split'] = split

mean_test_accuracy = np.array(mean_test_accuracy)
mean_test_accuracy = np.mean(mean_test_accuracy)
result['mean_test_accuracy'] = float(mean_test_accuracy)
print('mean_test_accuracy:', mean_test_accuracy)
print(result)

result_list = []
for key, value in result.items():
    if isinstance(value, dict):
        value = pd.DataFrame([value])
    if isinstance(value, (int, float, bool, str)):
        value = pd.DataFrame([value], columns=[key])
    result_list.append(value)
result = pd.concat(result_list, axis=0)

from datetime import datetime

current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
filename = f'result_{current_time}.xlsx'
result.to_excel(filename, index=False)

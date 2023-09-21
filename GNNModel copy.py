import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, average_precision_score
import torch.nn as nn
import matplotlib.pyplot as plt
# Check if a GPU is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn.functional as F
from torch.nn import Linear

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channel=16):
        super().__init__()
        
        self.conv1 = GCNConv(num_features, hidden_channel, aggr="mean")
        self.conv2 = GCNConv(hidden_channel, num_classes, aggr="mean")

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
class GCN4(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channel=16):
        super().__init__()
        self.conv1 = GCNConv(num_features, num_classes, aggr="mean")

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)
    

class GCN1(torch.nn.Module):
    def __init__(self,  num_features, num_classes, hidden_channel=16):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channel, aggr="mean")
        self.conv2 = GCNConv(hidden_channel, 4, aggr="mean")
        self.classifier = Linear(4, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.classifier(x)
        return x
    
    
    
class GCN2(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channel=30):
        super(GCN2, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channel, aggr="mean")
        self.conv2 = GCNConv(hidden_channel, 4, aggr="mean")
        self.conv3 = GCNConv(4, 4)
        self.classifier = Linear(4, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        # x = torch.sigmoid(x) 
        return x



class GCN3(torch.nn.Module):
# Satyaki Code...
  def __init__(self, num_features, num_classes, hidden_channel=4):
    super(GCN3, self).__init__()
    torch.manual_seed(42)
    self.conv1 = GCNConv(num_features, hidden_channel, aggr="mean")
    self.conv2 = GCNConv(hidden_channel, hidden_channel, aggr="mean")
    self.conv3 = GCNConv(hidden_channel, 2)
    self.classifier = Linear(2, num_classes)

  def forward(self, data):
    x, edge_index = data.x, data.edge_index
    h = self.conv1(x, edge_index)
    h = h.tanh()
    h = self.conv2(h, edge_index)
    h = h.tanh()
    h = self.conv3(h, edge_index)
    h = h.tanh()
    out = self.classifier(h)
    return out
  

class SAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channel=4):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, num_classes, aggr="mean") # max, mean, add ...)
        # self.conv2 = SAGEConv(hidden_channel, num_classes, aggr="mean") # max, mean, add ...)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        # x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)
    


def train_node_classifier(model, graph, optimizer, criterion, n_epochs=200):
    L_train, L_val, ep = [],[], []
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(graph)
        loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()

        # pred = out.argmax(dim=1)
        acc, val_loss = eval_node_classifier(model, graph, graph.val_mask, criterion)

        L_train.append(loss.item())
        L_val.append(val_loss)
        ep.append(epoch)
        if epoch % 10 ==0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Acc: {acc:.3f}')

    df = pd.DataFrame({'epoch': ep, 'Train Loss': L_train, 'Val Loss': L_val})
    return model, df


def eval_node_classifier(model, graph, mask, criterion=nn.CrossEntropyLoss()):
    model.eval()
    with torch.no_grad():
        outputs = model(graph)
        loss = criterion(outputs[mask], graph.y[mask])
        pred = outputs.argmax(dim=1)
        correct = (pred[mask] == graph.y[mask]).sum()
        acc = int(correct) / int(mask.sum())
    return acc, loss.item()


def create_confusion_matrix(predicted, true_labels):
    # print(sum(x != y for x, y in zip(predicted, true_labels))/998)
    print(confusion_matrix(true_labels, predicted))
    print('F1 score = ', f1_score(true_labels, predicted, average='macro'))
    print('Precision score = ', precision_score(true_labels, predicted, average='macro'))
    print('AUC Precision score = ', average_precision_score(true_labels, predicted, average='macro'))
    

def plt_performance(df):
    # Plot the DataFrame
    plt.figure(figsize=(8, 6))
    plt.plot(df['epoch'], df['Train Loss'], label='Train Loss')
    plt.plot(df['epoch'], df['Val Loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()




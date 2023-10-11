import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, multilabel_confusion_matrix, precision_score, average_precision_score
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn.functional as F
from torch.nn import Linear

# Check if a GPU is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




class GCN_MLC(torch.nn.Module):
    '''Multi-label classification GNN model'''
    def __init__(self, num_features, num_classes, hidden_channel=16):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channel)
        self.fc = torch.nn.Linear(hidden_channel, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)  # Apply ReLU activation to hidden layer
        x = self.fc(x)  # Linear layer with sigmoid activation
        return x


class SAGE_MLC(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channel=16):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_channel, aggr="mean") # max, mean, add ...)
        self.fc = torch.nn.Linear(hidden_channel, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)  # Apply ReLU activation to hidden layer
        x = self.fc(x)  # Linear layer with sigmoid activation
        return x

    
class GCNRegression(torch.nn.Module):
    def __init__(self, num_features, hidden_channel=16):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channel)
        self.fc = nn.Linear(hidden_channel, 1)  # Output layer with one neuron for regression

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.fc(x)  # Linear activation for regression
        return x  # Return the continuous prediction


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
    def __init__(self, num_features, num_classes, hidden_channel=16):
        super(SAGE, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channel, aggr="mean")
        self.conv2 = SAGEConv(hidden_channel, num_classes, aggr="mean") # max, mean, add ...)
        # self.conv2 = SAGEConv(hidden_channel, num_classes, aggr="mean") # max, mean, add ...)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = self.conv2(x, data.edge_index)
        # x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)

    


def train_node_classifier(model, graph, optimizer, criterion, n_epochs=200):
    L_train, L_val, ep = [],[], []
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(graph)

        # print(out.shape)
        # print(graph.y.shape)

        loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()

        # pred = out.argmax(dim=1)
        acc, val_loss = eval_node_classifier(model, graph, criterion)

        L_train.append(loss.item())
        L_val.append(val_loss)
        ep.append(epoch)
        if epoch % 10 ==0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Acc: {acc:.3f}')

    df = pd.DataFrame({'epoch': ep, 'Train Loss': L_train, 'Val Loss': L_val})
    return model, df


def eval_node_classifier(model, graph, criterion=nn.CrossEntropyLoss()):
    mask = graph.val_mask
    model.eval()
    with torch.no_grad():
        out = model(graph)
        loss    = criterion(out[mask], graph.y[mask])

        # pred    = out.argmax(dim=1)
        pred    = out
        
        # print(pred.shape)
        # print(graph.y.shape)   

        correct = (pred[mask] == graph.y[mask]).sum()
        acc     = int(correct) / int(mask.sum())

    return acc, loss.item()


def create_confusion_matrix(predicted, true_labels):
    # print(sum(x != y for x, y in zip(predicted, true_labels))/998)
    print(confusion_matrix(true_labels, predicted))
    print('F1 score = ', f1_score(true_labels, predicted, average='macro'))
    print('Precision score = ', precision_score(true_labels, predicted, average='macro'))
    print('AUC Precision score = ', average_precision_score(true_labels, predicted, average='macro'))


def create_multilabel_confusion_matrix(pred, correct):
    # Compute the classification report
    classification_rep = classification_report(correct, pred)
    print(classification_rep)

    # Calculate the multilabel confusion matrix
    mcm = multilabel_confusion_matrix(correct, pred)

    # Calculate F1 score, precision score, and AUC Precision score for each label
    f1_scores = f1_score(correct, pred, average=None)
    precision_scores = precision_score(correct, pred, average=None)
    auc_precision_scores = average_precision_score(correct, pred, average=None)

    # Calculate macro-averaged metrics
    macro_f1 = f1_score(correct, pred, average='macro', zero_division=1)
    macro_precision = precision_score(correct, pred, average='macro')
    macro_auc_precision = average_precision_score(correct, pred, average='macro')

    # Calculate micro-averaged metrics
    micro_f1 = f1_score(correct, pred, average='micro')
    micro_precision = precision_score(correct, pred, average='micro')
    micro_auc_precision = average_precision_score(correct, pred, average='micro')

    print("Averaged Metrics:")
    print(f"Averaged F1 score = {f1_scores}")
    print(f"Averaged Precision score = {precision_scores}")
    print(f"Averaged AUC Precision score = {auc_precision_scores}\n")


    print("Macro-Averaged Metrics:")
    print(f"Macro-Averaged F1 score = {macro_f1}")
    print(f"Macro-Averaged Precision score = {macro_precision}")
    print(f"Macro-Averaged AUC Precision score = {macro_auc_precision}\n")

    print("Micro-Averaged Metrics:")
    print(f"Micro-Averaged F1 score = {micro_f1}")
    print(f"Micro-Averaged Precision score = {micro_precision}")
    print(f"Micro-Averaged AUC Precision score = {micro_auc_precision}\n")

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




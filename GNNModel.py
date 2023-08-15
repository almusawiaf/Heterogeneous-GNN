import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, average_precision_score
import torch.nn as nn

# Check if a GPU is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        output = self.conv2(x, edge_index)

        return output
    
    

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
    import matplotlib.pyplot as plt
    
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




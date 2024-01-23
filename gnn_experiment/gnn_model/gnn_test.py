import os
import json
import torch
import copy

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.nn import Linear, Dropout, LeakyReLU, MultiheadAttention
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, ASAPooling, global_max_pool, global_add_pool
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.utils import dropout_edge


all_preds = []
all_labels = []
final_acc = 0
number_threads = 0
data_path = 'data_conversion/pheme_saved_graph_format/pheme_graph_sentencetransformers_stance_cls'



#_____Define Model Architecture_____


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.conv1_p = SAGEConv(384, hidden_channels,aggr='mean') #384
        self.conv2_p = SAGEConv(hidden_channels, hidden_channels,aggr='mean')
        self.conv3_p = SAGEConv(hidden_channels, hidden_channels,aggr='mean')

        self.conv1_d = SAGEConv(384, hidden_channels,aggr='mean') #384
        self.conv2_d = SAGEConv(hidden_channels, hidden_channels,aggr='mean')
        self.conv3_d = SAGEConv(hidden_channels, hidden_channels,aggr='mean')

        self.conv1_s = SAGEConv(768, 32,aggr='mean')
        self.conv2_s = SAGEConv(32, 32,aggr='mean')
        
        #self.lstm = torch.nn.LSTM(input_size = hidden_channels, hidden_size=hidden_channels,bidirectional=True)
        self.attention_p = GATv2Conv(hidden_channels,hidden_channels,heads=1)
        self.attention_d = GATv2Conv(hidden_channels,hidden_channels,heads=1)
        self.attention_s = GATv2Conv(32,32,heads=3)

        self.attention = MultiheadAttention(2*hidden_channels+96+96,num_heads = 8)
        self.lin = Linear(2*hidden_channels+96+96, 3)

    
    def forward(self,x, edge_index,batch):
        # 1. Obtain node embeddings for propagation graph
        x1 = x[:,:384]
        x1.requires_grad_()
        edge_index, _ = dropout_edge(edge_index,p=0.1)
        x1 = self.conv1_p(x1,edge_index)
        x1 = x1.relu()
        x1 = self.conv2_p(x1,edge_index)
        x1 = x1.relu()
        x1 = self.conv3_p(x1,edge_index)
        x1 = self.attention_p(x1,edge_index)

        # 2. Obtain node embeddings for dispersion graph
        x2 = x[:,384:768]
        x2.requires_grad_()
        edge_index2 = torch.tensor([edge_index.tolist()[1],edge_index.tolist()[0]])
        x2 = self.conv1_d(x2,edge_index2)
        x2 = x2.relu()
        x2 = self.conv2_d(x2,edge_index2)
        x2 = x2.relu()
        x2 = self.conv3_d(x2,edge_index2)
        x2 = self.attention_d(x2,edge_index2)


        # Obtain embeddings for stance graph
        x3 = x[:,768:]
        x3.requires_grad_()
        x3 = self.conv1_s(x3,edge_index)
        x3 = LeakyReLU(0.1)(x3)
        x3 = self.conv2_s(x3,edge_index)
        x3 = self.attention_s(x3,edge_index)

        # 3. Concatenate stance + Readout layer
        x1 = torch.cat((x1,x3),1)
        x2 = torch.cat((x2,x3),1)
        
        x1 = global_max_pool(x1, batch)  # [batch_size, hidden_channels]
        x2 = global_max_pool(x2, batch)  # [batch_size, hidden_channels]

        # 4. Concatenate propagation and dispersion outputs
        x = torch.cat((x1,x2),1)
        

        # 5. Apply a final classifier
        x, _ = self.attention(x, x, x)
        x = self.lin(x)
        
        return x


# Create the edge index for a dispersion graph (the current graph is saved in propagation format)
def dispersion_conversion(edge_index):
    edge_index_dispersion = edge_index.tolist()
    edge_index_dispersion = [[x[1],x[0]] for x in edge_index_dispersion]
    edge_index_dispersion = torch.tensor(edge_index_dispersion)
    edge_index_dispersion = edge_index_dispersion.t().contiguous()
    
    return edge_index_dispersion




def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    #print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()



for fold in ['ferguson','sydneysiege','germanwings-crash','ottawashooting','charliehebdo']:
    print("Experiment for ", fold)
    #______Prepare Data Loaders______

    train_list = []
    test_list = []
    val_list = []

    for f in ['sydneysiege','germanwings-crash','ottawashooting','charliehebdo','ferguson']:
        #Load the train test
        if f!=fold:
            fold_path = os.path.join(data_path,f)
            for file in os.listdir(fold_path):
                if file[0]=='.':
                    continue
                data = torch.load(os.path.join(fold_path,file))
                data.edge_index = data.edge_index.t().contiguous()


                if len(data.edge_index.size()) == 1:
                    #print('problem!!!')
                    continue

                train_list.append(data)
        #Load the test set
        else:
            fold_path = os.path.join(data_path,fold)
            for file in os.listdir(fold_path):
                if file[0]=='.':
                    continue
                data = torch.load(os.path.join(fold_path,file))
                data.edge_index = data.edge_index.t().contiguous()

          
                if len(data.edge_index.size()) == 1:
                    #print('problem!!!')
                    continue
                test_list.append(data)   

    
    #Load validation step (designated as the 'charliehebdo' fold according to prior work due to its label balance)
    fold_path = os.path.join(data_path,'charliehebdo') 
    for file in os.listdir(fold_path):
        if file[0]=='.':
            continue
        data = torch.load(os.path.join(fold_path,file))
        data.edge_index = data.edge_index.t().contiguous()

        
        if len(data.edge_index.size()) == 1:
            #print('problem!!!')
            continue
        val_list.append(data)   



    print(len(train_list),len(test_list))

    train_loader = DataLoader(train_list, shuffle=True, batch_size=20)
    test_loader = DataLoader(test_list, batch_size=20)
    val_loader = DataLoader(val_list, batch_size=20)
    

    #____Load the saved model____
    model = GCN(hidden_channels=256)
    model_path = os.path.join('gnn_model/graph/best_results_stance',fold+'_saved_gnn')
    model.load_state_dict(torch.load(model_path))



    #_____Test the model_____

    test_preds = []
    test_labels = []

    model.eval()
    correct = 0
    for data in test_loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        test_preds.extend(pred.tolist())
        test_labels.extend(data.y.tolist())

    print(test_preds)
    print(test_labels)
    all_preds.extend(test_preds)
    all_labels.extend(test_labels)
    final_acc+=correct
    number_threads+=len(test_list)
    acc =  correct / len(test_list)  # Derive ratio of correct predictions.
    print('Test accuracy {}'.format(acc))
    print('F1 score ',f1_score(test_labels, test_preds, average='macro'))
   
    
    model.apply(reset_weights)


print('Macro F1 score ',f1_score(all_labels, all_preds, average='macro'))
print('Final accuracy:', final_acc/number_threads)
print('F1 score for each label ',f1_score(all_labels, all_preds, average = None))

import torch_geometric
import captum
import torch
import os
import json
import shutil

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.nn import Linear, Dropout, LeakyReLU, MultiheadAttention
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, ASAPooling, global_max_pool, global_add_pool
from torch_geometric.explain import Explainer, ExplainerAlgorithm
from torch_geometric.explain.algorithm import CaptumExplainer
from captum.attr import IntegratedGradients
from torch_geometric.utils import dropout_edge

from captum.attr import ShapleyValueSampling

from thread_tiers import tiers






####_____Define Model Architecture_____


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
        #x1 = copy.deepcopy(x)
        #x1 = torch.tensor([x1[i][:384].tolist() for i in range(x1.shape[0])])
        x1 = x[:,:384]
        #torch.tensor([x[i][:384].tolist() for i in range(x.shape[0])],requires_grad=True)
        x1.requires_grad_()
        edge_index, _ = dropout_edge(edge_index,p=0.1)
        x1 = self.conv1_p(x1,edge_index)
        x1 = x1.relu()
        x1 = self.conv2_p(x1,edge_index)
        x1 = x1.relu()
        x1 = self.conv3_p(x1,edge_index)
        x1 = self.attention_p(x1,edge_index)

        # 2. Obtain node embeddings for dispersion graph
        #x2 = copy.deepcopy(x)
        #x2 = torch.tensor([x2[i][:384].tolist() for i in range(x2.shape[0])])
        x2 = x[:,384:768]
        #torch.tensor([x[i][384:768].tolist() for i in range(x.shape[0])],requires_grad=True)
        x2.requires_grad_()
        edge_index2 = torch.tensor([edge_index.tolist()[1],edge_index.tolist()[0]])
        x2 = self.conv1_d(x2,edge_index2)
        x2 = x2.relu()
        x2 = self.conv2_d(x2,edge_index2)
        x2 = x2.relu()
        x2 = self.conv3_d(x2,edge_index2)
        x2 = self.attention_d(x2,edge_index2)


        # Obtain embeddings for stance graph
        #x3 = copy.deepcopy(x)
        #x3 = torch.tensor([x3[i][384:].tolist() for i in range(x3.shape[0])])
        x3 = x[:,768:]
        #torch.tensor([x[i][768:].tolist() for i in range(x.shape[0])],requires_grad=True)
        x3.requires_grad_()
        x3 = self.conv1_s(x3,edge_index)
        #x3 = x3.relu()
        x3 = LeakyReLU(0.1)(x3)
        x3 = self.conv2_s(x3,edge_index)
        #x3 = x3.relu()
        #x3 = self.conv3_s(x3,edge_index)
        x3 = self.attention_s(x3,edge_index)

        # 3. Readout layer
        x1 = torch.cat((x1,x3),1)
        x2 = torch.cat((x2,x3),1)
        
        
        x1 = global_max_pool(x1, batch)  # [batch_size, hidden_channels]
        x2 = global_max_pool(x2, batch)  # [batch_size, hidden_channels]

        # 4. Concatenate propagation and dispersion outputs
        x = torch.cat((x1,x2),1)
        

        # 5. Apply a final classifier
        x, _ = self.attention(x, x, x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x





# Create the edge index for a dispersion graph (the current graph is saved in propagation format)
def dispersion_conversion(edge_index):
    edge_index_dispersion = edge_index.tolist()
    edge_index_dispersion = [[x[1],x[0]] for x in edge_index_dispersion]
    edge_index_dispersion = torch.tensor(edge_index_dispersion)
    edge_index_dispersion = edge_index_dispersion.t().contiguous()
    
    return edge_index_dispersion



def model_forward(x,edge_index):
    batch = torch.zeros(data.x.shape[0], dtype=int)
    out = model(x, edge_index,batch)
    return out




#####__________Load the model_________

for fold in ['ferguson','charliehebdo','sydneysiege','germanwings-crash','ottawashooting']:

    model = GCN(hidden_channels=256)
    model_path = os.path.join('best_results_stance',fold+'_saved_gnn')
    model.load_state_dict(torch.load(model_path))




    ####_________Load the test data_________

    data_path = 'pheme_graph_sentencetransformers_stance_cls'
    hydrated_data_path = 'pheme_mapping'

    fold_path = os.path.join(data_path,fold)
    hydrated_fold_path = os.path.join(hydrated_data_path,fold)
    file_names = []

    test_list = []
    hydrated_test_list = []

    for file in os.listdir(fold_path):
        if file[0]=='.':
            continue

        path_to_save = os.path.join("reply_contributions_stance_shapley",fold, file+'.json')
        if os.path.isfile(path_to_save):
            continue

        data = torch.load(os.path.join(fold_path,file))
        data.edge_index2 = dispersion_conversion(data.edge_index)
        data.edge_index = data.edge_index.t().contiguous()
        
        if len(data.edge_index.size()) == 1:
            #print('problem!!!')
            continue
        file_names.append(file)
        test_list.append(data) 


    for file in file_names:
        if file[0]=='.':
            continue
        
        

        with open(os.path.join(hydrated_fold_path,file+'.json')) as f:
            data = json.load(f)
        hydrated_test_list.append(data)


    test_loader = DataLoader(test_list, batch_size=1)





    ####_____Generate explanation for graph instance____:


    for idx in range(len(test_list)):
        data = test_list[idx]
        hydrated_data = hydrated_test_list[idx]
        batch = torch.tensor([0]*data.x.shape[0])
        gold_label = data.y
        edgesA = list(list(data.edge_index)[0])
        edgesB = list(list(data.edge_index)[1])
        edges = [[edgesA[i].item(),edgesB[i].item()] for i in range(len(edgesA))]
        print(data.x.size())


        #Establish feature mask to minimise computational overhead (i.e. only compute scores over nodes, instead of emb positions of nodes which is unnecesarily detailed) 
        feature_mask = []
        number_nodes = data.x.size()[0]
        for i in range(number_nodes):
            feature_mask.append([i]*data.x.size()[1])
        feature_mask = torch.tensor(feature_mask)
        print(feature_mask)
        print(feature_mask.size())    


        #Set up the explainer infrastructure
        explainer = Explainer(
            model=model,
            algorithm=CaptumExplainer(attribution_method = captum.attr.ShapleyValueSampling,show_progress = True,n_samples=25,perturbations_per_eval=1,feature_mask = feature_mask),
            explanation_type='model',
            node_mask_type = 'attributes',
            edge_mask_type = None,
            model_config=dict(
                mode='multiclass_classification',
                task_level='graph',
                return_type='raw',  # Model returns log probabilities.
            ),
        )


        try:
            # Order nodes by attribution score
            explanation = explainer(data.x, data.edge_index,batch=batch, target = gold_label)
            attributions = list(explanation.node_mask)

            print(attributions)

            overall_attributions = [torch.mean(node).item() for node in attributions]
            overall_attributions = [(i,overall_attributions[i]) for i in range(len(overall_attributions))]
            overall_attributions = sorted(overall_attributions, key = lambda pair:pair[1],reverse = True)
            #important_nodes = [x[0] for x in overall_attributions if x[1]>0]

            # Get prediction of model
            out = explainer.get_prediction(data.x, data.edge_index, batch=batch)
            pred = out.argmax(dim=1)

            # Get node tiers
            source_index = tiers(hydrated_data)[0]
            direct_replies = tiers(hydrated_data)[1]
            lower_replies = tiers(hydrated_data)[2]

            #Save results with node attributions
            dict_to_save = dict()

            dict_to_save["thread_id"] = hydrated_data["thread_id"]
            dict_to_save["gold_label"] = gold_label.item()
            dict_to_save["predicted_label"] = pred.item()
            dict_to_save["tweet_attributions"] = []

            for x in overall_attributions:
                node = dict()
                node["index"] = x[0]
                node["attribution"] = x[1]


                for tweet_id in hydrated_data["node_mapping"].keys():
                    if hydrated_data["node_mapping"][tweet_id]==node["index"]:
                        node["tweet_id"] = int(tweet_id) 
                        break

                for instance in hydrated_data["text_thread"]:
                    if instance[0] == node["tweet_id"]:
                        node["text"] = instance[1]

                if node["index"] == source_index:
                    node["type"] = 'source'
                elif node["index"] in direct_replies:
                    node["type"] = 'direct_reply'
                else:
                    node["type"] = 'lower_reply'

                dict_to_save["tweet_attributions"].append(node) 

            path_to_save = os.path.join("reply_contributions_stance_shapley",fold,dict_to_save["thread_id"]+'.json')
            if not os.path.isfile(path_to_save):
                with open(path_to_save,'w') as g:
                    json.dump(dict_to_save,g)
            print('Saved results for ',fold,dict_to_save["thread_id"])         
        except:
            pass            
 
 
for fold in ['ferguson','charliehebdo','sydneysiege','germanwings-crash','ottawashooting']:
    fold_path = os.path.join('sampling/long_nonempty_threads',fold)
    for file in os.listdir(fold_path):
        file_path = os.path.join('reply_contributions_stance_shapley',fold,file)
        new_path = os.path.join('sampling/long_nonempty_threads_shapley',fold,file)
        shutil.copyfile(file_path,new_path)

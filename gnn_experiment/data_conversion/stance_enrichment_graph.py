import json
import os
import torch
import copy
import torch_geometric
import re

from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from transformers import AutoModel, AutoTokenizer 





# Open file with silver labels for PHEME stance 
with open('data_conversion/roberta_embedding/roberta_silver_labels_cls_token.json') as f:
    silver_labels = json.load(f)




def stance_enrich(data_path,silver_labels,logits = True):
    homo_data = torch.load(data_path)
    
    # Initialise a new homogenous graph with extended stance embeddings
    homo_data2 = Data()
    homo_data2.edge_index = homo_data.edge_index
    homo_data2.y = homo_data.y

    embs = homo_data.x.tolist()

    # Open graph mapping file
    data_mapping_path = data_path.replace('pheme_graph_sentencetransformers','pheme_mapping')+'.json'
    with open(data_mapping_path) as f:
        mapping = json.load(f)
  





    if logits == False:
    
        # Find stance values
        stance_values = []
        for tweet_id,index in mapping["node_mapping"].items():
            if tweet_id in silver_labels.keys():
                stance_values.append((index,silver_labels[tweet_id]))
            else:
                stance_values.append((index,1))   # if we cannot retrieve the silver label, then define the stance as 'comment' 
        stance_values.sort(key=lambda x:x[0])

        # Construct stance vectors through one-hot encoding
        stance_vectors = []
        for value in stance_values:
            vec = [0,0,0,0]
            vec[value[1]] = 1
            stance_vectors.append(vec)
    
    
    else:
        # Find stance vectors represented by logits
        stance_vectors = []
        for tweet_id,index in mapping["node_mapping"].items():
            if tweet_id in silver_labels.keys():
                stance_vectors.append((index,silver_labels[tweet_id]["last_layer"]))
            else:
                stance_vectors.append((index,[0,1,0,0]))   # if we cannot retrieve the silver label, then define the stance as 'comment' 
        stance_vectors.sort(key=lambda x:x[0])
        stance_vectors = [x[1] for x in stance_vectors]



    # Construct final node embeddings
    new_embs = []
    for i in range(len(embs)):  
        new_emb = embs[i] + embs[i] + stance_vectors[i]
        new_embs.append(new_emb)  
    homo_data2.x = torch.tensor(new_embs)

    path_to_save = data_path.replace('pheme_graph_sentencetransformers','pheme_graph_sentencetransformers_stance_cls')
    #if not os.path.isfile(path_to_save):
    torch.save(homo_data2,path_to_save)
    
    return homo_data2.x.shape









if __name__=='__main__':
    for fold in ['ferguson','charliehebdo','germanwings-crash','ottawashooting','sydneysiege']:
        fold_path = os.path.join('data_conversion/pheme_saved_graph_format/pheme_graph_sentencetransformers',fold)
        
        for file in os.listdir(fold_path):
            if file[0]=='.':
                continue
            file_path = os.path.join(fold_path,file)

            homo_data2 = stance_enrich(file_path,silver_labels,True)
            print(homo_data2)
            
            
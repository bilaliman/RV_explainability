import json
import os
import preprocessor as p
import torch
import copy
import torch_geometric
import re

from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from transformers import AutoModel, AutoTokenizer 

bertweet = AutoModel.from_pretrained("vinai/bertweet-large")

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large")




#Preprocess function for tweets
p.set_options(p.OPT.EMOJI, p.OPT.SMILEY)
def preprocess_tweet(tweet):
    
    tweet = tweet.lower()
    tweet = tweet.replace('\n','').replace('\t','')
    tweet = re.sub(r'http\S+', 'URL', tweet)
    tweet = re.sub(r'@\w+', 'MENTION', tweet)
    tweet = p.clean(tweet)
 
    return tweet



def emb_method(text,method ='all-MiniLM-L6-v2'):
    
    if method == 'all-MiniLM-L6-v2':
        model = SentenceTransformer('all-MiniLM-L6-v2') 
        emb = model.encode(preprocess_tweet(text))
    
    elif method == 'bertweet_pooler':
        input_ids = torch.tensor([tokenizer.encode(preprocess_tweet(text))])
        with torch.no_grad():
            features = bertweet(input_ids)  # Models outputs are now tuples

        emb = features.pooler_output[0].numpy()
    elif method == 'bertweet_mean':
        input_ids = torch.tensor([tokenizer.encode(preprocess_tweet(text))])
        with torch.no_grad():
            features = bertweet(input_ids)  # Models outputs are now tuples

        number_tokens  = features.last_hidden_state.shape[1]
        emb = features.last_hidden_state.reshape([number_tokens,1024])
        emb = torch.mean(emb,dim = 0).numpy()

    else:
        input_ids = torch.tensor([tokenizer.encode(preprocess_tweet(text))])
        with torch.no_grad():
            features = bertweet(input_ids)  # Models outputs are now tuples
        emb = features.last_hidden_state[0][0].numpy()   
    
    return emb
    



def extract_text(thread_path):
    
    tweets_list = []

    #Add text of source tweet to list
    thread_id = thread_path.split('/')[-1]
    path_to_source = os.path.join(thread_path,'source-tweets',thread_id + '.json')
    with open(path_to_source) as f:
        source = json.load(f)

    tweets_list.append((source['id'],source['text']))    

    #Add text of remaining thread to list
    path_to_replies = os.path.join(thread_path,'reactions')
    for file in os.listdir(path_to_replies):
        #Ignore some useless files
        if file[0]=='.' or file in [t[0] for t in tweets_list]:
            continue

        with open(os.path.join(path_to_replies,file)) as f:
            reply = json.load(f) 
 
        tweets_list.append((reply['id'],reply['text']))

    return tweets_list         




def extract_text2(thread_path):
    
    tweets_list = []

    #Add text of source tweet to list
    thread_id = thread_path.split('/')[-1]
    path_to_source = os.path.join(thread_path,'source-tweets',thread_id + '.json')
    with open(path_to_source) as f:
        source = json.load(f)

    tweets_list.append((source['id'],source['text']))    

    #Add text of remaining thread to list
    path_to_replies = os.path.join(thread_path,'reactions')
    for file in os.listdir(path_to_replies):
        #Ignore some useless files
        if file[0]=='.' or file in [t[0] for t in tweets_list]:
            continue

        with open(os.path.join(path_to_replies,file)) as f:
            reply = json.load(f) 
        
        if reply['id'] not in [t[0] for t in tweets_list]:
            tweets_list.append((reply['id'],reply['text']))

    return tweets_list 




class Edges:
    def __init__(self,edge_set):
        self.edge_set = edge_set


    def walk_thread(self,structure,node):
        if structure[node] == []:
            self.edge_set.extend([])
        else:
            batch_edges = []
            for node2 in structure[node].keys():
                # Add edges corresponding to that node
                batch_edges.append([node,node2])
                
            self.edge_set.extend(batch_edges)
            for node2 in structure[node].keys():
                self.walk_thread(structure[node],node2)




def extract_hierarchy(thread_path):
    #Note that the graph is directed 

    path_to_struct = os.path.join(thread_path,'structure.json')
    thread_id = thread_path.split('/')[-1]
    with open(path_to_struct) as f:
        structure = json.load(f)
    
    # If the source id is accounted in the structure
    if thread_id in structure.keys():    
        edge_instance = Edges([])
        edge_instance.walk_thread(structure,thread_id)
        k = 0
    # If the source id is not accounted in the structure
    else:
        edges = []
        for key in structure.keys():
            edges.append([thread_id,key])

        edge_instance = Edges(edges)    
        for key in structure.keys():
            edge_instance.walk_thread(structure,key)
        k = 1    

    return edge_instance.edge_set, k               
        
            

def convert_annotations(thread_path, string=False):
    
    annotation_path = os.path.join(thread_path,'annotation.json')
    with open(annotation_path) as f:
        annotation = json.load(f)


    if 'misinformation' in annotation.keys() and 'true'in annotation.keys():
        if (int(annotation['misinformation']) == 0) and (
                            int(annotation['true']) == 0):
            if string:
                label = "unverified"
            else:
                label = 2
        elif (int(annotation['misinformation']) == 0) and (
                              int(annotation['true']) == 1):
            if string:
                label = "true"
            else:
                label = 0
        elif (int(annotation['misinformation']) == 1) and (
                              int(annotation['true']) == 0):
            if string:
                label = "false"
            else:
                label = 1
        elif (int(annotation['misinformation']) == 1) and (
                              int(annotation['true']) == 1):
            print ("OMG! They both are 1!")
            print(annotation['misinformation'])
            print(annotation['true'])
            label = None
    elif ('misinformation' in annotation.keys()) and (
                      'true' not in annotation.keys()):
        if int(annotation['misinformation']) == 0:
            if string:
                label = "unverified"
            else:
                label = 2
        elif int(annotation['misinformation']) == 1:
            if string:
                label = "false"
            else:
                label = 1

    elif ('true' in annotation.keys()) and (
          'misinformation' not in annotation.keys()):
        print ('Has true not misinformation')
        label = None
    else:
        print('No annotations')
        label = None
    return label



def graph_conversion(thread_path,embedding):     
    
    data = Data()

    hierarchy_thread, _ = extract_hierarchy(thread_path) 
    text_thread = extract_text2(thread_path)
    print('Text thread is unique: ',len(text_thread)==len(set(text_thread)))
    mapped_edges = []

    #Create mapping of nodes replacing the tweet ids
    mapping = {}
    for i,tweet in enumerate(text_thread):
        tweet_id = tweet[0]
        tweet_text = tweet[1]
        
        mapping[str(tweet_id)] = i

    #Convert edges using the mapped indices 
    for edge in hierarchy_thread:
        node1 = mapping[edge[0]]
        node2 = mapping[edge[1]]

        mapped_edges.append([node1,node2])

    data.x = torch.tensor([emb_method(tweet[1],method = embedding) for tweet in text_thread])
    data.edge_index = torch.tensor(mapped_edges)
    data.y = torch.tensor(convert_annotations(thread_path))
    
    #Save the graph data
    thread_id = thread_path.split('/')[-1]
    fold = thread_path.split('/')[-3]
    path_to_save = os.path.join('pheme_graph_'+ embedding,fold,thread_id)
    
    #if not os.path.isfile(path_to_save):
    torch.save(data,path_to_save)



    #Save the text data with node mapping included for future experiments
    cluster = {}
    cluster['thread_id'] = thread_id
    cluster['text_thread'] = text_thread
    cluster['node_mapping'] = mapping
    cluster['edges'] = mapped_edges
    path_to_save2 = os.path.join('pheme_mapping',fold,thread_id +'.json')

    if not os.path.isfile(path_to_save2):
        with open(path_to_save2,'w') as g:
            json.dump(cluster,g)


    return fold, thread_id, text_thread,mapping, data.edge_index




if __name__=='__main__':
 
    # Desired embedding type
    embedding = 'bertweet_cls'

    # Input path of original pheme data
    original_pheme_data = 'pheme'

    for fold in ['ferguson','charliehebdo','germanwings-crash','ottawashooting','sydneysiege']:
        # Input path of original
        fold_path = os.path.join(original_pheme_data,fold,'rumours')
        for folder in os.listdir(fold_path):
            if folder[0]=='.':
                continue

            with open(os.path.join(fold_path,folder,'structure.json')) as f:
                data = json.load(f)
            if data == []:
                continue    
            
            thread_path = os.path.join(fold_path,folder)
            thread_id = thread_path.split('/')[-1]
            fold = thread_path.split('/')[-3]
            path_to_save = os.path.join('data_conversion','pheme_saved_graph_format','pheme_graph_' + embedding,fold,thread_id)
            if os.path.isfile(path_to_save):
                continue
      
            print(graph_conversion(thread_path,embedding))


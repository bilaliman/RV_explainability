import torch
import json
import os
import re

from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
import preprocessor as p
torch.manual_seed(42)


def tokenize_function(example):   
    return tokenizer(example["text"], truncation=True,max_length=512)

#Preprocess function for tweets
p.set_options(p.OPT.EMOJI, p.OPT.SMILEY)
def preprocess_tweet(tweet):
    
    tweet = tweet.lower()
    tweet = tweet.replace('\n','').replace('\t','')
    tweet = re.sub(r'http\S+', 'URL', tweet)
    tweet = re.sub(r'@\w+', 'MENTION', tweet)
    tweet = p.clean(tweet)
 
    return tweet




# Load the checkpoint
checkpoint = "cardiffnlp/twitter-roberta-base" 
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=4)
model.load_state_dict(torch.load("data_conversion/roberta_embedding/roberta_base_alldata",map_location=torch.device('cpu')))
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)



# Load the data
silver_labels = {}
for fold in ['ferguson','charliehebdo','germanwings-crash','ottawashooting','sydneysiege']:
    counter = 0 
    fold_path = os.path.join('data_conversion/pheme_mapping',fold)
    for file in os.listdir(fold_path):
        if file[0]=='.':
            continue
        with open(os.path.join(fold_path,file)) as f:
            thread = json.load(f)
        
        for post in thread["text_thread"]:
            if str(post[0])==thread["thread_id"]:
                source_text = post[1]
                break

        for post in thread["text_thread"]:
            concatenation = preprocess_tweet(source_text) + ' </s> ' + preprocess_tweet(post[1])
            idx = str(post[0])
            tokenized_example = tokenizer(concatenation, truncation=True,max_length=512,return_tensors='pt')

            # Run the model
            model.eval()
            with torch.no_grad():
                outputs = model(**tokenized_example,output_hidden_states=True)
            
            
            #last_layer = torch.mean(outputs.hidden_states[-1][0],dim=0).tolist()  #mean of token representations
            last_layer = outputs.hidden_states[-1][0][0].tolist() # representation of </s> token (equivalent of CLS token)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()
            logits = logits.tolist()[0]
            silver_labels[idx] = {"text":concatenation,"logits":logits,"label":prediction,"last_layer":last_layer}
            print(fold, counter)
            counter += 1
    
    #Save results
    with open('data_conversion/roberta_emebdding/roberta_silver_labels_cls_token.json','w') as g:
        json.dump(silver_labels,g)       



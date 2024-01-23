# Part 1: Convert microblog threads, Construct GNN and Explain GNN using rationales
Explainability for Rumour Verification
Due to file constraints, download the entire project with saved data structures from [here](https://drive.google.com/file/d/1QVA6bfunNu-dx-cd-Qkbcx-0NErWhD7s/view?usp=drive_link).


#1. Data Conversion

## 1.1 Transform pheme threads into graphs 
Use script: gnn_experiment/data_conversion/graph_conversion.py
Can choose embedding type: BERTweet, Sentence Transformers with options for pooled, mean and cls output
Saved graph outputs in folder: gnn_experiment/data_conversion/pheme_saved_graph_format

## 1.2 Enrich graphs with Stance embeddings
Use script: gnn_experiment/data_conversion/roberta_stance_vectorization.py
Silver stance RoBERTa embeddings for each tweet: gnn_experiment/data_conversion/roberta_embedding/roberta_silver_labels_cls_token.json
Trained Stance RoBERTa: gnn_experiment/data_conversion/roberta_embedding/roberta_base_alldata
Saved stance-enriched graphs: gnn_experiment/data_conversion/pheme_saved_graph_format/pheme_graph_sentencetransformers_stance_cls (used in the final experiment)

## 1.3 Additional info about graph structures
Use data: gnn_experiment/data_conversion/pheme_mapping
Contains bijections between graph nodes and tweet ids 


# 2. GNN Model construction

## 2.1. Training
Use script: gnn_experiment/gnn_model/gnn_training.py
Saved checkpoints: gnn_experiment/gnn_model/best_results_stance.zip
Model contains components: propagation, dispersion and stance

## 2.2 Testing
Use script: gnn_experiment/gnn_model/gnn_test.py


# 3. Explain GNN Model
Use script: gnn_experiment/explain_model/explain_gnn_stance_shapley.py
It uses the algorithm Shapley values, but this can be substituted with other captum-enabled explainers such as Integrated Gradients

Output data: gnn_experiment/explain_model/reply_contributions_stance_shapley
Contains the threads such that their posts are ordered by the importance score (via Shapley Values) to the rumour verification model

Output data: gnn_experiment/explain_model/reply_contributions_stance_integrated_gradients
Contains the threads such that their posts are ordered by the importance score (via Integrated Gradients) to the rumour verification model


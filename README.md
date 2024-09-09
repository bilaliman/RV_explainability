[Corresponding paper](https://arxiv.org/pdf/2401.12713): "Generating Zero-shot Abstractive Explanations for Rumour Verification". Iman Munire Bilal and Preslav Nakov and Rob Procter and Maria Liakata. 2024

# Explainability for Rumour Verification
Due to file constraints, download the entire project with saved data structures from [here](https://drive.google.com/drive/folders/1fncXMlHcde2yA4uNJNo84X_28PCSZqGW?usp=drive_link).

## 1. Data Conversion

### 1.1 Transform pheme threads into graphs 
Use script: `gnn_experiment/data_conversion/graph_conversion.py`

Can choose embedding type: BERTweet, Sentence Transformers with options for pooled, mean and cls output

Saved graph outputs in folder: `gnn_experiment/data_conversion/pheme_saved_graph_format`

### 1.2 Enrich graphs with Stance embeddings
Use script: `gnn_experiment/data_conversion/roberta_embedding/roberta_stance_vectorization.py`

Silver stance RoBERTa embeddings for each tweet: `gnn_experiment/data_conversion/roberta_embedding/roberta_silver_labels_cls_token.json` (to be uploaded later due to size limit)

Trained Stance RoBERTa: `gnn_experiment/data_conversion/roberta_embedding/roberta_base_alldata` (to be uploaded later due to size limit)

Saved stance-enriched graphs: `gnn_experiment/data_conversion/pheme_saved_graph_format/pheme_graph_sentencetransformers_stance_cls` (used in the final experiment)

### 1.3 Additional info about graph structures
Use data: `gnn_experiment/data_conversion/pheme_mapping`

Contains bijections between graph nodes and tweet ids 



## 2. GNN Model construction

### 2.1. Training
Use script: `gnn_experiment/gnn_model/gnn_training.py`

Saved checkpoints: `gnn_experiment/gnn_model/best_results_stance.zip`

Model contains components: propagation, dispersion and stance

### 2.2 Testing
Use script: `gnn_experiment/gnn_model/gnn_test.py`



## 3. Explain GNN Model
Use script: `explain_model/explain_gnn_stance_shapley.py`

It uses the algorithm Shapley values, but this can be substituted with other captum-enabled explainers such as Integrated Gradients

Output data: `explain_model/reply_contributions_stance_shapley`

Contains the threads such that their posts are ordered by the importance score (via Shapley Values) to the rumour verification model

Output data: `explain_model/reply_contributions_stance_integrated_gradients`

Contains the threads such that their posts are ordered by the importance score (via Integrated Gradients) to the rumour verification model



## 4. Summarise the sample of posts ranked in the previous steps by the post-hoc explainers

Use script: `summarisation/bart_summary.py`

Pre-trained model and tokenizer: `summarisation/BART_model_opi` (to be uploaded later due to size limit)

Use data: `summarisation/long_nonempty_threads`

Saved explanations: `summarisation/explanations_integrated` (Integrated Gradients explanations)
		                `summarisation/explanations_shapley` (Shapley Values explanations)


## 5. LLM-based Evaluation

### 5.1. Find and adjust prompt if necessary
File: `llm_eval/prompt_informativeness.txt`

### 5.2. Run your LLM of choice to evaluate the informativeness of the samples
Use script: `llm_eval/chatgpt_informativeness.py`

Models tested: ChatGPT 3.5 turbo 0301, ChatGPT 3.5 turbo 0613 and GPT-4 0613


import torch, dgl, accelerate
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import set_seed
from dgl.distributed import partition_graph
import time
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import txgnn.TxGNN as TxGNN, txgnn.TxData as TxData, txgnn.TxEval as TxEval
import pickle
import random

file_path = 'psuedo_scores_all_indication.pkl'
with open(file_path, 'rb') as file:
    indication_results = pickle.load(file)
# indication_results
    
    file_path = 'psuedo_scores_all_contraindication.pkl'
with open(file_path, 'rb') as file:
    contraindication_results = pickle.load(file)
# contraindication_results
    
## load seed 1 train, val, test splits
df_train = pd.read_csv('data/complex_disease_1/train.csv', low_memory=False)
df_valid = pd.read_csv('data/complex_disease_1/valid.csv', low_memory=False)
df_test = pd.read_csv('data/complex_disease_1/test.csv', low_memory=False)
dd_df_train = df_train[(df_train.y_type == 'disease') & (df_train.x_type == 'drug')]
dd_df_valid = df_valid[(df_valid.y_type == 'disease') & (df_valid.x_type == 'drug')]
dd_df_test = df_test[(df_test.y_type == 'disease') & (df_test.x_type == 'drug')]
dd_df_all = pd.concat([dd_df_train, dd_df_valid, dd_df_test])

def turn_to_df(results, rel, k):
    # additional_train_dict = []
    # concat_additional_train_df = []
    count = 0
    nested_concat_dfs = {"train": [], "valid": [], "test": []}
    for (dis_id, drug_ids), drug_idxs, dis_idx, ranked_scores in zip(results['ranked_drug_ids'].items(), results['ranked_drug_idxs'].values(), results['dis_idx'].values(), results['ranked_scores'].values()):
        new_dicts = [{'y_id': dis_id, 'y_idx': dis_idx, 'x_id': drug_id, 'x_idx': drug_idx, 'relation': rel, 'score': ranked_score} for i, (drug_id, drug_idx, ranked_score) in enumerate(zip(drug_ids, drug_idxs, ranked_scores))]
        
        # ## top and bottom k approach
        # top_k = new_dicts[:k]
        # bottom_k = new_dicts[-k:]
        # additional_train_dict += top_k + bottom_k

        ## Random Sampling
        # concat_additional_train_df.append(pd.DataFrame(random.sample(new_dicts, k)))

        ## in df_train, df_valid, or df_
        temp_df = pd.DataFrame(new_dicts)
        temp_train = temp_df.merge(dd_df_train[['x_id', 'y_id', 'relation']], on=['x_id', 'y_id', 'relation'], how="inner")
        temp_valid = temp_df.merge(dd_df_valid[['x_id', 'y_id', 'relation']], on=['x_id', 'y_id', 'relation'], how="inner")
        temp_test = temp_df.merge(dd_df_test[['x_id', 'y_id', 'relation']], on=['x_id', 'y_id', 'relation'], how="inner")
        # temp_df = temp_df.merge(dd_df_train[['x_id', 'y_id', 'relation']], on=['x_id', 'y_id', 'relation'], how="inner")
        nested_concat_dfs["train"].append(temp_train)
        nested_concat_dfs["valid"].append(temp_valid)
        nested_concat_dfs["test"].append(temp_test)
        # count += 1
        # if count == 100:
        #     break
        # if len(temp_train) != 0:
        #     break

    for k, v in nested_concat_dfs.items():
        df = pd.concat(v)
        print(f"before dropping dup: {len(df)}")
        df = df.drop_duplicates()
        print(f"after dropping dup: {len(df)}")
        df["x_idx"] = df["x_idx"].astype(float)
        df["y_type"] = "disease"
        df["x_type"] = "drug"
        nested_concat_dfs[k] = df
    return nested_concat_dfs

k = 100
indication_dfs = turn_to_df(indication_results, "indication", k)
contraindication_dfs = turn_to_df(contraindication_results, "contraindication", k)

full_additional_train = pd.concat([indication_dfs['train'], contraindication_dfs['train']])
full_additional_valid = pd.concat([indication_dfs['valid'], contraindication_dfs['valid']])
full_additional_test = pd.concat([indication_dfs['test'], contraindication_dfs['test']])

full_additional_train.to_csv("toy_pseudo_scores_train", index=False)
full_additional_valid.to_csv("toy_pseudo_scores_valid", index=False)
full_additional_test.to_csv("toy_pseudo_scores_test", index=False)
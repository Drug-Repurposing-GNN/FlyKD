import torch, dgl, accelerate
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import set_seed
from dgl.distributed import partition_graph
import numpy as np

import pandas as pd
kg = pd.read_csv('data/kg.csv')
# kg = pd.read_csv('../../PrimeKG/datasets/data/kg/auxillary/data/kg_giant.csv')
kg

## extract all disease's ids
diseases1= kg[kg['x_type'] == 'disease']['x_id']
diseases2 = kg[kg['y_type'] == 'disease']['y_id']
disease_ids = pd.concat([diseases1, diseases2]).unique()
len(disease_ids)

## obtain all diseases' degree from disease-drug relation only
disease_drug1 = kg[(kg['x_type'] == 'disease') & (kg['y_type'] == 'drug')]['x_id']
disease_drug2 = kg[(kg['x_type'] == 'drug') & (kg['y_type'] == 'disease')]['y_id']
disease_drug_value_counts = pd.concat([disease_drug1, disease_drug2]).value_counts()
disease_drug_degree = disease_drug_value_counts.reindex(disease_ids).fillna(0).astype(int)
disease_drug_degree.sum()

# pd.concat([disease_drug1, disease_drug2])
# kg[kg['x_type'] == 'disease'].iloc[:20]
# kg[kg['x_name'] == 'osteogenesis imperfecta']
# # kg[kg['x_name'] == 'osteogenesis imperfecta']['x_id'].iloc[0]
# kg[kg['x_source'] == 'MONDO_grouped']
kg[kg['x_id'] == '13924']

# %% [markdown]
# ### Below here are just experimental. 
# 
# Plan:
# 1. Obtain predictions on relation of interest (for now: Indication)
# 2. See if you can pass that to the training dataset and train on it (Make sure the psuedo-labels do not leak to validation or test data)

# %%
drug_ids_x = kg[kg['x_type'] == 'drug']['x_id']
drug_ids_y = kg[kg['y_type'] == 'drug']['y_id']
drug_ids_value_count = pd.concat([drug_ids_x, drug_ids_y]).value_counts()
drug_ids_value_count
# drug_ids = pd.DataFrame({"x_idx": drug_ids})
# drug_ids

# %%
# gen_diseases = disease_drug_degree[disease_drug_degree < 1]
# gen_diseases
# drug_ids['relation'] = 'indication'
# drug_ids
# test_diseases = pd.DataFrame({'y_idx': ['400000', '400002']})
# test_diseases['relation'] = 'indication'
# test_diseases
# gen_df = drug_ids.merge(test_diseases, on='relation', how='left')
# gen_df
# def convert2str(x):
#     try:
#         if '_' in str(x): 
#             pass
#         else:
#             x = float(x)
#     except:
#         pass

#     return str(x)

disease_drug_degree.index.values
low_disease = disease_drug_degree[disease_drug_degree < 1]

# %%
from txgnn import TxData, TxGNN, TxEval

saving_path = './saved_models/'
# split = 'random'
split = 'complex_disease'
# split = 'cell_proliferation'
# split = 'mental_health'
# split = 'cardiovascular'
# split = 'anemia'
# split = 'adrenal_gland'
print(split)

additional_train = [{'x_type':'gene/protein', 'x_id': '9796.0',	'relation':'protein_protein',	'y_type':'gene/protein',	'y_id':'56992.0',	'x_idx':27422.0,	'y_idx':19536.0,},
         {'x_type':'gene/protein', 'x_id': '9796.0',	'relation':'protein_protein',	'y_type':'gene/protein',	'y_id':'56992.0',	'x_idx':27609.0,	'y_idx':19536.0,},
         {'x_type':'gene/protein', 'x_id': '9796.0',	'relation':'protein_protein',	'y_type':'gene/protein',	'y_id':'56342.0',	'x_idx':27609.0,	'y_idx':19536.0,},
         {'x_type':'gene/protein', 'x_id': '9796.0',	'relation':'protein_protein',	'y_type':'gene/protein',	'y_id':'24234.0',	'x_idx':24609.0,	'y_idx':22222.0,},
         {'x_type':'gene/protein', 'x_id': '9796.0',	'relation':'protein_protein',	'y_type':'gene/protein',	'y_id':'324343.0',	'x_idx':11111.0,	'y_idx':19536.0,}]
create_psuedo_edges = True

# very_strt = time.time()
TxData1 = TxData(data_folder_path = './data/')
TxData1.prepare_split(split = split, seed = 1, no_kg = False, additional_train=additional_train, create_psuedo_edges=create_psuedo_edges)

# %%
TxGNN1 = TxGNN(
        data = TxData1, 
        weight_bias_track = False,
        proj_name = 'TxGNN',
        exp_name = 'TxGNN'
    )

# %%
TxGNN1.model_initialize(n_hid = 100, 
                      n_inp = 100, 
                      n_out = 100, 
                      proto = True,
                      proto_num = 3,
                      attention = False,
                      sim_measure = 'all_nodes_profile',
                      bert_measure = 'disease_name',
                      agg_measure = 'rarity',
                      num_walks = 200,
                      walk_mode = 'bit',
                      path_length = 2)

print('here we go!')
TxGNN.pretrain(n_epoch = 2, ## was 2
               learning_rate = 1e-3,
               batch_size = 1024, 
               train_print_per_n = 20)
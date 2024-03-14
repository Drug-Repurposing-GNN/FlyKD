# import torch, dgl, accelerate
from txgnn import TxData, TxGNN, TxEval
# import torch.nn as nn
# import torch.nn.functional as F
# from accelerate.utils import set_seed
# from dgl.distributed import partition_graph
import numpy as np
import time
import pandas as pd
import pickle
import os
import sys
import pprint
import random
# saving_path = './saved_models/'
# split = 'random'
# split = 'complex_disease'
# split = 'cell_proliferation'
# split = 'mental_health'
# split = 'cardiovascular'
# split = 'anemia'
# split = 'adrenal_gland'
# print(split)

'''
    Let's first try one iteration to increase performance.
'''

def obtain_disease_idx(TxData1, deg):
    '''
        returns the disease idx that have less than k degrees (drug-disease relation)
    '''
    ## extract all disease's ids
    kg = pd.read_csv('data/kg.csv')
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

    ## length of ID
    # drug_ids_x = kg[kg['x_type'] == 'drug']['x_id']
    # drug_ids_y = kg[kg['y_type'] == 'drug']['y_id']
    # drug_ids_value_count = pd.concat([drug_ids_x, drug_ids_y]).value_counts()
    # drug_ids_value_count

    low_disease = disease_drug_degree[disease_drug_degree < deg]

    id_mapping = TxData1.retrieve_id_mapping()
    id2idx = {id:idx for idx, id in id_mapping['idx2id_disease'].items()}
    print(f"Total number of diseases?: {len(id2idx)}")
    print(f"total number of {deg} > degree diseases?: {len(low_disease)}")
    low_disease_idx = low_disease.index.map(lambda x: id2idx[x] if '_' in x else id2idx[x+'.0'])#.apply(lambda x: id2idx[x])
    low_disease_idx = np.array(low_disease_idx)

    return low_disease_idx

def obtain_high_degree_disease_id_w_df(pseudo_train, seed, deg):
    '''
        return pseudo_train with at least one degree in df_train
    '''
    df_train = pd.read_csv(f'data/complex_disease_{seed}/train.csv')
    ## extract all disease's ids
    diseases1= df_train[df_train['x_type'] == 'disease']['x_id']
    diseases2 = df_train[df_train['y_type'] == 'disease']['y_id']
    disease_ids = pd.concat([diseases1, diseases2]).unique()
    len(disease_ids)

    ## obtain all diseases' degree from disease-drug relation only
    disease_drug1 = df_train[(df_train['x_type'] == 'disease') & (df_train['y_type'] == 'drug')]['x_id']
    disease_drug2 = df_train[(df_train['x_type'] == 'drug') & (df_train['y_type'] == 'disease')]['y_id']
    drugs1 = df_train[(df_train['x_type'] == 'disease') & (df_train['y_type'] == 'drug')]['y_id']
    drugs2 = df_train[(df_train['x_type'] == 'drug') & (df_train['y_type'] == 'disease')]['x_id']
    drugs_value_counts = pd.concat([drugs1, drugs2]).value_counts()
    disease_drug_value_counts = pd.concat([disease_drug1, disease_drug2]).value_counts()
    disease_drug_degree = disease_drug_value_counts.reindex(disease_ids).fillna(0).astype(int)
    disease_drug_degree.sum()

    ## length of ID
    # drug_ids_x = kg[kg['x_type'] == 'drug']['x_id']
    # drug_ids_y = kg[kg['y_type'] == 'drug']['y_id']
    # drug_ids_value_count = pd.concat([drug_ids_x, drug_ids_y]).value_counts()
    # drug_ids_value_count

    # disease_drug_degree.index.values
    high_disease_series = disease_drug_degree[disease_drug_degree >= deg]
    high_drug_series = drugs_value_counts[drugs_value_counts >= deg]
    high_disease_set = set(high_disease_series.index)
    high_drug_set = set(high_drug_series.index)
    filtered_pseudo_train = pseudo_train[(pseudo_train.y_id.isin(high_disease_set)) & (pseudo_train.x_id.isin(high_drug_set))]
    return filtered_pseudo_train


# def turn_into_dataframe(results, t, least_score):
#     '''
#         t = number of psuedo_labels to be generated for low_diseases
#         Takes in the results eval file and returns the disease idx that have less than k degrees (drug-disease relation)
#     '''
#     additional_train_dict = []
#     for rel, result in results.items():
#         for (dis_id, drug_ids), drug_idxs, dis_idx, ranked_scores in zip(result['ranked_drug_ids'].items(), result['ranked_drug_idxs'].values(), result['dis_idx'].values(), result['ranked_scores'].values()):
#             if least_score is None:
#                 new_dicts = [{'y_id': dis_id, 'y_idx': dis_idx, 'x_id': drug_id, 'x_idx': drug_idx, 'relation': rel, 'score': ranked_scores[i]} for i, (drug_id, drug_idx) in enumerate(zip(drug_ids, drug_idxs)) if i < t]
#             else:
#                 new_dicts = [{'y_id': dis_id, 'y_idx': dis_idx, 'x_id': drug_id, 'x_idx': drug_idx, 'relation': rel, 'score': ranked_scores[i]} for i, (drug_id, drug_idx, ranked_score) in enumerate(zip(drug_ids, drug_idxs, ranked_scores)) if ranked_score > least_score]
#             additional_train_dict += new_dicts

#     df = pd.DataFrame(additional_train_dict)
#     df["x_idx"] = df["x_idx"].astype(float)
#     df["y_type"] = "disease"
#     df["x_type"] = "drug"
#     return df

def _turn_into_df_helper(result, rel, dd_df, random_k):
    concat_df_dd = []
    for (dis_id, drug_ids), drug_idxs, dis_idx, ranked_scores in zip(result['ranked_drug_ids'].items(), result['ranked_drug_idxs'].values(), result['dis_idx'].values(), result['ranked_scores'].values()):
        new_dicts = [{'y_id': dis_id, 'y_idx': dis_idx, 'x_id': drug_id, 'x_idx': drug_idx, 'relation': rel, 'score': ranked_score} for i, (drug_id, drug_idx, ranked_score) in enumerate(zip(drug_ids, drug_idxs, ranked_scores))]
        ## generating random k pseudo labels on non-existing relations
        if random_k is not None:
            concat_df_dd.append(pd.DataFrame(random.sample(new_dicts, random_k)))
        ## generate pseudo labels on existing relations
        temp_df = pd.DataFrame(new_dicts)
        temp_dd_df = temp_df.merge(dd_df[['x_id', 'y_id', 'relation']], on=['x_id', 'y_id', 'relation'], how="inner")
        concat_df_dd.append(temp_dd_df)
    df = pd.concat(concat_df_dd)
    print(f"before dropping dup: {len(df)}")
    df = df.drop_duplicates()
    print(f"after dropping dup: {len(df)}")
    df["x_idx"] = df["x_idx"].astype(float)
    df["y_type"] = "disease"
    df["x_type"] = "drug"
    return df

def turn_into_df(results, random_k=None):
    dd_df = pd.read_csv('data/og_df_dd.csv')
    concat_df = []
    for rel, result in results.items():
        concat_df.append(_turn_into_df_helper(result, rel, dd_df, random_k))
    pseudo_df = pd.concat(concat_df)
    # pseudo_df.to_csv(os.path.join(save_dir, "pseudo_df.csv"), index=False)
    return pseudo_df

def init_logfile(i, seed, args):
    '''
        create and set logfile to be written. Also write init messages such as args and seed
    '''
    # if 'force' in args.fname:
    #     save_dir = './' + args.fname + '/'
    # else:
    save_dir = f'./logs/eval_perf/{args.fname}/{seed}/'
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # else:
    #     save_dir = './logs/'
    log_file = open(save_dir + f"{i}.txt", 'w', buffering=1)
    sys.stderr = log_file
    sys.stdout = log_file
    print("Arguments received:")
    pprint.pprint(vars(args))
    print("------------------------------")
    print(f"Using seed: {seed}")
    return save_dir, log_file
    
# def generate_and_save_pseudo_labels(pre_trained_dir, save_dir, size, seed, args):
#     '''
#         with open(args.psuedo_label_fname[:-4] + f'_{k}.pkl', 'wb') as f:
#         first size prolly needs to be 100 then whatever size afterwards
#         also want to set pre_trained dir to 'properly_pre_trained_model_ckpt/seed_1_restrained_saveG' and then wherever the newly trained model is.

#         use generate_inog to generate pseudo labels much faster
#     '''
#     print(f"Generating pseudo labels with pre-trained model. Generate only on existing edges is set to {args.generate_inog}")
#     if args.generate_indication:
#         mode = "indication"
#     elif args.generate_contraindication:
#         mode = "contraindication"
#     else:
#         mode = "both"

#     # if args.train_then_generate:
#     #     ## used for training then generating psuedo labels to avoid save-loading error
#     #     print(f"Training THEN generate without save/loading the model: {args.psuedo_label_fname}")
#     #     pseudo_labels = train_then_generate_psuedo_labels(split=args.split, size=100, seed=seed, deg=args.deg, k_top_candidates=args.k_top_candidates, least_score=args.least_score, mode=mode)
#     #     # pseudo_labels.to_csv(args.psuedo_label_fname, index=False)
#     # else:
#     ## only for generating psuedo labels using pre-trained model
#     pseudo_labels = generate_pseudo_labels(pre_trained_dir=pre_trained_dir, 
#                                             split=args.split, 
#                                             size=size, 
#                                             seed=seed, 
#                                             deg=args.deg, 
#                                             k_top_candidates=args.k_top_candidates, 
#                                             least_score=args.least_score,
#                                             mode=mode,
#                                             generate_inog=args.generate_inog)
#     for k, p_labels in pseudo_labels.items():
#         if p_labels is not None:
#             with open(save_dir + f'_{k}.pkl', 'wb') as f:
#                 pickle.dump(p_labels, f, protocol=pickle.HIGHEST_PROTOCOL)
#             file_size = os.path.getsize(save_dir + f'_{k}.pkl')
#             print(file_size)

# def generate_pseudo_labels(pre_trained_dir, size, seed, mode=None):
def generate_pseudo_labels(pre_trained_dir, size, seed, args, mode=None):
    '''
        Loads a pre-trained model, calls (obtain_disease_idx, turn_into_dataframe) to generates psuedo_labels for diseases less than 'deg'. Returns dataframe ready to be augmented to df_train.
    '''
    split, deg, generate_inog = args.split, args.deg, args.generate_inog

    strt = time.time()
    TxData1 = TxData(data_folder_path = './data/')
    TxData1.prepare_split(split=split, seed=seed, no_kg=False)

    txGNN = TxGNN(
                data = TxData1, 
                weight_bias_track = False,
                proj_name = 'TxGNN',
                exp_name = 'TxGNN'
            )
        
    
    # txGNN.model_initialize(n_hid = size, 
    #                         n_inp = size, 
    #                         n_out = size, 
    #                         proto = True,
    #                         proto_num = 3,
    #                         attention = False,
    #                         sim_measure = 'all_nodes_profile',
    #                         bert_measure = 'disease_name',
    #                         agg_measure = 'rarity',
    #                         num_walks = 200,
    #                         walk_mode = 'bit',
    #                         path_length = 2)
    txGNN.load_pretrained(pre_trained_dir)

    if generate_inog:
        ind_or_cind = (txGNN.df.relation == "indication") | (txGNN.df.relation == "indication")
        disease_idxs = txGNN.df[ind_or_cind].y_idx.unique() #### Test #### to check reproducibility of valid pseudo scores
    else:
        low_disease_idx = obtain_disease_idx(TxData1=TxData1, deg=deg)
        disease_idxs = low_disease_idx
    txEval = TxEval(model = txGNN)
    indication, contraindication = None, None
    if mode != "contraindication":
        indication = txEval.eval_disease_centric(disease_idxs = disease_idxs,
                                            relation = 'indication',
                                            save_name = None, 
                                            return_raw="concise",
                                            save_result = False)
    
    if mode != "indication":
        contraindication = txEval.eval_disease_centric(disease_idxs = disease_idxs, 
                                            relation = 'contraindication',
                                            save_name = None, 
                                            return_raw="concise",
                                            save_result = False)
    # psuedo_training_df = turn_into_dataframe(results, t=k_top_candidates, least_score=least_score)
    psuedo_end = time.time() 
    print(f"time it took to generate psuedo_labels: {psuedo_end - strt}")
    return {'indication': indication, 'contraindication': contraindication}



def train_w_psuedo_labels(additional_train, seed, save_dir, args, size=None):
# def train_w_psuedo_labels(size=100, split='complex_disease', additional_train=None, create_psuedo_edges=False, seed=1, save_dir=None, dropout=0, reparam_mode=False, weight_decay=0, soft_psuedo=False):
    '''
        Takes in pretrained model and generate psuedo label? 
    '''
    dropout, create_psuedo_edges, split, reparam_mode, weight_decay, soft_pseudo, kl, neg_pseudo_sampling, no_dpm, use_og, LSP, LSP_size, T = args.dropout,\
            args.psuedo_edges, args.split, args.reparam_mode, args.weight_decay, args.soft_pseudo, args.kl, args.neg_pseudo_sampling, args.no_dpm, args.use_og, \
            args.LSP, args.LSP_size,args.T
    size = size if size is not None else args.student_size

    strt = time.time()
    TxData1 = TxData(data_folder_path = './data/')
    ## add additional psuedo-training labels
    TxData1.prepare_split(split=split, seed=seed, no_kg=False, additional_train=additional_train, create_psuedo_edges=create_psuedo_edges, soft_pseudo=soft_pseudo)
    TxGNN1 = TxGNN(
            data = TxData1, 
            weight_bias_track = True,
            proj_name = 'TxGNN',
            exp_name = 'TxGNN',
            use_og = use_og,
            T = T
        )
    TxGNN1.model_initialize(n_hid = size, 
                            n_inp = size, 
                            n_out = size, 
                            proto = not no_dpm,
                            proto_num = 3,
                            attention = False,
                            sim_measure = 'all_nodes_profile',
                            bert_measure = 'disease_name',
                            agg_measure = 'rarity',
                            num_walks = 200,
                            walk_mode = 'bit',
                            path_length = 2,
                            dropout=dropout,
                            reparam_mode=reparam_mode if additional_train is not None else None,
                            kl = kl,
                            neg_pseudo_sampling = neg_pseudo_sampling,
                            LSP = LSP,
                            LSP_size=LSP_size,)
    # Train
    # TxGNN1.pretrain(n_epoch = 1, #---
    #                 learning_rate = 1e-3,
    #                 batch_size = 1024, 
    #                 train_print_per_n = 20)

    # TxGNN1.finetune(n_epoch = 2000, #---
    #                 learning_rate = 5e-4,
    #                 train_print_per_n = 5,
    #                 valid_per_n = 20,
    #                 weight_decay = weight_decay,)

    TxGNN1.finetune(n_epoch = 5, #---
                    learning_rate = 5e-4,
                    train_print_per_n = 5,
                    valid_per_n = 20,
                    weight_decay = weight_decay,)

    print(f"time it took for this training iteration: {time.time() - strt}")
    if save_dir is not None:
        # noisy_student_fpath = './Noisy_student/'
        TxGNN1.save_model(path=save_dir)




# def train_then_generate_psuedo_labels(split, size, seed, deg, k_top_candidates, least_score, save_name, mode):
#     '''
#         Loads a pre-trained model, calls (obtain_disease_idx, turn_into_dataframe) to generates psuedo_labels for diseases less than 'deg'. Returns dataframe ready to be augmented to df_train.
#     '''
#     strt = time.time()
#     TxData1 = TxData(data_folder_path = './data/')
#     TxData1.prepare_split(split=split, seed=seed, no_kg=False)
#     low_disease_idx = obtain_disease_idx(TxData1=TxData1, deg=deg)

#     txGNN = TxGNN(
#                 data = TxData1, 
#                 weight_bias_track = False,
#                 proj_name = 'TxGNN',
#                 exp_name = 'TxGNN'
#             )
        
#     txGNN.model_initialize(n_hid = size, 
#                             n_inp = size, 
#                             n_out = size, 
#                             proto = True,
#                             proto_num = 3,
#                             attention = False,
#                             sim_measure = 'all_nodes_profile',
#                             bert_measure = 'disease_name',
#                             agg_measure = 'rarity',
#                             num_walks = 200,
#                             walk_mode = 'bit',
#                             path_length = 2)
#     # Train
#     # txGNN.pretrain(n_epoch = 1, #---
#     #                 learning_rate = 1e-3,
#     #                 batch_size = 1024, 
#     #                 train_print_per_n = 20)
#     # # txGNN.finetune(n_epoch = 40, #---
#     # txGNN.finetune(n_epoch = 500, #---
#     #                 learning_rate = 5e-4,
#     #                 train_print_per_n = 5,
#     #                 valid_per_n = 20,)
    
#     if save_name is not None:
#         noisy_student_fpath = './Noisy_student/'
#         txGNN.save_model(path = noisy_student_fpath+'properly_pre_trained_model_ckpt/seed_1_restrained_saveG')
    
#     disease_idxs = low_disease_idx
#     txEval = TxEval(model = txGNN)
#     indication, contraindication = None, None
#     if mode != "contraindication":
#         indication = txEval.eval_disease_centric(disease_idxs = disease_idxs,
#                                             relation = 'indication',
#                                             save_name = None, 
#                                             return_raw="concise",
#                                             save_result = False)
    
#     if mode != "indication":
#         contraindication = txEval.eval_disease_centric(disease_idxs = disease_idxs, 
#                                             relation = 'contraindication',
#                                             save_name = None, 
#                                             return_raw="concise",
#                                             save_result = False)
#     # psuedo_training_df = turn_into_dataframe(results, t=k_top_candidates, least_score=least_score)
#     psuedo_end = time.time() 
#     print(f"time it took to generate psuedo_labels: {psuedo_end - strt}")
#     return {'indication': indication, 'contraindication': contraindication}
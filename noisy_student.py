# import torch, dgl, accelerate
from txgnn import TxData, TxGNN, TxEval
# import torch.nn as nn
# import torch.nn.functional as F
# from accelerate.utils import set_seed
# from dgl.distributed import partition_graph
import numpy as np
import time
import pandas as pd
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

    disease_drug_degree.index.values
    low_disease = disease_drug_degree[disease_drug_degree < deg]

    id_mapping = TxData1.retrieve_id_mapping()
    id2idx = {id:idx for idx, id in id_mapping['idx2id_disease'].items()}
    print(f"Total number of diseases?: {len(id2idx)}")
    print(f"total number of {deg} > degree diseases?: {len(low_disease)}")
    low_disease_idx = low_disease.index.map(lambda x: id2idx[x] if '_' in x else id2idx[x+'.0'])#.apply(lambda x: id2idx[x])
    low_disease_idx = np.array(low_disease_idx)

    return low_disease_idx

def turn_into_dataframe(results, t, least_score):
    '''
        t = number of psuedo_labels to be generated for low_diseases
        Takes in the results eval file and returns the disease idx that have less than k degrees (drug-disease relation)
    '''
    additional_train_dict = []
    for rel, result in results.items():
        for (dis_id, drug_ids), drug_idxs, dis_idx, ranked_scores in zip(result['ranked_drug_ids'].items(), result['ranked_drug_idxs'].values(), result['dis_idx'].values(), result['ranked_scores'].values()):
            if least_score is None:
                new_dicts = [{'y_id': dis_id, 'y_idx': dis_idx, 'x_id': drug_id, 'x_idx': drug_idx, 'relation': rel, 'score': ranked_scores[i]} for i, (drug_id, drug_idx) in enumerate(zip(drug_ids, drug_idxs)) if i < t]
            else:
                new_dicts = [{'y_id': dis_id, 'y_idx': dis_idx, 'x_id': drug_id, 'x_idx': drug_idx, 'relation': rel, 'score': ranked_scores[i]} for i, (drug_id, drug_idx, ranked_score) in enumerate(zip(drug_ids, drug_idxs, ranked_scores)) if ranked_score > least_score]
            additional_train_dict += new_dicts

    df = pd.DataFrame(additional_train_dict)
    df["x_idx"] = df["x_idx"].astype(float)
    df["y_type"] = "disease"
    df["x_type"] = "drug"
    return df

def generate_psuedo_labels(pre_trained_dir='pre_trained_model_ckpt/1', split='complex_disease', size=100, seed=1, deg=1, k_top_candidates=5, least_score=None):
    '''
        Loads a pre-trained model, calls (obtain_disease_idx, turn_into_dataframe) to generates psuedo_labels for diseases less than 'deg'. Returns dataframe ready to be augmented to df_train.
    '''
    strt = time.time()
    TxData1 = TxData(data_folder_path = './data/')
    TxData1.prepare_split(split=split, seed=seed, no_kg=False)
    low_disease_idx = obtain_disease_idx(TxData1=TxData1, deg=deg)

    txGNN = TxGNN(
                data = TxData1, 
                weight_bias_track = False,
                proj_name = 'TxGNN',
                exp_name = 'TxGNN'
            )
        
    txGNN.model_initialize(n_hid = size, 
                            n_inp = size, 
                            n_out = size, 
                            proto = True,
                            proto_num = 3,
                            attention = False,
                            sim_measure = 'all_nodes_profile',
                            bert_measure = 'disease_name',
                            agg_measure = 'rarity',
                            num_walks = 200,
                            walk_mode = 'bit',
                            path_length = 2)
    txGNN.load_pretrained(pre_trained_dir)
    disease_idxs = low_disease_idx
    txEval = TxEval(model = txGNN)
    indication = txEval.eval_disease_centric(disease_idxs = disease_idxs,
                                         relation = 'indication',
                                         save_name = None, 
                                         return_raw="concise",
                                         save_result = False)
    
    contraindication = txEval.eval_disease_centric(disease_idxs = disease_idxs, 
                                        relation = 'contraindication',
                                        save_name = None, 
                                        return_raw="concise",
                                        save_result = False)
    results =  {"indication":indication, "contraindication":contraindication}
    psuedo_training_df = turn_into_dataframe(results, t=k_top_candidates, least_score=least_score)
    psuedo_end = time.time() 
    print(f"time it took to generate psuedo_labels: {psuedo_end - strt}")
    return psuedo_training_df

def train_w_psuedo_labels(size, split, additional_train, create_psuedo_edges, seed, save_dir, dropout, reparam_mode, weight_decay, soft_pseudo):
# def train_w_psuedo_labels(size=100, split='complex_disease', additional_train=None, create_psuedo_edges=False, seed=1, save_dir=None, dropout=0, reparam_mode=False, weight_decay=0, soft_psuedo=False):
    '''
        Takes in pretrained model and generate psuedo label? 
    '''
    strt = time.time()
    TxData1 = TxData(data_folder_path = './data/')
    ## add additional psuedo-training labels
    TxData1.prepare_split(split=split, seed=seed, no_kg=False, additional_train=additional_train, create_psuedo_edges=create_psuedo_edges, soft_pseudo=soft_pseudo)
    TxGNN1 = TxGNN(
            data = TxData1, 
            weight_bias_track = False, #True,
            proj_name = 'TxGNN',
            exp_name = 'TxGNN'
        )
    TxGNN1.model_initialize(n_hid = size, 
                            n_inp = size, 
                            n_out = size, 
                            proto = True,
                            proto_num = 3,
                            attention = False,
                            sim_measure = 'all_nodes_profile',
                            bert_measure = 'disease_name',
                            agg_measure = 'rarity',
                            num_walks = 200,
                            walk_mode = 'bit',
                            path_length = 2,
                            dropout=dropout,
                            reparam_mode=reparam_mode)
    # Train
    # TxGNN1.pretrain(n_epoch = 1, #---
    #                 learning_rate = 1e-3,
    #                 batch_size = 1024, 
    #                 train_print_per_n = 20)
    TxGNN1.finetune(n_epoch = 500, #---
                    learning_rate = 5e-4,
                    train_print_per_n = 5,
                    valid_per_n = 20,
                    weight_decay = weight_decay,)
    print(f"time it took for this training iteration: {time.time() - strt}")
    if save_dir is not None:
        noisy_student_fpath = './Noisy_student/'
        TxGNN1.save_model(path = noisy_student_fpath+save_dir)


if __name__ == '__main__':
    import argparse
    import os
    import random

    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', default=0.0)
    parser.add_argument('--reparam_mode', default=False, help='choose from {MLP, RMLP, MPNN}')
    parser.add_argument('--psuedo_label_fname', default=None, help='choose from {psuedo_labels_75000.csv, }') ## is default None? 
    parser.add_argument('--split', default='complex_disease', help='choose from {complex_disease, ...}')
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--k_top_candidates', default=5, type=int)
    parser.add_argument('--psuedo_edges', action='store_true')
    # parser.add_argument('--train_from_scratch', action='store_true')
    parser.add_argument('--student_size', default=120, type=int)
    # parser.add_argument('--three_iter_from_scratch', action='store_true')
    parser.add_argument('--save_name', type=str, default=None)
    # parser.add_argument('--scaling_psuedo', action='store_true')
    # parser.add_argument('--testing', action='store_true')
    parser.add_argument('--deg', default=1, type=float) ## 'inf' for all diseases?
    # parser.add_argument('--more_than_one_model_per_script', action='store_true')
    parser.add_argument('--least_score', default=None, type=float)
    parser.add_argument('--random_seed', action='store_true')
    parser.add_argument('--set_seed', default=1, type=int)
    parser.add_argument('--soft_pseudo', action='store_true')

    ## pre_trained_model
    args = parser.parse_args()
    # save_name = 'Noisy_student2' if args.use_diff_savedir else 'Noisy_student'

    if args.random_seed:
        seed = random.randint(2, 100)
    else:
        seed = args.set_seed
    print(f"Using seed: {seed}")
    # if args.three_iter_from_scratch:
    #     # seed = random.randint(2, 100)
    #     print(f"Using seed{seed} to do full three iteration training")
    #     train_w_psuedo_labels(split=args.split, seed=seed, save_dir="Teacher")
    #     for i in range(3):
    #         print(f"generating and training Student{i+1}")
    #         args.k_top_candidates = args.k_top_candidates * (i+1) if args.scaling_psuedo else args.k_top_candidates ## scale num of psuedo labels
    #         size = 100 if i == 0 else args.student_size
    #         pre_trained_dir = f'./{save_dir}/Student{i}' if i > 0 else f"./{save_dir}/Teacher"
    #         psuedo_labels = generate_psuedo_labels(pre_trained_dir=pre_trained_dir, split=args.split, size=size, seed=seed, deg=args.deg, k_top_candidates=args.k_top_candidates)
    #         train_w_psuedo_labels(size=args.student_size, 
    #                         dropout=args.dropout, 
    #                         save_dir=f"Student{i+1}/", 
    #                         additional_train=psuedo_labels, 
    #                         create_psuedo_edges=args.psuedo_edges, 
    #                         split=args.split, 
    #                         reparam_mode=args.reparam_mode,
    #                         weight_decay=args.weight_decay)
    # elif args.testing:
    #     for i in range(2):
    #         train_w_psuedo_labels(size=args.student_size, 
    #                         dropout=args.dropout, 
    #                         save_dir="The_First_Student/", 
    #                         additional_train=None, 
    #                         create_psuedo_edges=args.psuedo_edges, 
    #                         split=args.split, 
    #                         seed=seed,
    #                         reparam_mode=args.reparam_mode,
    #                         weight_decay=args.weight_decay)
    # elif args.train_from_scratch: ## To evaluate model at different seed
    #         ## should not be used until fixes
    #         seed = random.randint(2, 100)
    #         print(f"randomly picked seed is: {seed}")
    #         train_w_psuedo_labels(split=args.split, seed=seed, save_dir="random_seed")
    #         psuedo_labels = generate_psuedo_labels(pre_trained_dir=f'./{save_dir}/random_seed', split=args.split, size=100, seed=seed, deg=args.deg, k_top_candidates=args.k_top_candidates)
    if args.psuedo_label_fname is not None and os.path.exists(args.psuedo_label_fname):
        ## only for training w/ psuedo labels
        psuedo_labels = pd.read_csv(args.psuedo_label_fname)
        print(f"Loading generated psuedo_labels with size: {len(psuedo_labels)}")
        train_w_psuedo_labels(size=args.student_size, 
                            dropout=args.dropout, 
                            save_dir=args.save_name, 
                            additional_train=psuedo_labels, 
                            create_psuedo_edges=args.psuedo_edges, 
                            split=args.split, 
                            seed=seed,
                            reparam_mode=args.reparam_mode,
                            weight_decay=args.weight_decay,
                            soft_pseudo=args.soft_pseudo)
    else:
        ## only for generating psuedo labels
        print(f"Only generating: {args.psuedo_label_fname}")
        psuedo_labels = generate_psuedo_labels(pre_trained_dir='pre_trained_model_ckpt/seed_1_normal', split=args.split, size=100, seed=seed, deg=args.deg, k_top_candidates=args.k_top_candidates, least_score=args.least_score)
        psuedo_labels.to_csv(args.psuedo_label_fname, index=False)

###

def generate_n_psuedo_labels(n=5, pre_trained_dir='pre_trained_model_ckpt/1', split='complex_disease', size=100, seed=1, deg=1, k_top_candidates=5, least_score=None):
    '''
        Loads a pre-trained model, calls (obtain_disease_idx, turn_into_dataframe) to generates psuedo_labels for diseases less than 'deg'. Returns dataframe ready to be augmented to df_train.
    '''
    strt = time.time()
    TxData1 = TxData(data_folder_path = './data/')
    TxData1.prepare_split(split=split, seed=seed, no_kg=False)
    low_disease_idx = np.random.choice(obtain_disease_idx(TxData1=TxData1, deg=deg), n, replace=False)

    txGNN = TxGNN(
                data = TxData1, 
                weight_bias_track = False,
                proj_name = 'TxGNN',
                exp_name = 'TxGNN'
            )
        
    txGNN.model_initialize(n_hid = size, 
                            n_inp = size, 
                            n_out = size, 
                            proto = True,
                            proto_num = 3,
                            attention = False,
                            sim_measure = 'all_nodes_profile',
                            bert_measure = 'disease_name',
                            agg_measure = 'rarity',
                            num_walks = 200,
                            walk_mode = 'bit',
                            path_length = 2)
    txGNN.load_pretrained(pre_trained_dir)
    disease_idxs = low_disease_idx
    txEval = TxEval(model = txGNN)
    indication = txEval.eval_disease_centric(disease_idxs = disease_idxs,
                                         relation = 'indication',
                                         save_name = None, 
                                         return_raw="concise",
                                         save_result = False)
    
    contraindication = txEval.eval_disease_centric(disease_idxs = disease_idxs, 
                                        relation = 'contraindication',
                                        save_name = None, 
                                        return_raw="concise",
                                        save_result = False)
    results =  {"indication":indication, "contraindication":contraindication}
    psuedo_training_df = turn_into_dataframe(results, t=k_top_candidates, least_score=least_score)
    psuedo_end = time.time() 
    print(f"time it took to generate psuedo_labels: {psuedo_end - strt}")
    return psuedo_training_df

def finetune_pretrained(pre_trained_dir='pre_trained_model_ckpt/1', size=100, split='complex_disease', additional_train=None, create_psuedo_edges=False, seed=1, save_dir=None, dropout=0, reparam_mode=False, weight_decay=0):
    strt = time.time()
    TxData1 = TxData(data_folder_path = './data/')
    ## add additional psuedo-training labels
    TxData1.prepare_split(split=split, seed=seed, no_kg=False, additional_train=additional_train, create_psuedo_edges=create_psuedo_edges,)
    TxGNN1 = TxGNN(
            data = TxData1, 
            weight_bias_track = False, #True,
            proj_name = 'TxGNN',
            exp_name = 'TxGNN'
        )
    TxGNN1.model_initialize(n_hid = size, 
                            n_inp = size, 
                            n_out = size, 
                            proto = True,
                            proto_num = 3,
                            attention = False,
                            sim_measure = 'all_nodes_profile',
                            bert_measure = 'disease_name',
                            agg_measure = 'rarity',
                            num_walks = 200,
                            walk_mode = 'bit',
                            path_length = 2,
                            dropout=dropout,
                            reparam_mode=reparam_mode)
    
    TxGNN1.load_pretrained(pre_trained_dir)
    
    TxGNN1.finetune(n_epoch = 500, #---
                    learning_rate = 5e-4,
                    train_print_per_n = 5,
                    valid_per_n = 20,
                    weight_decay = weight_decay,)
    print(f"time it took for this training iteration: {time.time() - strt}")
    if save_dir is not None:
        noisy_student_fpath = './Noisy_student/'
        TxGNN1.save_model(path = noisy_student_fpath+save_dir)
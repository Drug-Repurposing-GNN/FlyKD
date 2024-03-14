from txgnn import TxData, TxGNN, TxEval
from txgnn.utils import create_split, print_val_test_auprc
import numpy as np
import time
import pandas as pd
import pickle
import os
import sys
import pprint
import random

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
    low_disease = disease_drug_degree[disease_drug_degree < deg]

    id_mapping = TxData1.retrieve_id_mapping()
    id2idx = {id:idx for idx, id in id_mapping['idx2id_disease'].items()}
    print(f"Total number of diseases?: {len(id2idx)}")
    print(f"total number of {deg} > degree diseases?: {len(low_disease)}")
    low_disease_idx = low_disease.index.map(lambda x: id2idx[x] if '_' in x else id2idx[x+'.0'])#.apply(lambda x: id2idx[x])
    low_disease_idx = np.array(low_disease_idx)

    return low_disease_idx

def disease_idx_wout_val_test(seed, args):
    ## extract all disease's ids
    data_folder = "./data/"
    kg_path = data_folder + 'kg_directed.csv'
    df = pd.read_csv(kg_path)
    df_train, df_valid, df_test = create_split(df, args.split, None, None, seed)

    df_dd_x_idx = df[df.x_type == "disease"].x_idx
    df_dd_y_idx = df[df.y_type == "disease"].y_idx
    df_dd_idx = set()
    df_dd_idx.update(df_dd_x_idx.values, df_dd_y_idx.values)
    full_length = len(df_dd_idx)
    df_valid_test = pd.concat([df_valid, df_test])
    df_valid_test_disease_idx = df_valid_test[df_valid_test.relation.isin(["indication", "contraindication"])].y_idx.drop_duplicates().values
    df_dd_idx.difference_update(df_valid_test_disease_idx)
    assert full_length - len(df_valid_test_disease_idx) == len(df_dd_idx)
    print(f"created list of disease idxs excluding ones existing in validation, test. Went from size {full_length} to {len(df_dd_idx)} without val and test")
    return df_train.drop_duplicates(), df_valid.drop_duplicates(), df_test.drop_duplicates(), np.array(list(df_dd_idx))

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
    high_disease_series = disease_drug_degree[disease_drug_degree >= deg]
    high_drug_series = drugs_value_counts[drugs_value_counts >= deg]
    high_disease_set = set(high_disease_series.index)
    high_drug_set = set(high_drug_series.index)
    filtered_pseudo_train = pseudo_train[(pseudo_train.y_id.isin(high_disease_set)) & (pseudo_train.x_id.isin(high_drug_set))]
    return filtered_pseudo_train


def _turn_into_df_helper(result, rel, dd_df, args):
    random_k, k_top_bottom_candidates, strong_scores =  args.random_pseudo_k, args.k_top_bottom_candidates, args.strong_scores
    concat_df_dd = []
    extra_df_dd = []
    NUM_DRUGS = len(next(iter(result['ranked_drug_ids'].values())))
    strt = time.time()
    for (dis_id, drug_ids), drug_idxs, dis_idx, ranked_scores in zip(result['ranked_drug_ids'].items(), result['ranked_drug_idxs'].values(), result['dis_idx'].values(), result['ranked_scores'].values()):
        extra_dicts = None
        if k_top_bottom_candidates is not None:
            extra_dicts = [{'y_id': dis_id, 'y_idx': dis_idx, 'x_id': drug_id, 'x_idx': drug_idx, 'relation': rel, 'score': ranked_score} for i, (drug_id, drug_idx, ranked_score) in enumerate(zip(drug_ids, drug_idxs, ranked_scores)) if i < k_top_bottom_candidates or i >= NUM_DRUGS - k_top_bottom_candidates]
        elif strong_scores is not None:
            extra_dicts = [{'y_id': dis_id, 'y_idx': dis_idx, 'x_id': drug_id, 'x_idx': drug_idx, 'relation': rel, 'score': ranked_score} for i, (drug_id, drug_idx, ranked_score) in enumerate(zip(drug_ids, drug_idxs, ranked_scores)) if abs(ranked_score) > strong_scores]
        new_dicts = [{'y_id': dis_id, 'y_idx': dis_idx, 'x_id': drug_id, 'x_idx': drug_idx, 'relation': rel, 'score': ranked_score} for i, (drug_id, drug_idx, ranked_score) in enumerate(zip(drug_ids, drug_idxs, ranked_scores))]
        ## generating random k pseudo labels on non-existing relations
        if random_k is not None:
            extra_df_dd.append(pd.DataFrame(random.sample(new_dicts, random_k)))
        ## generate pseudo labels on existing relations
        temp_df = pd.DataFrame(new_dicts)
        concat_df_dd.append(temp_df)
        if extra_dicts is not None:
            extra_df = pd.DataFrame(extra_dicts)        
            extra_df_dd.append(extra_df)
    b4_merge = time.time()
    print(f"time b4 merge: {b4_merge - strt}")

    ## Filter and concatenate list of dataframes
    df = pd.concat(concat_df_dd)
    extra_df = pd.concat(extra_df_dd) if len(extra_df_dd) > 0 else None
    if extra_df is None:    
        print(f"No pseudo labels outside of training dataset were added")
    ## random relation can only happen between entities that have seen labels. But this random relation could be a non-existent relation
    if extra_df is not None and not args.include_all_pseudo:
        b4_enforcing = len(extra_df)
        print(extra_df.x_id.isin(dd_df.x_id).sum(), extra_df.y_id.isin(dd_df.y_id).sum())
        extra_df = extra_df[(extra_df.x_id.isin(dd_df.x_id)) & (extra_df.y_id.isin(dd_df.y_id))]
        print(f"b4 and after enforcing only between entities that have seen labels: {b4_enforcing}, {len(extra_df)}")
    df = df.merge(dd_df[['x_id', 'y_id', 'relation']], on=['x_id', 'y_id', 'relation'], how="inner")
    df = pd.concat([df, extra_df])
    print(f"merging time: {time.time() - b4_merge}")

    b4_dropping_dup = len(df)
    df = df.drop_duplicates()
    print(f"[{rel}]: before and after dropping dup: {b4_dropping_dup}, {len(df)}")
    df["x_idx"] = df["x_idx"].astype(float)
    df["y_type"] = "disease"
    df["x_type"] = "drug"
    return df


def turn_into_df(results, txdata, args,):
    print("turning pseudo labels into dataframe...")
    strt = time.time()
    dd_df = txdata.df_train if args.ptrain or args.include_all_pseudo else txdata.og_filtered_dd
    dd_df = dd_df[dd_df.relation.isin(["indication", "contraindication"])]
    concat_df = []
    for rel, result in results.items():
        concat_df.append(_turn_into_df_helper(result, rel, dd_df, args,))
    pseudo_df = pd.concat(concat_df)
    return pseudo_df


def init_logfile(i, seed, args):
    '''
        create and set logfile to be written. Also write init messages such as args and seed
    '''
    if args.epochs:
        save_dir = f"./Results_e{args.epochs}/{args.fname}/{seed}/"
    else:
        save_dir = f"./after_saving_rev_e1200_90/{args.fname}/{seed}/" 

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_file = open(save_dir + f"{i}.txt", 'w', buffering=1)
    sys.stderr = log_file
    sys.stdout = log_file
    print("Arguments received:")
    pprint.pprint(vars(args))
    print("------------------------------")
    print(f"Using seed: {seed}")
    return save_dir, log_file

def print_val_test_auprc_w_path(pretrained_path, split, seed):
    TxData1 = TxData(data_folder_path = './data/')
    TxData1.prepare_split(split=split, seed=seed, no_kg=False)
    TxGNN1 = TxGNN(
            data = TxData1, 
            weight_bias_track = False,
            proj_name = 'TxGNN',
            exp_name = 'TxGNN'
        )
    TxGNN1.load_pretrained(pretrained_path)
    TxGNN1.print_model_size()
    G = TxGNN1.G.to(TxGNN1.device)
    best_G = TxGNN1.best_G.to(TxGNN1.device)
    best_model = TxGNN1.best_model.to(TxGNN1.device)
    best_model.eval()
    g_valid_pos, g_valid_neg = TxGNN1.g_valid_pos, TxGNN1.g_valid_neg
    g_test_pos, g_test_neg = TxGNN1.g_test_pos, TxGNN1.g_test_neg
    print_val_test_auprc(best_model, g_valid_pos, g_valid_neg, g_test_pos, g_test_neg, best_G, TxGNN1.dd_etypes, TxGNN1.device)
    
# def generate_pseudo_labels(pre_trained_dir, size, seed, mode=None):
def generate_pseudo_labels(pre_trained_dir, size, seed, args, mode=None):
    '''
        Loads a pre-trained model, calls (obtain_disease_idx, turn_into_dataframe) to generates psuedo_labels for diseases less than 'deg'. Returns dataframe ready to be augmented to df_train.
    '''
    split = args.split
    # split, deg, generate_inog = args.split, args.deg, args.generate_inog

    strt = time.time()
    TxData1 = TxData(data_folder_path = './data/')
    TxData1.prepare_split(split=split, seed=seed, no_kg=False, pseudo_on_train=args.ptrain)

    txGNN = TxGNN(
                data = TxData1, 
                weight_bias_track = False,
                proj_name = 'TxGNN',
                exp_name = 'TxGNN'
            )
        
    txGNN.load_pretrained(pre_trained_dir)

    if args.debug:
        print("using ptrain")
        dd_df = TxData1.df_train
        ind_idx = dd_df[dd_df.relation == "indication"].y_idx.unique()[:10]
        cind_idxs = dd_df[dd_df.relation == "contraindication"].y_idx.unique()[:10] #### Test #### to check reproducibility of valid pseudo scores
        dd_df = dd_df[dd_df.relation.isin(["indication", "contraindication"])]
    elif args.ptrain:
        print("using ptrain")
        dd_df = TxData1.df_train
        ind_idx = dd_df[dd_df.relation == "indication"].y_idx.unique()
        cind_idxs = dd_df[dd_df.relation == "contraindication"].y_idx.unique() #### Test #### to check reproducibility of valid pseudo scores
        dd_df = dd_df[dd_df.relation.isin(["indication", "contraindication"])]
    elif args.exlucde_valid_test:
        df_train, df_valid, df_test, disease_idx = disease_idx_wout_val_test(seed, args)
        assert sum(df_train.x_idx - TxData1.df_train.x_idx) == 0, "split is somehow different"
        assert sum(df_valid.x_idx - TxData1.df_valid.x_idx) == 0, "split is somehow different"
        assert sum(df_test.x_idx - TxData1.df_test.x_idx) == 0 , "split is somehow different"
        ind_idx = disease_idx
        cind_idxs = disease_idx
    else:
        all_disease_idx = TxData1.df[TxData1.df['y_type'] == "disease"].y_idx.unique()
        ind_idx = all_disease_idx
        cind_idxs = all_disease_idx

    txEval = TxEval(model = txGNN)
    indication, contraindication = None, None
    if mode != "contraindication":
        indication = txEval.eval_disease_centric(disease_idxs = ind_idx,
                                            relation = 'indication',
                                            save_name = None, 
                                            return_raw="concise",
                                            save_result = False)
    
    if mode != "indication":
        contraindication = txEval.eval_disease_centric(disease_idxs = cind_idxs, 
                                            relation = 'contraindication',
                                            save_name = None, 
                                            return_raw="concise",
                                            save_result = False)
    pretrain_scores_dict = None
    if args.pseudo_pretrain:
        best_G, best_model = txGNN.best_G, txGNN.best_model
        h, beta_kl_loss, distmult = best_model(best_G, pretrain_mode=True, mode='train', return_h_and_kl=True,)
        pretrain_scores_dict, _ = distmult(best_G, best_G, h, mode=None, pretrain_mode=True,)
    psuedo_end = time.time() 
    print(f"time it took to generate psuedo_labels: {psuedo_end - strt}")
    return TxData1, {'indication': indication, 'contraindication': contraindication}, pretrain_scores_dict


def train_w_psuedo_labels(additional_train, pretrain_scores_dict, seed, save_dir, args, size=None, i=0):
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
    ## pass in pseudo_pretrain scores
    args.pretrain_scores_dict = pretrain_scores_dict
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
                            reparam_mode=reparam_mode if i != 0 or args.force_reparam else None,
                            kl = kl,
                            LSP = LSP if i != 0 else None,
                            LSP_size=LSP_size if i != 0 else None,
                            args=args,)
    
    # Train
    if args.debug:
        TxGNN1.print_model_size()
        TxGNN1.finetune(n_epoch = 19, #---
                    learning_rate = 5e-4,
                    train_print_per_n = 5,
                    valid_per_n = 20,
                    weight_decay = weight_decay,
                    args=args)
    else:
        if args.i==0:
            pretrain_phase_ckpt = f"./pretrained_models/{args.teacher_size}_pretrain/{seed}"
        else:
            pretrain_phase_ckpt = f"./pretrained_models/{args.student_size}_pretrain/{seed}"
            
        if (not args.force_iter0 or args.force_finetune_iter0) and os.path.exists(pretrain_phase_ckpt):
            print(f"loading pretraining phase from {pretrain_phase_ckpt}")
            TxGNN1.load_pretrained(pretrain_phase_ckpt, keep_config=True)
        else:
            print(f"no saved pretrain phase detected. Starting pretraining from scratch.")
            TxGNN1.pretrain(n_epoch = 1, #---
                    learning_rate = 1e-3,
                    batch_size = 1024, 
                    train_print_per_n = 20)
            
            TxGNN1.save_model(pretrain_phase_ckpt)
        if args.only_pretrain:
            raise ValueError("Only saving pretraining phase successful")

        TxGNN1.print_model_size()
        
        n_epoch = 1200
        if args.epochs:
            n_epoch = args.epochs
        TxGNN1.finetune(n_epoch = n_epoch, #---
                learning_rate = 5e-4,
                train_print_per_n = 5,
                valid_per_n = 20,
                weight_decay = weight_decay,
                args=args,)
    print(f"time it took for this training iteration: {time.time() - strt}")
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # noisy_student_fpath = './Noisy_student/'
        TxGNN1.save_model(path=save_dir)
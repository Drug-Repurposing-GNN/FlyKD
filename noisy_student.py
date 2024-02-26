# import torch, dgl, accelerate
from txgnn import TxData, TxGNN, TxEval
# import torch.nn as nn
# import torch.nn.functional as F
from accelerate.utils import set_seed as rng_set_seed
# from dgl.distributed import partition_graph
import numpy as np
import time
import pandas as pd
from noisy_student_utils import *
import pickle
import sys
import pprint
import wandb
import argparse
import os
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
if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', default=0.0)
    parser.add_argument('--reparam_mode', default=False, help='choose from {MLP, RMLP, MPNN}')
    parser.add_argument('--psuedo_label_fname', default=None, help='choose from {psuedo_labels_75000.csv, }') ## is default None? 
    parser.add_argument('--split', default='complex_disease', help='choose from {complex_disease, ...}')
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--k_top_candidates', default=float('inf'), type=float)
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
    parser.add_argument('--fix_split_random_seed', action='store_true')
    parser.add_argument('--random_seed', action='store_true')
    parser.add_argument('--set_seed', default=1, type=int)
    parser.add_argument('--soft_pseudo', action='store_true')
    parser.add_argument('--kl', action='store_true')
    parser.add_argument('--train_then_generate', action='store_true')
    parser.add_argument('--use_high_degree', action='store_true')
    parser.add_argument('--neg_pseudo_sampling', action='store_true')
    parser.add_argument('--generate_indication', action='store_true')
    parser.add_argument('--generate_contraindication', action='store_true')
    parser.add_argument('--no_dpm', action='store_true', help="disables disease pooling mechanism during evaluation")
    parser.add_argument('--use_og', action='store_true', help="use original loss as additional loss in pseudo training")
    parser.add_argument('--LSP', type=str, help="additional Local Sturcture (similarity vector) loss using KL div. Mode: {cosine, L2, RBF, Poly}")
    parser.add_argument('--LSP_size', type=str, help="use full or partial relations to compute the LS vector?")
    parser.add_argument('--T', type=float, default=1, help="temperature for pseudo loss")
    parser.add_argument("--generate_inog", action="store_true", help="Generate only on existing relations (train, val, data) for quicker time")
    parser.add_argument("--repeat", type=int, default=1, help="repeat n many times with random seed to create confidence interval")
    parser.add_argument("--iter", type=int, default=0, help="iteration for noisy-student framework. 0 means only train teacher method")
    parser.add_argument("--teacher_size", type=int, default=100, help="the name of the filename or directory for eval_perf")
    parser.add_argument("fname", type=str, help="the name of the filename or directory for eval_perf")
    parser.add_argument("--debug", action="store_true", help="send files to debug folder")
    parser.add_argument("--random_pseudo_k", type=int, help="number of pseudo labels on random relations. (Uses disease pooling mechanism)")
    parser.add_argument("--no_wandb", action="store_true", help="turn off wandb")
    
    ## passing in args
    args = parser.parse_args()
    if args.debug:
        args.fname = "debug" 
    ## -1 means all return all drug candidates
    # args.k_top_candidates =  if args.k_top_candidates == -1 else args.k_top_candidates 

    ## to get notified when training is finished
    if args.no_wandb is None:
        proj_name = 'TxGNN'
        exp_name = 'TxGNN'
        wandb.init(project=proj_name, name=exp_name)  
        wandb.config.update(vars(args))

    for _ in range(args.repeat):
        ## setting seed for rng and data split
        seed = args.set_seed
        if args.fix_split_random_seed:
            rng_seed = random.randint(2, 100)
            # print(f"tesing pseudo label approach is time consuming so using seed 1 for data but randomize RNG with seed: {rng_seed}")
            rng_set_seed(rng_seed)
        elif args.repeat > 1 or args.random_seed:
            seed = random.randint(2, 100)

        ## instantiating log file
        # fname = f"soft_pseudo" if args.soft_pseudo else ""
        # fname += f"_pneg" if args.neg_pseudo_sampling else ""
        # fname += f"_inog" if args.generate_inog else ""
        # fname += f"_wog" if args.use_og else ""
        # fname += f"_{args.LSP_size}" if args.LSP_size is not None else ""
        # fname += f"_{args.LSP}" if args.LSP is not None else ""
        # if args.iter > 0:

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
        
        # if args.psuedo_label_fname is None:
        #     ## Training w/out pseudo labels
        #     print("Training without pseudo labels")
        #     train_w_psuedo_labels(size=args.student_size, 
        #                         dropout=args.dropout, 
        #                         save_dir=args.save_name, 
        #                         additional_train=None, 
        #                         create_psuedo_edges=args.psuedo_edges, 
        #                         split=args.split, 
        #                         seed=seed,
        #                         reparam_mode=args.reparam_mode,
        #                         weight_decay=args.weight_decay,
        #                         soft_pseudo=args.soft_pseudo,
        #                         kl = args.kl,
        #                         neg_pseudo_sampling = args.neg_pseudo_sampling,
        #                         no_dpm = args.no_dpm,
        #                         use_og = args.use_og,
        #                         LSP = args.LSP,
        #                         LSP_size=args.LSP_size,
        #                         T=args.T)

        # elif args.psuedo_label_fname is not None and os.path.exists(args.psuedo_label_fname):
        #     ## only for training w/ psuedo labels
        #     pseudo_labels = pd.read_csv(args.psuedo_label_fname)
        #     if args.use_high_degree:
        #         print(f'before applying high-degree-only filter: {len(pseudo_labels)}')
        #         pseudo_labels = obtain_high_degree_disease_id_w_df(pseudo_labels, seed=seed, deg=1)
        #         print(f'after applying high-degree-only filter: {len(pseudo_labels)}')
        #     else:
        #         pseudo_labels

        #     print(f"Loading generated psuedo_labels with size: {len(pseudo_labels)}")
        #     train_w_psuedo_labels(size=args.student_size, 
        #                         dropout=args.dropout, 
        #                         save_dir=args.save_name, 
        #                         additional_train=pseudo_labels, 
        #                         create_psuedo_edges=args.psuedo_edges, 
        #                         split=args.split, 
        #                         seed=seed,
        #                         reparam_mode=args.reparam_mode,
        #                         weight_decay=args.weight_decay,
        #                         soft_pseudo=args.soft_pseudo,
        #                         kl = args.kl,
        #                         neg_pseudo_sampling = args.neg_pseudo_sampling,
        #                         no_dpm = args.no_dpm,
        #                         use_og = args.use_og,
        #                         LSP = args.LSP,
        #                         LSP_size=args.LSP_size,
        #                         T=args.T)

        for i, _ in enumerate(range(args.iter + 1)):
            save_dir, log_file = init_logfile(i, seed, args)
            # pre_trained_dir = 'properly_pre_trained_model_ckpt/seed_1_restrained_saveG'
            ith_model_ckpt = f"{save_dir}{i}_model_ckpt/" if args.iter > 0 else None
            # ith_savedir = f"{save_dir}{i}_" ## add numbers if you want to specify which iteration
            # ith_model_ckpt = ith_savedir+"model_ckpt/"
            if i == 0:
                ## Use teacher size for first model (teacher model)
                size = args.teacher_size 
                ## training function
                print("Training *without* pseudo labels")
                train_w_psuedo_labels(None, seed, ith_model_ckpt, args, size=size)
            else:
                prev_trained_dir = f"{save_dir}{i-1}_model_ckpt/"
                size = args.teacher_size if i == 1 else args.student_size
                results = generate_pseudo_labels(prev_trained_dir, size, seed, args)
                psuedo_df = turn_into_df(results, args.random_pseudo_k) ## saves the pseudo csv file
                # psuedo_dir = prev_trained_dir + "pseudo_df.csv"
                train_w_psuedo_labels(psuedo_df, seed, ith_model_ckpt, args)
                del results, psuedo_df
            log_file.close()


    

























# def generate_n_psuedo_labels(n=5, pre_trained_dir='pre_trained_model_ckpt/1', split='complex_disease', size=100, seed=1, deg=1, k_top_candidates=5, least_score=None):
#     '''
#         Loads a pre-trained model, calls (obtain_disease_idx, turn_into_dataframe) to generates psuedo_labels for diseases less than 'deg'. Returns dataframe ready to be augmented to df_train.
#     '''
#     strt = time.time()
#     TxData1 = TxData(data_folder_path = './data/')
#     TxData1.prepare_split(split=split, seed=seed, no_kg=False)
#     low_disease_idx = np.random.choice(obtain_disease_idx(TxData1=TxData1, deg=deg), n, replace=False)

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
#     txGNN.load_pretrained(pre_trained_dir)
#     disease_idxs = low_disease_idx
#     txEval = TxEval(model = txGNN)
#     indication = txEval.eval_disease_centric(disease_idxs = disease_idxs,
#                                          relation = 'indication',
#                                          save_name = None, 
#                                          return_raw="concise",
#                                          save_result = False)
    
#     contraindication = txEval.eval_disease_centric(disease_idxs = disease_idxs, 
#                                         relation = 'contraindication',
#                                         save_name = None, 
#                                         return_raw="concise",
#                                         save_result = False)
#     results =  {"indication":indication, "contraindication":contraindication}
#     psuedo_training_df = turn_into_dataframe(results, t=k_top_candidates, least_score=least_score)
#     psuedo_end = time.time() 
#     print(f"time it took to generate psuedo_labels: {psuedo_end - strt}")
#     return psuedo_training_df

# def finetune_pretrained(pre_trained_dir='pre_trained_model_ckpt/1', size=100, split='complex_disease', additional_train=None, create_psuedo_edges=False, seed=1, save_dir=None, dropout=0, reparam_mode=False, weight_decay=0):
#     strt = time.time()
#     TxData1 = TxData(data_folder_path = './data/')
#     ## add additional psuedo-training labels
#     TxData1.prepare_split(split=split, seed=seed, no_kg=False, additional_train=additional_train, create_psuedo_edges=create_psuedo_edges,)
#     TxGNN1 = TxGNN(
#             data = TxData1, 
#             weight_bias_track = False, #True,
#             proj_name = 'TxGNN',
#             exp_name = 'TxGNN'
#         )
#     TxGNN1.model_initialize(n_hid = size, 
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
#                             path_length = 2,
#                             dropout=dropout,
#                             reparam_mode=reparam_mode)
    
#     TxGNN1.load_pretrained(pre_trained_dir)
    
#     TxGNN1.finetune(n_epoch = 500, #---
#                     learning_rate = 5e-4,
#                     train_print_per_n = 5,
#                     valid_per_n = 20,
#                     weight_decay = weight_decay,)
#     print(f"time it took for this training iteration: {time.time() - strt}")
#     if save_dir is not None:
#         noisy_student_fpath = './Noisy_student/'
#         TxGNN1.save_model(path = noisy_student_fpath+save_dir)
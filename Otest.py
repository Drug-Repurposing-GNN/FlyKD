# import noisy_student

# pLabelsWscore = noisy_student.generate_psuedo_labels()

# pLabelsWscore.to_csv('psuedo_labels_w_scores.csv')

from accelerate.utils import set_seed as rng_set_seed
from txgnn import TxGNN, TxEval, TxData, NewDataHandler
from txgnn.utils import *
from tqdm import tqdm
import time

split = "complex_disease"
seed = 41

rng_set_seed(1)

TxData1 = TxData(data_folder_path = './data/')
TxData1.prepare_split(split = split, seed = 1, no_kg = False)
TxGNN1 = TxGNN(
        data = TxData1, 
        weight_bias_track = False,
        proj_name = 'TxGNN',
        exp_name = 'TxGNN'
    )
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

TxGNN1.pretrain(n_epoch = 1, 
               learning_rate = 1e-3,
               batch_size = 1024, 
               train_print_per_n = 20)

TxGNN1.finetune(n_epoch = 250, #---
                learning_rate = 5e-4,
                train_print_per_n = 5,
                valid_per_n = 20,)

print("##########\n" * 30)
rng_set_seed(41)


TxData1 = TxData(data_folder_path = './data/')
TxData1.prepare_split(split = split, seed = seed, no_kg = False)
TxGNN1 = TxGNN(
        data = TxData1, 
        weight_bias_track = False,
        proj_name = 'TxGNN',
        exp_name = 'TxGNN'
    )
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

TxGNN1.pretrain(n_epoch = 1, 
               learning_rate = 1e-3,
               batch_size = 1024, 
               train_print_per_n = 20)

TxGNN1.finetune(n_epoch = 250, #---
                learning_rate = 5e-4,
                train_print_per_n = 5,
                valid_per_n = 20,)

print("##########\n" * 30)
rng_set_seed(41)


TxData1 = TxData(data_folder_path = './data/')
TxData1.prepare_split(split = split, seed = 1, no_kg = False)
TxGNN1 = TxGNN(
        data = TxData1, 
        weight_bias_track = False,
        proj_name = 'TxGNN',
        exp_name = 'TxGNN'
    )
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

TxGNN1.pretrain(n_epoch = 1, 
               learning_rate = 1e-3,
               batch_size = 1024, 
               train_print_per_n = 20)

TxGNN1.finetune(n_epoch = 250, #---
                learning_rate = 5e-4,
                train_print_per_n = 5,
                valid_per_n = 20,)
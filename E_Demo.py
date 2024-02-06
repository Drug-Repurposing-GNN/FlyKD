from txgnn import TxData, TxGNN, TxEval
import time
import argparse
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument("--drop_dup", action="store_true")
parser.add_argument("--size", default=100, type=int)
# parser.add_argument("dropout", default=0)
args = parser.parse_args()

strt = time.time()
saving_path = './pre_trained_model_ckpt/'
# split = 'cell_proliferation'

TxData = TxData(data_folder_path = './data/')
TxData.prepare_split(split = 'complex_disease', seed = 1, no_kg = False, additional_train = pd.read_csv('psuedo_labels_15000.csv'))
# TxData.prepare_split(split=split, seed = 42, no_kg = False)

txGNN = TxGNN(
        data = TxData, 
        weight_bias_track = False,
        proj_name = 'TxGNN',
        exp_name = 'TxGNN',
    )

# to load a pretrained model: 
# saving_path = './model_ckpt'
# # TxGNN.load_pretrained(saving_path)

size = args.size
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

# ## here we did not run this, since the output is too long to fit into the notebook
# txGNN.pretrain(n_epoch = 1, ## was 2
#                learning_rate = 1e-3,
#                batch_size = 1024, 
#                train_print_per_n = 20)

# here as a demo, the n_epoch is set to 30. Change it to n_epoch = 500 when you use it
txGNN.finetune(n_epoch = 500,  ## could be set to 30 for speed
               learning_rate = 5e-4,
               train_print_per_n = 5,
               valid_per_n = 20)

# # TxGNN.save_model('./model_ckpt')
# TxGNN.save_model(saving_path+split)
# TxGNN.save_model(saving_path + 'mlp_1/')
print("Done training")
end = time.time()
print(end - strt)

# TxGNN.load_pretrained(saving_path)

# TxGNN.train_graphmask(relation = 'indication',
#                       learning_rate = 3e-4,
#                       allowance = 0.005,
#                     #   epochs_per_layer = 3,
#                       penalty_scaling = 1,
#                       epochs_per_layer = 5,
#                       valid_per_n = 20)

# output = TxGNN.retrieve_save_gates(saving_path)
# TxGNN.save_graphmask_model('./graphmask_model_ckpt')

# from txgnn import TxEval
# TxEval = TxEval(model = TxGNN)

# # evaluate individual diseases
# # result = TxEval.eval_disease_centric(disease_idxs = [12661.0, 11318.0], 
# #                                      relation = 'indication', 
# #                                      save_result = False)

# # evaluate the entire test set
# result = TxEval.eval_disease_centric(disease_idxs = 'test_set', 
#                                      show_plot = False, 
#                                      verbose = True, 
#                                      save_result = True,
#                                      return_raw = False)

# print("Done Evaluating!")
# TxEval.retrieve_disease_idxs_test_set('indication')



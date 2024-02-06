from txgnn import TxEval

from txgnn.NewDataHandler import NewDataHandler
from txgnn.TxGNNwNorm import TxGNNwNorm

datahandler = NewDataHandler()
datahandler.prepare_split(split = 'complex_disease', seed = 42)

TxGNNwBatchNorm = TxGNNwNorm(data = datahandler, 
              weight_bias_track = False,
              proj_name = 'TxGNN',
              exp_name = 'TxGNN'
              )
# TxGNNwBatchNorm.model_initialize(n_hid = 100, 
#                       n_inp = 100, 
#                       n_out = 100, 
#                       proto = True,
#                       proto_num = 3,
#                       attention = False,
#                       sim_measure = 'all_nodes_profile',
#                       bert_measure = 'disease_name',
#                       agg_measure = 'rarity',
#                       num_walks = 200,
#                       walk_mode = 'bit',
#                       path_length = 2)
# TxGNNwBatchNorm.save_model('models/pretrained_batchNorm_model_ckpt')

# TxGNNwBatchNorm.pretrain(n_epoch = 2, 
#                learning_rate = 1e-3,
#                batch_size = 1024, 
#                train_print_per_n = 20)

# TxGNNwBatchNorm.save_model('models/pretrained_batchNorm_model_ckpt')

# TxGNNwBatchNorm.finetune(n_epoch = 500, 
#                learning_rate = 5e-4,
#                       train_print_per_n = 5,
#                valid_per_n = 20,
#               #  save_name = finetune_result_path,
#               )

# TxGNNwBatchNorm.save_model('models/batchNorm_model_ckpt')

TxGNNwBatchNorm.load_pretrained('models/batchNorm_model_ckpt')

TxEval = TxEval(model = TxGNNwBatchNorm)

result = TxEval.eval_disease_centric(disease_idxs = 'test_set', 
                                     show_plot = False, 
                                     verbose = True, 
                                     save_result = False,
                                     return_raw = False)

print("Done Evaluating!")
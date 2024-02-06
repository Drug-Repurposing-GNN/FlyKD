from txgnn import TxEval, TxGNN
from txgnn.NewDataHandler import NewDataHandler
from txgnn.TxGNN import TxGNN

splits = [
#     'cell_proliferation',
#     'mental_health',
#     'cardiovascular',
    'anemia',
    'adrenal_gland',
]



def train_disease_split(split):
    assert split in splits
    
    datahandler = NewDataHandler()
    datahandler.prepare_split(split = split, seed = 42)
    
    model = TxGNN(data = datahandler, 
              weight_bias_track = False,
              proj_name = 'TxGNN',
              exp_name = 'TxGNN'
              )
    
    model.model_initialize(n_hid = 100, 
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
    
    model.pretrain(n_epoch = 1, #---
               learning_rate = 1e-3,
               batch_size = 1024, 
               train_print_per_n = 20)
    
    model.finetune(n_epoch = 500, #---
               learning_rate = 5e-4,
                      train_print_per_n = 5,
               valid_per_n = 20,
               save_name = 'finetuned_' + split)
    
    model.save_model(split)
    ev = TxEval(model = model)
    result = ev.eval_disease_centric(disease_idxs = 'test_set', 
                                     show_plot = False, 
                                     verbose = True, 
                                     save_result = True,
                                     return_raw = False,
                                     save_name = split + '_eval')
    return result
    
    
for split in splits:
    train_disease_split(split)
    
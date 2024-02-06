from txgnn import TxGNN, TxEval, TxData, NewDataHandler
from tqdm import tqdm
import time
def generate_eval_files(seed, split, save_name, loading_path=None):
    ## We need to prepare_split again because the notebook requires a evaluation file with certain seed?
    if split in ['disease_eval', 'cell_proliferation', 'mental_health', 'cardiovascular', 'anemia', 'adrenal_gland']:
        TxData1 = NewDataHandler(data_folder_path = './data/')
    else:
        TxData1 = TxData(data_folder_path = './data/')
    TxData1.prepare_split(split = split, seed = seed, no_kg = False)
    TxGNN1 = TxGNN(
            data = TxData1, 
            weight_bias_track = True,
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
    ## Train
    TxGNN1.pretrain(n_epoch = 1, #---
                    learning_rate = 1e-3,
                    batch_size = 1024, 
                    train_print_per_n = 20)
    TxGNN1.finetune(n_epoch = 500, #---
                    learning_rate = 5e-4,
                    train_print_per_n = 5,
                    valid_per_n = 20,)

    TxEval1 = TxEval(model = TxGNN1)
    # evaluate the entire test set
    result = TxEval1.eval_disease_centric(disease_idxs = 'test_set', 
                                        show_plot = False, 
                                        verbose = True, 
                                        save_result = True,
                                        return_raw = False,
                                        save_name=save_name)
    
    
# options: ['random', 'complex_disease', 'disease_eval', 'cell_proliferation', 'mental_health', 'cardiovascular', 'anemia', 'adrenal_gland', 'full_graph', 'downstream_pred']
splits = ['complex_disease']
# splits = ['random']
seed_list = [5]
# seed_list = [4, 5]
# seed_list = [4, 5]
strt = time.time()
for split in tqdm(splits):
    for seed in seed_list:
        save_name = 'data/TxGNN_'+ str(seed)+ '_' + split + '_eval'
        # loading_path = './pre_trained_model_ckpt/'
        generate_eval_files(seed, split, save_name)
print(time.time() - strt)
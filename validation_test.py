from txgnn import TxGNN, TxEval, TxData, NewDataHandler
from txgnn.utils import *
from tqdm import tqdm
import time

split = "complex_disease"
seed = 1
saving_path = './properly_pre_trained_model_ckpt/'
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, help="fname in properly_pre_trained_model_ckpt folder") #### Test #### How about without default=None, is this implied? 
args = parser.parse_args()
save_dir = f'{saving_path}{args.save_dir}'

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

# TxGNN1.pretrain(n_epoch = 1, ## was 2
#                learning_rate = 1e-3,
#                batch_size = 1024, 
#                train_print_per_n = 20)

# TxGNN1.finetune(n_epoch = 500, #---
#                 learning_rate = 5e-4,
#                 train_print_per_n = 5,
#                 valid_per_n = 20,)

# TxGNN1.save_model(save_dir)
TxGNN1.load_pretrained(save_dir)
G = TxGNN1.G.to(TxGNN1.device)
best_G = TxGNN1.best_G.to(TxGNN1.device)
best_model = TxGNN1.best_model.to(TxGNN1.device)
best_model.eval()
# print(TxGNN1.dd_etypes.device)
# TxGNN1.model = TxGNN1.model.to(TxGNN1.device)
# TxGNN1.best_model = TxGNN1.best_model.to(TxGNN1.device)
# print(TxGNN1.model)
# print("encore")

## Print out accuracies
neg_sampler = Full_Graph_NegSampler(TxGNN1.G, 1, 'fix_dst', TxGNN1.device)
g_train_neg = neg_sampler(TxGNN1.G)
g_valid_pos, g_valid_neg = TxGNN1.g_valid_pos, TxGNN1.g_valid_neg
g_test_pos, g_test_neg = TxGNN1.g_test_pos, TxGNN1.g_test_neg
## parameters: (model, G, train_pos, train_neg, val_pos, val_neg, test_pos, test_neg, device=None)
# accuracies_dict = evaluate_accuracy_per_split(best_model, best_G, g_train_neg, g_valid_pos, g_valid_neg, g_test_pos, g_test_neg)
# for k, v in accuracies_dict.items():
#     print(f"Accuracy for Positive {k} graph is {v[0].item()}")
#     print(f"Accuracy for Negative {k} graph is {v[1].item()}")

## How to extract the relation (if you so need it to create a df)
# with torch.no_grad():
#     ## obtain the distmult, add an argument so I obtain the corresponding edge relation as well. 
#     # h, beta_kl_loss, distmult = best_model(best_G, g_train_neg, pretrain_mode = False, mode = 'train', return_h_and_kl=True)
#     h, beta_kl_loss, distmult = best_model(best_G, pretrain_mode = False, mode = 'train', return_h_and_kl=True)
#     ## distmult params: (eval_G, G, h, pretrain_mode, mode = mode + '_pos', pseudo_training=pseudo_training)
#     pred_score_pos, out_pos = distmult(g_valid_pos, best_G, h, pretrain_mode=False, mode='train_pos')
#     pred_score_neg, out_neg = distmult(g_valid_neg, best_G, h, pretrain_mode=False, mode='train_pos')
#     pseudo_dd_etypes = [('drug', 'contraindication', 'disease'), 
#                         ('drug', 'indication', 'disease'), ]
#     srcdst_dict = {f"{k}_pos": [] for k in pseudo_dd_etypes}
#     srcdst_dict.update({f"{k}_neg": [] for k in pseudo_dd_etypes})
#     for k in pseudo_dd_etypes:
#         srcdst_dict[f"{k}_pos"] = pred_score_pos[f"{k}_srcdst"]
#         srcdst_dict[f"{k}_neg"] = pred_score_pos[f"{k}_srcdst"]
#         print(srcdst_dict[f"{k}_neg"])
#         break
    # pred_score_pos, pred_score_neg, pos_score, neg_score, _ = model(G, g_neg, g_pos, pretrain_mode = False, mode = 'mode')
    
#     pos_score = torch.cat([pred_score_pos[i] for i in dd_etypes])
#     neg_score = torch.cat([pred_score_neg[i] for i in dd_etypes])
    
#     scores = torch.sigmoid(torch.cat((pos_score, neg_score)).reshape(-1,))
#     # labels = [1] * len(pos_score) + [0] * len(neg_score)
#     labels = torch.cat((torch.ones(len(pos_score), device=device),
#                         torch.zeros(len(neg_score), device=device)))
#     loss = F.binary_cross_entropy(scores, labels)

## Testing LSP
LS_train = obtain_LS_matrix(best_model, G, G)
LS_valid = obtain_LS_matrix(best_model, g_valid_pos, G)
LS_test = obtain_LS_matrix(best_model, g_test_pos, G)
LS_dict = {"train": LS_train, "val": LS_valid, "test": LS_test}
import pickle
fname = "LSP.pkl"
with open(fname, "wb") as file:
    pickle.dump(LS_dict, file)

## printing auprc
# param: (best_model, g_valid_pos, g_valid_neg, g_test_pos, g_test_neg, G, dd_etypes, device)
# print_val_test_auprc(best_model, g_valid_pos, g_valid_neg, g_test_pos, g_test_neg, best_G, TxGNN1.dd_etypes, TxGNN1.device)

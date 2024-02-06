from txgnn import TxData, TxGNN, TxEval
from txgnn.model import *
from txgnn.utils import * 
import torch

TxData = TxData(data_folder_path = './data/')
TxData.prepare_split(split = 'complex_disease', seed = 42, no_kg = False)
G = TxData.G
model = HeteroRGCN( G,
            in_size=100,
            hidden_size=100,
            out_size=100,
            attention = False,
            proto = True,
            proto_num = 3,
            sim_measure = 'all_nodes_profile',
            bert_measure = 'disease_name', 
            agg_measure = 'agg_measure',
            num_walks = 200,
            walk_mode = 'bit',
            path_length = 2,
            split = 'complex_disease',
            data_folder = './data/',
            exp_lambda = 0.7,
            device = torch.device('cuda:0')
            ).to(torch.device('cuda:0'))
device = torch.device('cuda:0')
neg_sampler = Full_Graph_NegSampler(G, 1, 'fix_dst', device)
torch.nn.init.xavier_uniform(model.w_rels)

print(model.w_rels)
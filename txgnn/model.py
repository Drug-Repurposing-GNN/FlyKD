import dgl
from dgl.ops import edge_softmax
import math
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from torch.utils import data
import pandas as pd
import copy
import os
import random
import pickle
import time

import warnings
warnings.filterwarnings("ignore")
from .utils import sim_matrix, exponential, obtain_disease_profile, obtain_protein_random_walk_profile, convert2str
from .graphmask.multiple_inputs_layernorm_linear import MultipleInputsLayernormLinear
from .graphmask.squeezer import Squeezer
from .graphmask.sigmoid_penalty import SoftConcrete

class DistMultPredictor(nn.Module):
    def __init__(self, n_hid, w_rels, G, rel2idx, proto, proto_num, sim_measure, bert_measure, agg_measure, num_walks, walk_mode, path_length, split, data_folder, exp_lambda, device, seed):
        super().__init__()
        
        self.proto = proto
        self.sim_measure = sim_measure
        self.bert_measure = bert_measure
        self.agg_measure = agg_measure
        self.num_walks = num_walks
        self.walk_mode = walk_mode
        self.path_length = path_length
        self.exp_lambda = exp_lambda
        self.device = device
        self.W = w_rels
        self.rel2idx = rel2idx
        
        self.etypes_dd = [('drug', 'contraindication', 'disease'), 
                           ('drug', 'indication', 'disease'),
                           ('drug', 'off-label use', 'disease'),
                           ('disease', 'rev_contraindication', 'drug'), 
                           ('disease', 'rev_indication', 'drug'),
                           ('disease', 'rev_off-label use', 'drug')]
        self.restrained_dd_etypes = [('drug', 'contraindication', 'disease'), 
                        ('drug', 'indication', 'disease'), ]
        etypes_all = G.canonical_etypes ### Question: Does this always end up being all edge types? 
        
        self.disease_etypes_all = []
        self.wrev_disease_etypes_all = []

        for etype in etypes_all:
            src, relation, dst = etype
            if "rev" not in relation and (src == "disease" or dst == "disease"):
                self.disease_etypes_all.append(etype)
            if (src == "disease" or dst == "disease"):
                self.wrev_disease_etypes_all.append(etype)
        
        self.node_types_dd = ['disease', 'drug']
        
        if proto:
            self.W_gate = {}
            for i in self.node_types_dd:
                temp_w = nn.Linear(n_hid * 2, 1)
                nn.init.xavier_uniform_(temp_w.weight)
                self.W_gate[i] = temp_w.to(self.device)
            self.k = proto_num
            self.m = nn.Sigmoid()
                   
            if sim_measure in ['bert', 'profile+bert']:
                
                data_path = os.path.join(data_folder, 'kg.csv')
                        
                if os.path.exists(data_path):
                    df = pd.read_csv(data_path)
                                
                self.disease_dict = dict(df[df.x_type == 'disease'][['x_idx', 'x_id']].values)
                self.disease_dict.update(dict(df[df.y_type == 'disease'][['y_idx', 'y_id']].values))
                
                if bert_measure == 'disease_name':
                    self.bert_embed = np.load('/n/scratch3/users/k/kh278/bert_basic.npy')
                    df_nodes_bert = pd.read_csv('/n/scratch3/users/k/kh278/nodes.csv')
                    
                elif bert_measure == 'v1':
                    self.bert_embed = np.load('/n/scratch3/users/k/kh278/disease_embeds_single_def.npy')
                    df_nodes_bert = pd.read_csv('/n/scratch3/users/k/kh278/disease_nodes_for_BERT_embeds.csv')
                
                df_nodes_bert['node_id'] = df_nodes_bert.node_id.apply(lambda x: convert2str(x))
                self.id2bertindex = dict(zip(df_nodes_bert.node_id.values, df_nodes_bert.index.values))
                
            self.diseases_profile = {}
            self.sim_all_etypes = {}
            self.diseaseid2id_etypes = {}
            self.diseases_profile_etypes = {}
            
            disease_etypes = ['disease_disease', 'rev_disease_protein']
            disease_nodes = ['disease', 'gene/protein']
            
            ## Precompute all similarity 
            etypes = self.disease_etypes_all #if LSP_size else self.etypes_dd
            all_disease_ids = torch.arange(G.num_nodes("disease"))
            path = f"./data/{split}_{seed}"
            if os.path.exists(f"{path}/sim_all_etypes.pkl") and os.path.exists(f"{path}/diseaseid2id_etypes.pkl") and os.path.exists(f"{path}/diseases_profile_etypes.pkl"):
                with open(f"{path}/sim_all_etypes.pkl", "rb") as file:
                    self.sim_all_etypes = pickle.load(file)
                with open(f"{path}/diseaseid2id_etypes.pkl", "rb") as file2:
                    self.diseaseid2id_etypes = pickle.load(file2)
                with open(f"{path}/diseases_profile_etypes.pkl", "rb") as file3:
                    self.diseases_profile_etypes = pickle.load(file3) 
            else:
                print("precomputing the similarity matrix for fast disease pooling mechanism...")
                if sim_measure == 'all_nodes_profile':
                    diseases_profile = {i.item(): obtain_disease_profile(G, i, disease_etypes, disease_nodes) for i in all_disease_ids}
                elif sim_measure == 'protein_profile':
                    diseases_profile = {i.item(): obtain_disease_profile(G, i, ['rev_disease_protein'], ['gene/protein']) for i in all_disease_ids}
                elif sim_measure == 'protein_random_walk':
                    diseases_profile = {i.item(): obtain_protein_random_walk_profile(i, num_walks, path_length, G, disease_etypes, disease_nodes, walk_mode) for i in all_disease_ids}
                elif sim_measure == 'bert':
                    diseases_profile = {i.item(): torch.Tensor(self.bert_embed[self.id2bertindex[self.disease_dict[i.item()]]]) for i in all_disease_ids}
                elif sim_measure == 'profile+bert':
                    ## all profile dictionary where keys are disease_ids
                    diseases_profile = {i.item(): torch.cat((obtain_disease_profile(G, i, disease_etypes, disease_nodes), torch.Tensor(self.bert_embed[self.id2bertindex[self.disease_dict[i.item()]]]))) for i in all_disease_ids}

                ## sim matrix are the same for all etype (at least for the "all_nodes_profile" version)
                ## create new "fake ids" that are contiguous using "real_ids" 
                diseaseid2id = dict(zip(all_disease_ids.detach().cpu().numpy(), range(len(all_disease_ids))))
                disease_profile_tensor = torch.stack([diseases_profile[i.item()] for i in all_disease_ids])
                sim_all = sim_matrix(disease_profile_tensor, disease_profile_tensor)
                for etype in etypes:
                    print(f"storing similarity matrix based on edge type '{etype[1]}' between all disease ids")
                # for etype in self.etypes_dd:
                    src, dst = etype[0], etype[2]
                    if not (src == "disease" or dst == "disease"):
                        print("computing simlarity between two non-disease entities is pointless")
                        raise KeyError
                    self.sim_all_etypes[etype] = sim_all
                    self.diseaseid2id_etypes[etype] = diseaseid2id
                    self.diseases_profile_etypes[etype] = diseases_profile

                with open(f"{path}/sim_all_etypes.pkl", "wb") as file:
                    pickle.dump(self.sim_all_etypes, file)
                with open(f"{path}/diseaseid2id_etypes.pkl", "wb") as file2:
                    pickle.dump(self.diseaseid2id_etypes, file2)
                with open(f"{path}/diseases_profile_etypes.pkl", "wb") as file3:
                    pickle.dump(self.diseases_profile_etypes, file3)
                print("Done!")


    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        rel_idx = self.rel2idx[edges._etype]
        h_r = self.W[rel_idx]
        score = torch.sum(h_u * h_r * h_v, dim=1)
        return {'score': score}

    def compute_local_structure_vectors(self, graph, sparse_LS_list, etype, LSP, sigma):
        # h = graph.ndata['h']
        if LSP == "cosine":
            f_sim = torch.nn.CosineSimilarity(dim=1)
        elif LSP == "L2":
            f_sim = lambda src, dst: (src-dst).norm(p=2, dim=1)
        elif LSP == "Poly":
            f_sim = lambda src, dst: (torch.dot(src, dst) + 2)**2
        elif LSP == "RBF":
            assert sigma is not None
            f_sim = lambda src, dst: torch.exp(-((src - dst).norm(p=2, dim=1)**2) / (2 * sigma ** 2))
        else:
            raise KeyError

        def apply_f_sim(edges):
            result = {'sim_score': f_sim(edges.src['h'], edges.dst['h'])}
            return result ## useful for concatenation like [] + []

        graph.apply_edges(apply_f_sim, etype=etype)
        src_type, dst_type = etype[0], etype[2]
        non_dis = src_type if dst_type == "disease" else dst_type
        src, dst = graph.edges(etype=etype)
        edge_data = graph.edges[etype].data["sim_score"]#.to(self.device)

        ## Pretty sure you only compute LSP when graph = G
        if src_type == "disease":
            sparse_LS_list.append({'indices': torch.stack([src, dst]), 'values': edge_data, 'column_size': graph.num_nodes(dst_type)})
        else:
            sparse_LS_list.append({'indices': torch.stack([dst, src]), 'values': edge_data, 'column_size': graph.num_nodes(src_type)})

    ## pseudo_training disabled dpm
    def forward(self, graph, G, h, pretrain_mode, mode, block = None, only_relation = None, keep_grad_for_sl=False, LSP=False, LSP_size=None, sigma=None, verbose=False):
        with graph.local_scope():
            scores = {}
            s_l = []
            
            if len(graph.canonical_etypes) == 1:
                etypes_train = graph.canonical_etypes
            else:
                etypes_train = self.restrained_dd_etypes #### IMPORTANT #### RESTRAINING DD_RELATIONS (No-rev, No off-label)

            if only_relation is not None:
                if only_relation == 'indication':
                    etypes_train = [('drug', 'indication', 'disease'),
                                    ('disease', 'rev_indication', 'drug')]
                elif only_relation == 'contraindication':
                    etypes_train = [('drug', 'contraindication', 'disease'), 
                                   ('disease', 'rev_contraindication', 'drug')]
                elif only_relation == 'off-label':
                    etypes_train = [('drug', 'off-label use', 'disease'),
                                   ('disease', 'rev_off-label use', 'drug')]
                else:
                    return ValueError

            graph.ndata['h'] = h
            
            if pretrain_mode:
                # during pretraining....
                etypes_all = [i for i in graph.canonical_etypes if graph.edges(etype = i)[0].shape[0] != 0]
                for etype in etypes_all:
                    graph.apply_edges(self.apply_edges, etype=etype)    
                    out = torch.sigmoid(graph.edges[etype].data['score'])
                    s_l.append(out)
                    scores[etype] = out
            else:
                # finetuning on drug disease only...
                ## It seems like disease, disease_proteins are only used for "all_nodes_profile" DPM
                if LSP_size == "full":
                    etypes = self.disease_etypes_all
                elif LSP_size == "partial":
                    etypes = [("disease", 'disease_disease', "disease"),
                              ('gene/protein', 'disease_protein', 'disease')]
                else:
                    etypes = etypes_train

                if LSP:
                    ## Create a LS dictionary (w/ LS vectors) that will get updated every etype iteration
                    ## contains ingrediants to concatenate to create a final sparse LS tensor which will be compressed during loss computation.
                    sparse_LS_list = [] 
                for i, etype in enumerate(etypes):
                    if self.proto: 
                        src, dst = etype[0], etype[2]
                        # ## Eval g's embeddings in etype
                        src_rel_idx = torch.where(graph.out_degrees(etype=etype) != 0)
                        dst_rel_idx = torch.where(graph.in_degrees(etype=etype) != 0)
                        src_h = h[src][src_rel_idx] ## obtain node type's h and advance index eval_g's diseases
                        dst_h = h[dst][dst_rel_idx]

                        # ## Train Graph's embeddings in etype
                        src_rel_ids_keys = torch.where(G.out_degrees(etype=etype) != 0)
                        dst_rel_ids_keys = torch.where(G.in_degrees(etype=etype) != 0)
                        src_h_keys = h[src][src_rel_ids_keys]
                        dst_h_keys = h[dst][dst_rel_ids_keys]

                        h_disease = {}

                        if src == 'disease':
                            h_disease['disease_query'] = src_h
                            h_disease['disease_key'] = src_h_keys
                            h_disease['disease_query_id'] = src_rel_idx
                            h_disease['disease_key_id'] = src_rel_ids_keys
                        elif dst == 'disease':
                            h_disease['disease_query'] = dst_h
                            h_disease['disease_key'] = dst_h_keys
                            h_disease['disease_query_id'] = dst_rel_idx
                            h_disease['disease_key_id'] = dst_rel_ids_keys

                        if self.sim_measure in ['protein_profile', 'all_nodes_profile', 'protein_random_walk', 'bert', 'profile+bert']:
                            try:
                                ## Here you are trying to access the similary vector ??(only contains df_train's disease sim vectors)
                                sim = self.sim_all_etypes[etype][np.array([self.diseaseid2id_etypes[etype][i.item()] for i in h_disease['disease_query_id'][0]])]
                                ## Pruning of sim matrix where we do not consider nodes without this relation (prevents index error later)
                                sim = sim[:, h_disease['disease_key_id'][0].cpu()].to(self.device)
                                if verbose:
                                    print(f"Using stored similarity matrix for {etype}")
                            except:
                                print(f"Computing similarity matrix for out of distribution data for {etype}")
                                
                                disease_etypes = ['disease_disease', 'rev_disease_protein']
                                disease_nodes = ['disease', 'gene/protein']
            
                                ## new disease not seen in the training set
                                for i in h_disease['disease_query_id'][0]:
                                    if i.item() not in self.diseases_profile_etypes[etype]:
                                        if self.sim_measure == 'all_nodes_profile':
                                            self.diseases_profile_etypes[etype][i.item()] = obtain_disease_profile(G, i, disease_etypes, disease_nodes)
                                        elif self.sim_measure == 'protein_profile':
                                            self.diseases_profile_etypes[etype][i.item()] = obtain_disease_profile(G, i, ['rev_disease_protein'], ['gene/protein'])
                                        elif self.sim_measure == 'protein_random_walk':
                                            self.diseases_profile_etypes[etype][i.item()] = obtain_protein_random_walk_profile(i, self.num_walks, self.path_length, G, disease_etypes, disease_nodes, self.walk_mode)
                                        elif self.sim_measure == 'bert':
                                            self.diseases_profile_etypes[etype][i.item()] = torch.Tensor(self.bert_embed[self.id2bertindex[self.disease_dict[i.item()]]])
                                        elif self.sim_measure == 'profile+bert':
                                            self.diseases_profile_etypes[etype][i.item()] = torch.cat((obtain_disease_profile(G, i, disease_etypes, disease_nodes), torch.Tensor(self.bert_embed[self.id2bertindex[self.disease_dict[i.item()]]])))
                                            
                                profile_query = [self.diseases_profile_etypes[etype][i.item()] for i in h_disease['disease_query_id'][0]]
                                profile_query = torch.cat(profile_query).view(len(profile_query), -1)

                                profile_keys = [self.diseases_profile_etypes[etype][i.item()] for i in h_disease['disease_key_id'][0]]
                                profile_keys = torch.cat(profile_keys).view(len(profile_keys), -1)

                                sim = sim_matrix(profile_query, profile_keys)

                            # masking of whether eval_g's node is seen in the Train_G
                            mask = torch.isin(h_disease['disease_query_id'][0], h_disease['disease_key_id'][0])

                            ## any eval_g is in Train_G
                            coef = torch.zeros(sim.size(0), self.k, device=self.device)
                            embed = torch.zeros(h_disease['disease_query'].size(0), self.k, h_disease['disease_query'].size(1), device=self.device)
                            # seen_any = mask.nonzero().squeeze() > 0
                            seen = mask.nonzero().squeeze(dim=-1) ## what does squeeze() do here? 
                            if len(seen) > 0:
                                topk_values, topk_indices = torch.topk(sim[seen], self.k + 1, dim=1)
                                coef[seen, :] = F.normalize(topk_values[:, 1:], p=1, dim=1)
                                embed[seen] = h_disease['disease_key'][topk_indices[:, 1:]]
                            unseen = (~mask).nonzero().squeeze(dim=-1)
                            if len(unseen) > 0:
                                topk_values, topk_indices = torch.topk(sim[unseen], self.k, dim=1)
                                coef[unseen, :] = F.normalize(topk_values, p=1, dim=1)
                                embed[unseen] = h_disease['disease_key'][topk_indices]
                            out = torch.mul(embed, coef.unsqueeze(dim = 2).to(self.device)).sum(dim = 1)

                        if self.sim_measure in ['protein_profile', 'all_nodes_profile', 'protein_random_walk', 'bert', 'profile+bert']:
                            # for protein profile, we are only looking at diseases for now...
                            if self.agg_measure == 'learn':
                                coef_all = self.m(self.W_gate['disease'](torch.cat((h_disease['disease_query'], out), dim = 1)))
                                proto_emb = (1 - coef_all)*h_disease['disease_query'] + coef_all*out
                            elif self.agg_measure == 'heuristics-0.8':
                                proto_emb = 0.8*h_disease['disease_query'] + 0.2*out
                            elif self.agg_measure == 'avg':
                                proto_emb = 0.5*h_disease['disease_query'] + 0.5*out
                            elif self.agg_measure == 'rarity':
                                if src == 'disease':
                                    coef_all = exponential(G.out_degrees(etype=etype)[torch.where(graph.out_degrees(etype=etype) != 0)], self.exp_lambda).reshape(-1, 1)
                                elif dst == 'disease':
                                    coef_all = exponential(G.in_degrees(etype=etype)[torch.where(graph.in_degrees(etype=etype) != 0)], self.exp_lambda).reshape(-1, 1)
                                proto_emb = (1 - coef_all)*h_disease['disease_query'] + coef_all*out
                            elif self.agg_measure == '100proto':
                                proto_emb = out

                            h['disease'][h_disease['disease_query_id']] = proto_emb
                        else:
                            if self.agg_measure == 'learn':
                                coef_src = self.m(self.W_gate[src](torch.cat((src_h, sim_emb_src), dim = 1)))
                                coef_dst = self.m(self.W_gate[dst](torch.cat((dst_h, sim_emb_dst), dim = 1)))
                            elif self.agg_measure == 'rarity':
                                # give high weights to proto embeddings for nodes that have low degrees
                                coef_src = exponential(G.out_degrees(etype=etype)[torch.where(graph.out_degrees(etype=etype) != 0)], self.exp_lambda).reshape(-1, 1)
                                coef_dst = exponential(G.in_degrees(etype=etype)[torch.where(graph.in_degrees(etype=etype) != 0)], self.exp_lambda).reshape(-1, 1)
                            elif self.agg_measure == 'heuristics-0.8':
                                coef_src = 0.2
                                coef_dst = 0.2
                            elif self.agg_measure == 'avg':
                                coef_src = 0.5
                                coef_dst = 0.5
                            elif self.agg_measure == '100proto':
                                coef_src = 1
                                coef_dst = 1

                            proto_emb_src = (1 - coef_src)*src_h + coef_src*sim_emb_src
                            proto_emb_dst = (1 - coef_dst)*dst_h + coef_dst*sim_emb_dst

                            h[src][src_rel_idx] = proto_emb_src
                            h[dst][dst_rel_idx] = proto_emb_dst

                        graph.ndata['h'] = h

                    if LSP:
                        self.compute_local_structure_vectors(graph, sparse_LS_list, etype, LSP=LSP, sigma=sigma)
                    
                    if etype in self.restrained_dd_etypes:
                        graph.apply_edges(self.apply_edges, etype=etype)
                        out = graph.edges[etype].data['score']
                        srcdst = graph.edges(etype=etype[1])

                        s_l.append(out)
                        scores[etype] = out
                        scores[f"{etype}_srcdst"] = srcdst

                    if self.proto: 
                        # recover back to the original embeddings for other relations
                        h[src][src_rel_idx] = src_h
                        h[dst][dst_rel_idx] = dst_h

            if LSP:  
                strt = time.time()
                final_indices = []
                final_values = []
                indices_cursor = 0
                for LS_dict in sparse_LS_list:
                    indices, values, max_node = LS_dict["indices"], LS_dict["values"], LS_dict["column_size"], 
                    pad_indices = torch.stack([indices[0], indices[1]+indices_cursor])
                    final_indices.append(pad_indices)
                    final_values.append(values)
                    indices_cursor += max_node
                final_indices = torch.cat(final_indices, dim=-1)
                final_values = torch.cat(final_values)

                LS = []
                for i in range(G.num_nodes('disease')):
                    ## masking of row indices
                    mask = final_indices[0] == i
                    LS.append(final_values[mask])
                print(f"time it took to create the sparse tensor: {time.time() - strt}")
                return scores, s_l, LS
            
            if pretrain_mode or keep_grad_for_sl:
                if len(s_l) > 0:
                    s_l = torch.cat(s_l)           
                else:
                    s_l = []
            else: 
                s_l = torch.cat(s_l).reshape(-1,).detach().cpu().numpy()

            return scores, s_l

    
class AttHeteroRGCNLayer(nn.Module):

    def __init__(self, in_size, out_size, etypes):
        super(AttHeteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, out_size) for name in etypes
            })
        
        self.attn_fc = nn.ModuleDict({
                name : nn.Linear(out_size * 2, 1, bias = False) for name in etypes
            })
    
    def edge_attention(self, edges):
        src_type = edges._etype[0]
        etype = edges._etype[1]
        dst_type = edges._etype[2]
        
        if src_type == dst_type:
            wh2 = torch.cat([edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % etype]], dim=1)
        else:
            if etype[:3] == 'rev':
                wh2 = torch.cat([edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % etype[4:]]], dim=1)
            else:
                wh2 = torch.cat([edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % 'rev_' + etype]], dim=1)
        a = self.attn_fc[etype](wh2)
        return {'e_%s' % etype: F.leaky_relu(a)}

    def message_func(self, edges):
        etype = edges._etype[1]
        return {'m': edges.src['Wh_%s' % etype], 'e': edges.data['e_%s' % etype]}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['m'], dim=1)
        return {'h': h}
    
    def forward(self, G, feat_dict, return_att = False):
        with G.local_scope():        
            funcs = {}
            att = {}
            etypes_all = [i for i in G.canonical_etypes if G.edges(etype = i)[0].shape[0] != 0]
            for srctype, etype, dsttype in etypes_all:
                Wh = self.weight[etype](feat_dict[srctype])
                G.nodes[srctype].data['Wh_%s' % etype] = Wh

            for srctype, etype, dsttype in etypes_all:
                G.apply_edges(self.edge_attention, etype=etype)
                if return_att:
                    att[(srctype, etype, dsttype)] = G.edges[etype].data['e_%s' % etype].detach().cpu().numpy()
                funcs[etype] = (self.message_func, self.reduce_func)
                
            G.multi_update_all(funcs, 'sum')
            
            return {ntype : G.dstdata['h'][ntype] for ntype in list(G.dstdata['h'].keys())}, att
    
class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, out_size) for name in etypes
            })
        self.in_size = in_size
        self.out_size = out_size
            
        self.gate_storage = {}
        self.gate_score_storage = {}
        self.gate_penalty_storage = {}
    
    
    def add_graphmask_parameter(self, gate, baseline, layer):
        self.gate = gate
        self.baseline = baseline
        self.layer = layer
    
    def forward(self, G, feat_dict):
        funcs = {}
        etypes_all = [i for i in G.canonical_etypes if G.edges(etype = i)[0].shape[0] != 0]
        
        for srctype, etype, dsttype in etypes_all:
            Wh = self.weight[etype](feat_dict[srctype])
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        G.multi_update_all(funcs, 'sum')
       
        return {ntype : G.dstdata['h'][ntype] for ntype in list(G.dstdata['h'].keys())}
 
    def gm_online(self, edges):
        etype = edges._etype[1]
        srctype = edges._etype[0]
        dsttype = edges._etype[2]
        
        if srctype == dsttype:
            gate, penalty, gate_score, penalty_not_sum = self.gate[etype][self.layer]([edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % etype]])
        else:
            if etype[:3] == 'rev':                
                gate, penalty, gate_score, penalty_not_sum = self.gate[etype][self.layer]([edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % etype[4:]]])
            else:
                gate, penalty, gate_score, penalty_not_sum = self.gate[etype][self.layer]([edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % 'rev_' + etype]])
        self.penalty.append(penalty)
        
        self.num_masked += len(torch.where(gate.reshape(-1) != 1)[0])
        
        message = gate.unsqueeze(-1) * edges.src['Wh_%s' % etype] + (1 - gate.unsqueeze(-1)) * self.baseline[etype][self.layer].unsqueeze(0)
        
        if self.return_gates:
            self.gate_storage[etype] = copy.deepcopy(gate.to('cpu').detach())
            self.gate_penalty_storage[etype] = copy.deepcopy(penalty_not_sum.to('cpu').detach())
            self.gate_score_storage[etype] = copy.deepcopy(gate_score.to('cpu').detach())
        return {'m': message}
    
    def message_func_no_replace(self, edges):
        etype = edges._etype[1]
        return {'m': edges.src['Wh_%s' % etype]}
    
    def graphmask_forward(self, G, feat_dict, graphmask_mode, return_gates):
        self.return_gates = return_gates
        self.penalty = []
        self.num_masked = 0
        self.num_of_edges = G.number_of_edges()
        
        funcs = {}
        etypes_all = G.canonical_etypes
        
        for srctype, etype, dsttype in etypes_all:
            Wh = self.weight[etype](feat_dict[srctype])
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            
        for srctype, etype, dsttype in etypes_all:
            
            if graphmask_mode:
                ## replace the message!
                funcs[etype] = (self.gm_online, fn.mean('m', 'h'))
            else:
                ## normal propagation!
                funcs[etype] = (self.message_func_no_replace, fn.mean('m', 'h'))
                
        G.multi_update_all(funcs, 'sum')
        
        
        if graphmask_mode:
            self.penalty = torch.stack(self.penalty).reshape(-1,)
            penalty = torch.mean(self.penalty)
        else:
            penalty = 0 

        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}, penalty, self.num_masked
    
class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size, attention, proto, proto_num, sim_measure, bert_measure, agg_measure, num_walks, walk_mode, path_length, split, data_folder, exp_lambda, device, dropout=False, reparam_mode=False, seed=1):
        super(HeteroRGCN, self).__init__()

        if attention:
            self.layer1 = AttHeteroRGCNLayer(in_size, hidden_size, G.etypes)
            self.layer2 = AttHeteroRGCNLayer(hidden_size, out_size, G.etypes)
        else:
            self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes)
            self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes)

            
        self.w_rels = nn.Parameter(torch.Tensor(len(G.canonical_etypes), out_size))
        nn.init.xavier_uniform_(self.w_rels, gain=nn.init.calculate_gain('relu'))
        rel2idx = dict(zip(G.canonical_etypes, list(range(len(G.canonical_etypes)))))
               
        self.pred = DistMultPredictor(n_hid = hidden_size, w_rels = self.w_rels, G = G, rel2idx = rel2idx, proto = proto, proto_num = proto_num, sim_measure = sim_measure, bert_measure = bert_measure, agg_measure = agg_measure, num_walks = num_walks, 
                                      walk_mode = walk_mode, path_length = path_length, split = split, data_folder = data_folder, exp_lambda = exp_lambda, device = device, seed=seed)
        self.attention = attention
        
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.etypes = G.etypes

        self.dropout = dropout    
        self.total_nodes = G.num_nodes() ## Maybe try only the num of fine-tuning labels count?
        self.reparam_mode = reparam_mode
        if reparam_mode:
            if reparam_mode not in {"RMLP", "MPNN", "MLP"}:
                raise NameError
            if self.reparam_mode == "MPNN":
                self.layer3 = HeteroRGCNLayer(hidden_size, out_size, G.etypes)
            ## elif shared MLP reparmeterization:
            elif self.reparam_mode == "MLP":
                self.mlp_mean = nn.Sequential(
                    nn.Linear(out_size, 2*out_size),
                    nn.LeakyReLU(),
                    nn.Linear(2*out_size, out_size),
                )
                self.mlp_logvar = nn.Sequential(
                    nn.Linear(out_size, 2*out_size),
                    nn.LeakyReLU(),
                    nn.Linear(2*out_size, out_size),
                )
            # else relational MLP reparameterization:
            elif self.reparam_mode == "RMLP":
                self.mlp_mean = nn.ModuleDict({
                    name : nn.Sequential(
                        nn.Linear(out_size, 2*out_size),
                        nn.LeakyReLU(),
                        nn.Linear(2*out_size, out_size),
                    ) for name in G.ntypes
                    })
                self.mlp_logvar = nn.ModuleDict({
                    name : nn.Sequential(
                        nn.Linear(out_size, 2*out_size),
                        nn.LeakyReLU(),
                        nn.Linear(2*out_size, out_size),
                    ) for name in G.ntypes
                    })

    def extract_distmult(self,):
        return self.pred
                        
    def total_kl_loss(self, mu_dict=None, logstd_dict=None):
        total_kl_div = 0
        MAX_LOGSTD = 10
        for key in mu_dict.keys():
            mu = mu_dict[key].clone() ### solving gradient in-place operation error. Gradient still flows thorugh. 
            logstd = logstd_dict[key].clamp(max=MAX_LOGSTD)
            kl_div = - 0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
            total_kl_div += kl_div
        return total_kl_div
    
    def reparameterize(self, mean_dict, logvar_dict, train=False):
        if train: ## add noise
            print("Using reparamerization trick in training mode")
            z_dict = {}
            for ntype in mean_dict:
                std = torch.exp(0.5 * logvar_dict[ntype])
                eps = torch.randn_like(std)
                z_dict[ntype] = mean_dict[ntype] + std * eps
            return z_dict
        else:
            return mean_dict ## doing the same as above code? 
        
    def forward_minibatch(self, pos_G, neg_G, blocks, G, mode = 'train', pretrain_mode = False):
        input_dict = blocks[0].srcdata['inp']
        h_dict = self.layer1(blocks[0], input_dict)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        
        ## three versions of reparameterization. Only MPNN reparam utilizes layer 2 differently
        beta_kl_loss = 0
        if self.reparam_mode != "MPNN":
            h = self.layer2(blocks[1], h_dict)
        if self.reparam_mode: ## if string, it is True in Boolean
            mean_dict = {}
            logvar_dict = {}
            if self.reparam_mode == "MPNN":
                mean_dict = self.layer2(blocks[1], h_dict)
                logvar_dict = self.layer3(blocks[1], h_dict)
            elif self.reparam_mode == "MLP":
                for ntype in h:
                    mean_dict[ntype] = self.mlp_mean(h[ntype])
                    logvar_dict[ntype] = self.mlp_logvar(h[ntype])
            elif self.reparam_mode == "RMLP":
                for ntype in h:
                    mean_dict[ntype] = self.mlp_mean[ntype](h[ntype])
                    logvar_dict[ntype] = self.mlp_logvar[ntype](h[ntype])
            h = self.reparameterize(mean_dict, logvar_dict, self.training) ## pre-training doesn't do validation
            beta_kl_loss = self.total_kl_loss(mean_dict, logvar_dict) / self.total_nodes

        scores, out_pos = self.pred(pos_G, G, h, pretrain_mode, mode = mode + '_pos', block = blocks[1])
        scores_neg, out_neg = self.pred(neg_G, G, h, pretrain_mode, mode = mode + '_neg', block = blocks[1])
        return scores, scores_neg, out_pos, out_neg, beta_kl_loss
        
    
    def forward(self, G, neg_G = None, eval_pos_G = None, return_h = False, return_att = False, mode = 'train', pretrain_mode = False, pseudo_training=False, return_h_and_kl=False, return_all_layer_h=False):
        with G.local_scope():
            input_dict = {ntype : G.nodes[ntype].data['inp'] for ntype in G.ntypes}

            if self.attention:
                h_dict, a_dict_l1 = self.layer1(G, input_dict, return_att)
                h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
                h, a_dict_l2 = self.layer2(G, h_dict, return_att)
            else:
                h_dict = self.layer1(G, input_dict)
                if self.dropout != 0:
                    print(f"dropout being applied {self.dropout}")
                    h_dict = {k : F.dropout(F.leaky_relu(h), p=self.dropout) for k, h in h_dict.items()}
                else:
                    h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
                h_inter = {k:v.clone() for k, v in h_dict.items()}

                ## three versions of reparameterization. Only MPNN reparam utilizes layer 2 differently
                beta_kl_loss = 0
                if self.reparam_mode != "MPNN":
                    h = self.layer2(G, h_dict)
                if self.reparam_mode: ## if string, it is True in Boolean
                    mean_dict = {}
                    logvar_dict = {}
                    if self.reparam_mode == "MPNN":
                        mean_dict = self.layer2(G, h_dict)
                        logvar_dict = self.layer3(G, h_dict)
                    elif self.reparam_mode == "MLP":
                        for ntype in h:
                            mean_dict[ntype] = self.mlp_mean(h[ntype])
                            logvar_dict[ntype] = self.mlp_logvar(h[ntype])
                    elif self.reparam_mode == "RMLP":
                        for ntype in h:
                            mean_dict[ntype] = self.mlp_mean[ntype](h[ntype])
                            logvar_dict[ntype] = self.mlp_logvar[ntype](h[ntype])
                    h = self.reparameterize(mean_dict, logvar_dict, eval_pos_G is None and self.training) ## turned off during eval
                    beta_kl_loss = self.total_kl_loss(mean_dict, logvar_dict) / self.total_nodes
                    
            
            if return_all_layer_h and return_h_and_kl:
                return h_inter, h, beta_kl_loss, self.pred
            
            if return_h_and_kl:
                return h, beta_kl_loss, self.pred
            
            if return_h:
                return h

            if return_att:
                return a_dict_l1, a_dict_l2

            # full batch
            if eval_pos_G is not None:
                # eval mode
                scores, out_pos = self.pred(eval_pos_G, G, h, pretrain_mode, mode = mode + '_pos', keep_grad_for_sl=pseudo_training)
                scores_neg, out_neg = self.pred(neg_G, G, h, pretrain_mode, mode = mode + '_neg', keep_grad_for_sl=pseudo_training)
                return scores, scores_neg, out_pos, out_neg, beta_kl_loss
            else:
                scores, out_pos = self.pred(G, G, h, pretrain_mode, mode = mode + '_pos', keep_grad_for_sl=pseudo_training)
                scores_neg, out_neg = self.pred(neg_G, G, h, pretrain_mode, mode = mode + '_neg', keep_grad_for_sl=pseudo_training)
                return scores, scores_neg, out_pos, out_neg, beta_kl_loss
    
    def graphmask_forward(self, G, pos_graph, neg_graph, graphmask_mode = False, return_gates = False, only_relation = None):
        with G.local_scope():
            input_dict = {ntype : G.nodes[ntype].data['inp'] for ntype in G.ntypes}
            h_dict_l1, penalty_l1, num_masked_l1 = self.layer1.graphmask_forward(G, input_dict, graphmask_mode, return_gates)
            h_dict = {k : F.leaky_relu(h) for k, h in h_dict_l1.items()}
            h, penalty_l2, num_masked_l2 = self.layer2.graphmask_forward(G, h_dict, graphmask_mode, return_gates)         
            
            scores_pos, out_pos = self.pred(pos_graph, G, h, False, mode = 'train_pos', only_relation = only_relation)
            scores_neg, out_neg = self.pred(neg_graph, G, h, False, mode = 'train_neg', only_relation = only_relation)
            return scores_pos, scores_neg, penalty_l1 + penalty_l2, [num_masked_l1, num_masked_l2]

    def enable_layer(self, layer):
        print("Enabling layer "+str(layer))
        
        for name in self.etypes:
            for parameter in self.gates_all[name][layer].parameters():
                parameter.requires_grad = True

            self.baselines_all[name][layer].requires_grad = True
    
    def count_layers(self):
        return 2
    
    def get_gates(self):
        return [self.layer1.gate_storage, self.layer2.gate_storage]
    
    def get_gates_scores(self):
        return [self.layer1.gate_score_storage, self.layer2.gate_score_storage]
    
    def get_gates_penalties(self):
        return [self.layer1.gate_penalty_storage, self.layer2.gate_penalty_storage]
    
    def add_graphmask_parameters(self, G):
        gates_all, baselines_all = {}, {}
        hidden_size = self.hidden_size
        out_size = self.out_size
        
        for name in G.etypes:
            ## for each relation type
            gates = []
            baselines = []

            vertex_embedding_dims = [hidden_size, out_size]
            message_dims = [hidden_size, out_size]
            h_dims = message_dims

            for v_dim, m_dim, h_dim in zip(vertex_embedding_dims, message_dims, h_dims):
                gate_input_shape = [m_dim, m_dim]

                ### different layers have different gates
                gate = torch.nn.Sequential(
                    MultipleInputsLayernormLinear(gate_input_shape, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    Squeezer(),
                    SoftConcrete()
                )

                gates.append(gate)

                baseline = torch.FloatTensor(m_dim)
                stdv = 1. / math.sqrt(m_dim)
                baseline.uniform_(-stdv, stdv)
                baseline = torch.nn.Parameter(baseline, requires_grad=True)

                baselines.append(baseline)

            gates = torch.nn.ModuleList(gates)
            gates_all[name] = gates

            baselines = torch.nn.ParameterList(baselines)
            baselines_all[name] = baselines

        self.gates_all = nn.ModuleDict(gates_all)
        self.baselines_all = nn.ModuleDict(baselines_all)

        # Initially we cannot update any parameters. They should be enabled layerwise
        for parameter in self.parameters():
            parameter.requires_grad = False
            
        self.layer1.add_graphmask_parameter(self.gates_all, self.baselines_all, 0)
        self.layer2.add_graphmask_parameter(self.gates_all, self.baselines_all, 1)
import os
import math
import argparse
import copy
import pickle
from argparse import ArgumentParser
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import dgl
from dgl.data.utils import save_graphs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import time

from .model import *
from .utils import *

from .TxData import TxData
from .TxEval import TxEval

from .graphmask.moving_average import MovingAverage
from .graphmask.lagrangian_optimization import LagrangianOptimization

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(0)

class TxGNN:
    
    def __init__(self, data,
                       weight_bias_track = False,
                       proj_name = 'TxGNN',
                       exp_name = 'TxGNN',
                       device = 'cuda',
                       use_og = False,
                       T = 1):
        self.device = torch.device(device)
        self.weight_bias_track = weight_bias_track
        self.G = data.G
        self.df, self.df_train, self.df_valid, self.df_test = data.df, data.df_train, data.df_valid, data.df_test
        self.data = data
        self.data_folder = data.data_folder
        self.disease_eval_idx = data.disease_eval_idx
        self.split = data.split
        self.no_kg = data.no_kg
        self.g_pos_pseudo = data.g_pos_pseudo.to(device) if data.g_pos_pseudo is not None else None ## positive psuedo graph (ind, c-ind graph, w/ revs)
        self.g_dd_train = data.g_dd_train
        self.pseudo_dd_etypes = [('drug', 'contraindication', 'disease'), 
                                ('drug', 'indication', 'disease'), ]
        self.MAX_LSP_LEN = None
        self.T = T
        self.seed = data.seed
        self.use_og = use_og
        if data.soft_psuedo_logits_rel is not None:
            ## move to cuda
            soft_psuedo_logits_rel = {k: v.to(self.device) for k, v in data.soft_psuedo_logits_rel.items()}
            ## compute and store pseudo probabilities to use as labels
            soft_psuedo_logits = torch.cat([soft_psuedo_logits_rel[i] for i in self.pseudo_dd_etypes])
            self.soft_psuedo_prob = torch.sigmoid(soft_psuedo_logits / T)
        else:
            self.soft_psuedo_prob = None
        self.OC = self.df_train[self.df_train.relation == "contraindication"].shape[0]
        self.OI = self.df_train[self.df_train.relation == "indication"].shape[0]
        self.disease_rel_types = ['rev_contraindication', 'rev_indication', 'rev_off-label use']
        
        self.dd_etypes = [('drug', 'contraindication', 'disease'), 
                        ('drug', 'indication', 'disease'), 
                        ]
        if self.weight_bias_track:
            import wandb
            wandb.init(project=proj_name, name=exp_name)  
            self.wandb = wandb
        else:
            self.wandb = None
        self.config = None
        
    def model_initialize(self, n_hid = 128, 
                               n_inp = 128, 
                               n_out = 128, 
                               proto = True,
                               proto_num = 5,
                               attention = False,
                               sim_measure = 'all_nodes_profile',
                               bert_measure = 'disease_name',
                               agg_measure = 'rarity', 
                               exp_lambda = 0.7,
                               num_walks = 200,
                               walk_mode = 'bit',
                               path_length = 2,
                               dropout = False,
                               reparam_mode=False,
                               kl = False,
                               LSP = None,
                               LSP_size="full",
                               args=None):
        
        if self.no_kg and proto:
            print('Ablation study on No-KG. No proto learning is used...')
            proto = False
        self.kl = kl
        self.LSP = LSP
        self.LS_target = None
        if LSP is not None:
            if args is None:
                raise ValueError("args is missing")
            self.LSP = LSP
            self.LSP_size = LSP_size
            double = f"_double" if args.all_layers_LSP else ""
            LSP_fname = f"LSP_{args.teacher_size}{double}_{self.seed}_{LSP_size}_{LSP}.pt"
            print(f"Loading saved LSP tensors from: {LSP_fname}")
            self.LS_target = torch.load(LSP_fname)
        self.pretrain_scores_dict = args.pretrain_scores_dict if args is not None else None

        self.G = self.G.to('cpu')
        self.G = initialize_node_embedding(self.G, n_inp)
        self.g_valid_pos, self.g_valid_neg = evaluate_graph_construct(self.df_valid, self.G, 'fix_dst', 1, self.device)
        self.g_test_pos, self.g_test_neg = evaluate_graph_construct(self.df_test, self.G, 'fix_dst', 1, self.device)

        self.config = {'n_hid': n_hid, 
                    'n_inp': n_inp, 
                    'n_out': n_out, 
                    'proto': proto,
                    'proto_num': proto_num,
                    'attention': attention,
                    'sim_measure': sim_measure,
                    'bert_measure': bert_measure,
                    'agg_measure': agg_measure,
                    'num_walks': num_walks,
                    'walk_mode': walk_mode,
                    'path_length': path_length,
                    "dropout": dropout,
                    "reparam_mode": reparam_mode,
                    "kl": kl,
                    "LSP": LSP,
                    "LSP_size": "LSP_size",
                    "args": args
                    }
        self.model = HeteroRGCN(self.G,
                in_size=n_inp,
                hidden_size=n_hid,
                out_size=n_out,
                attention = attention,
                proto = proto,
                proto_num = proto_num,
                sim_measure = sim_measure,
                bert_measure = bert_measure, 
                agg_measure = agg_measure,
                num_walks = num_walks,
                walk_mode = walk_mode,
                path_length = path_length,
                split = self.split,
                data_folder = self.data_folder,
                exp_lambda = exp_lambda,
                device = self.device,
                dropout=dropout,
                reparam_mode=reparam_mode,
                seed = self.seed,
                ).to(self.device)    
        self.best_model = self.model
        self.best_G = self.G ## to store best validation's Graph Embedding
        self.teacher_model = None
        self.teacher_G = None
        self.args = args
        if args is not None and args.on_the_fly_KD and args.i != 0:
            teacher_txgnn = TxGNN(
                data = self.data, 
            )
            teacher_txgnn.load_pretrained(args.prev_trained_dir)
            self.teacher_G = teacher_txgnn.best_G
            self.teacher_model = teacher_txgnn.best_model
            self.teacher_model.eval()
        
    def print_model_size(self,):
        if self.teacher_model is not None:
            print(f"#param teacher model: {sum([p.numel() for p in self.teacher_model.parameters()])}")
        print('#Param student model: %d' % (get_n_params(self.model)))
    
    def pretrain(self, n_epoch = 1, learning_rate = 1e-3, batch_size = 1024, train_print_per_n = 20, sweep_wandb = None):
        
        if self.no_kg:
            raise ValueError('During No-KG ablation, pretraining is infeasible because it is the same as finetuning...')
            
        self.G = self.G.to('cpu')
        print('Creating minibatch pretraining dataloader...')
        train_eid_dict = {etype: self.G.edges(form = 'eid', etype =  etype) for etype in self.G.canonical_etypes}
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        ## inject pseudo labels as edge data
        self.pretrain_G = copy.deepcopy(self.G)
        if self.pretrain_scores_dict is not None: 
            for full_etype in self.pretrain_G.canonical_etypes:
                etype = full_etype[1]
                self.pretrain_G.edges[etype].data["score"] = self.pretrain_scores_dict[full_etype].to(torch.device("cpu")).detach()
        dataloader = dgl.dataloading.EdgeDataLoader(
            self.pretrain_G, train_eid_dict, sampler,
            negative_sampler=Minibatch_NegSampler(self.G, 1, 'fix_dst'),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=2)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = learning_rate)

        print('Start pre-training with #param: %d' % (get_n_params(self.model)))

        for epoch in range(n_epoch):
            self.model.train()
            strt = time.time()
            for step, (nodes, pos_g, neg_g, blocks) in enumerate(dataloader):

                blocks = [i.to(self.device) for i in blocks]
                pos_g = pos_g.to(self.device)
                neg_g = neg_g.to(self.device)

                pred_score_pos, pred_score_neg, pos_score, neg_score, beta_kl_loss = self.model.forward_minibatch(pos_g, neg_g, blocks, self.G, mode = 'train', pretrain_mode = True)

                scores = torch.cat((pos_score, neg_score)).reshape(-1,)
                labels = [1] * len(pos_score) + [0] * len(neg_score)
                loss = F.binary_cross_entropy(scores, torch.Tensor(labels).float().to(self.device)) + beta_kl_loss

                if self.pretrain_scores_dict is not None:
                    pseudo_labels_pretrain = pos_g.edata["score"]
                    non_empty_pseudo_labels_pretrain = {k: v for k, v in pseudo_labels_pretrain.items() if v.numel() > 0}
                    for (k1, pl), (k2, ps) in zip(non_empty_pseudo_labels_pretrain.items(), pred_score_pos.items()):
                        ## checking dimensions
                        assert pl.shape == ps.shape, f"pseudo label and scores have different values {k1}: {pl.shape}, {k2}: {ps.shape}. Perhaps, the order is messed up?"
                    pseudo_labels_pretrain = torch.cat([v for v in pseudo_labels_pretrain.values()])
                    pseudo_loss_pretrain = F.binary_cross_entropy(pos_score, pseudo_labels_pretrain.float().to(self.device))
                    print(f"regular_loss: {loss}, pseudo_loss: {pseudo_loss_pretrain}")
                    loss = loss * 0.05 + pseudo_loss_pretrain 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.weight_bias_track:
                    self.wandb.log({"Pretraining Loss": loss})

                if step % train_print_per_n == 0:
                    # pretraining tracking...
                    self.model.eval()
                    auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc = get_all_metrics_fb(pred_score_pos, pred_score_neg, scores.reshape(-1,).detach().cpu().numpy(), labels, self.G, True)
                    
                    if self.weight_bias_track:
                        temp_d = get_wandb_log_dict(auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc, "Pretraining")
                        temp_d.update({"Pretraining LR": optimizer.param_groups[0]['lr']})
                        self.wandb.log(temp_d)
                    
                    
                    if sweep_wandb is not None:
                        sweep_wandb.log({'pretraining_loss': loss, 
                                  'pretraining_micro_auroc': micro_auroc,
                                  'pretraining_macro_auroc': macro_auroc,
                                  'pretraining_micro_auprc': micro_auprc, 
                                  'pretraining_macro_auprc': macro_auprc})
                    print(time.time() - strt)
                    strt = time.time()
                    print('Epoch: %d Step: %d LR: %.5f Loss %.4f, Pretrain Micro AUROC %.4f Pretrain Micro AUPRC %.4f Pretrain Macro AUROC %.4f Pretrain Macro AUPRC %.4f' % (
                        epoch,
                        step,
                        optimizer.param_groups[0]['lr'], 
                        loss.item(),
                        micro_auroc,
                        micro_auprc,
                        macro_auroc,
                        macro_auprc
                    ))
        self.best_model = copy.deepcopy(self.model)
        del self.pretrain_G
        
    def finetune(self, n_epoch = 500, 
                       learning_rate = 1e-3, 
                       train_print_per_n = 5, 
                       valid_per_n = 25,
                       sweep_wandb = None,
                       save_name = None,
                       weight_decay = 0,
                       no_dpm = False,
                       args=None):
        best_val_acc = 0

        self.G = self.G.to(self.device)
        
        # neg_sampler = Full_Graph_NegSampler(self.G, 1, 'fix_dst', self.device)
        neg_sampler = Full_Graph_NegSampler(self.g_dd_train, 1, 'fix_dst', self.device)
        if args.neg_pseudo_sampling:
            pseudo_neg_sampler = Full_Graph_NegSampler(self.g_pos_pseudo, 1, 'fix_dst', self.device)
        elif args.limited_neg_pseudo_sampling:
            pseudo_neg_sampler = Full_Graph_NegSampler(self.g_dd_train, 1, 'fix_dst', self.device)
        if args.strong_scores is not None:
            if args.fly_no_val_test:
                fly_kd_neg_sampler = Full_Graph_NegSampler(self.G, 1, 'fix_dst', self.device)
            else:
                fly_kd_neg_sampler = Full_Graph_NegSampler(self.data.full_G, 1, 'fix_dst', self.device)
            # fly_kd_neg_sampler = Full_Graph_NegSampler(g_on_the_fly, 1, 'fix_dst', self.device)
        torch.nn.init.xavier_uniform(self.model.w_rels) # reinitialize decoder
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.90)
        
        strt = time.time()
        for epoch in range(n_epoch):
            self.model.train()
            verbose = epoch <= 20
            if epoch == 0:
                print("first epoch initiating")
            negative_graph = neg_sampler(self.g_dd_train)

            ## extracting h to use in the following auxillary loss for psuedo labels
            strt = time.time()
            if args.all_layers_LSP:
                h_inter, h, beta_kl_loss, distmult = self.model(self.G, pretrain_mode = False, mode = 'train', return_h_and_kl=True, return_all_layer_h=True) 
            else:
                h, beta_kl_loss, distmult = self.model(self.G, pretrain_mode = False, mode = 'train', return_h_and_kl=True) 
            
            pred_score_pos, _ = distmult(self.G, self.G, h, pretrain_mode=False, mode='train_pos', verbose=verbose) ## Mystery No.2, why did psuedo=True cause error? 
            pred_score_neg, _ = distmult(negative_graph, self.G, h, pretrain_mode=False, mode='train_neg', verbose=verbose)

            pos_score = torch.cat([pred_score_pos[i] for i in self.dd_etypes])
            neg_score = torch.cat([pred_score_neg[i] for i in self.dd_etypes])

            scores = torch.sigmoid(torch.cat((pos_score, neg_score)).reshape(-1,))
            labels = [1] * len(pos_score) + [0] * len(neg_score)
            loss = F.binary_cross_entropy(scores, torch.Tensor(labels).float().to(self.device)) #+ beta_kl_loss

            Total_LSP_loss = 0
            if self.LS_target is not None:
                ## kg_all is used to compute the LS vector of all label pairs but only using the train split to compute the LS vector
                if args.all_layers_LSP:
                    h_layers = [h_inter, h]
                else:
                    h_layers = [h]
                if len(self.LS_target) == self.G.num_nodes("disease"):
                    self.LS_target = [self.LS_target]
                LSP_strt = time.time()
                for i, (h_layer, LS_target, sigma) in enumerate(zip(h_layers, self.LS_target, args.sigmas)):
                    _, _, LS_pred = distmult(self.G, self.G, h_layer, mode=None, pretrain_mode=False, LSP=self.LSP, LSP_size=self.LSP_size, sigma=sigma, verbose=verbose)

                    LSP_loss = 0
                    if epoch == 0:
                        count = 0
                        max_len = 0
                        for x in LS_pred:
                            new_len = len(x)
                            max_len = max_len if max_len >= new_len else new_len
                            if x.numel() > 1:
                                count += 1
                        self.MAX_LSP_LEN = max_len
                        count2 = 0
                        for x in LS_target:
                            if x.numel() > 1:
                                count2 += 1
                        assert count == count2
                        
                    pred_probs = []
                    target_probs = []
                    for LS_i, LS_target_i in zip(LS_pred, LS_target):
                        if len(LS_i) > 0:
                            ## softmax LS_i
                            pred_prob = F.softmax(LS_i)
                            ## Padding:
                            to_pad = (0, self.MAX_LSP_LEN - len(pred_prob))
                            padded_pred_prob = F.pad(pred_prob, to_pad, mode="constant", value=0)
                            padded_target_prob = F.pad(LS_target_i, to_pad, mode="constant", value=0)
                            pred_probs.append(padded_pred_prob)
                            target_probs.append(padded_target_prob)
                    stacked_pred_probs = torch.stack(pred_probs)
                    stacked_target_probs = torch.stack(target_probs)
                    mask = stacked_target_probs != 0
                    eps = 1e-8
                    LSP_loss = (torch.where(mask, stacked_target_probs * ((stacked_target_probs+eps).log() - (stacked_pred_probs+eps).log()), torch.tensor(0)).sum(dim=-1) / mask.sum(dim=-1)).sum()
                    print(f"{i} layer LSP loss: {LSP_loss}")
                    ## Lambda is incorporated below
                    Total_LSP_loss += 500 * LSP_loss / len(LS_pred)
                # print(f"num LSP layers: {len(h_layers)}")
                Total_LSP_loss /= len(h_layers) ## normalize LSP loss
                print(f"Total LSP loss is: {Total_LSP_loss}. LSP loss computation time taken: {time.time() - LSP_strt}")

            if self.g_pos_pseudo is not None or self.teacher_G is not None:
                ## second loss with psuedo labels
                if self.teacher_G is not None:
                    ## no augmentation of random graph if fixed enable
                    if (not args.fixed_flyKD or epoch==0) and (not args.occasional_flyKD or epoch % args.occasional_flyKD == 0):
                        del self.soft_psuedo_prob
                        k = args.random_pseudo_k
                        num_drug = self.G.num_nodes("drug")
                        num_disease = self.G.num_nodes("disease")
                        train_out = copy.deepcopy(self.data.dd_train_out)
                        is_train_mask_lst = []
                        pseudo_etype_len = {}
                        for etype, (src, dst) in train_out.items():
                            ## generate src dst pairs for random graph
                            if args.rel_multinomial_ptrain:
                                dis_idx = self.data.dis_rel_idx_ptrain[etype[1]].index
                                drug_idx = self.data.drug_rel_idx_ptrain[etype[1]].index
                                dis_idx_degree = self.data.dis_rel_idx_ptrain[etype[1]].values
                                drug_idx_degree = self.data.drug_rel_idx_ptrain[etype[1]].values
                                pseudo_src = np.random.choice(drug_idx, k, replace=True, p=drug_idx_degree)
                                pseudo_dst = np.random.choice(dis_idx, k, replace=True, p=dis_idx_degree)
                                if verbose:
                                    print(f"using rel_ptrain for {etype[1]} where num drugs_idx: {len(pseudo_src)}, num dis_idx: {len(pseudo_dst)} where\
sampled num drugs: {len(pseudo_src)}, sampled num dis: {len(pseudo_dst)}")
                            elif args.multinomial_ptrain:
                                dis_idx = self.data.dis_idx_ptrain.index
                                drug_idx = self.data.drug_idx_ptrain.index
                                dis_idx_degree = self.data.dis_idx_ptrain.values
                                drug_idx_degree = self.data.drug_idx_ptrain.values
                                if args.modified_multinomial:
                                    if verbose:
                                        print(f"using modified multinomial")
                                    iter_ = 1.5
                                    dis_idx_degree = np.power(dis_idx_degree, iter_)
                                    drug_idx_degree = np.power(drug_idx_degree, iter_)
                                    probs = np.power(probs, 1.5)
                                    dis_idx_degree /= dis_idx_degree.sum()
                                    drug_idx_degree /= drug_idx_degree.sum()
                                pseudo_src = np.random.choice(drug_idx, k, replace=True, p=drug_idx_degree)
                                pseudo_dst = np.random.choice(dis_idx, k, replace=True, p=dis_idx_degree)
                                if verbose:
                                    print(f"using multionomial_ptrain for {etype[1]} where num drugs_idx: {len(pseudo_src)}, num dis_idx: {len(pseudo_dst)} where\
sampled num drugs: {len(pseudo_src)}, sampled num dis: {len(pseudo_dst)}")
                            else:
                                pseudo_srcs = []
                                if args.ptrain:
                                    drug_idx = self.data.drug_idx_ptrain.index
                                    dis_idx = self.data.dis_idx_ptrain.index
                                elif args.rel_ptrain:
                                    ## relation specific ptrain constraint
                                    dis_idx = self.data.dis_rel_idx_ptrain[etype[1]].index
                                    drug_idx = self.data.drug_rel_idx_ptrain[etype[1]].index
                                    if verbose:
                                        print(f"using rel_ptrain for {etype[1]} where num drugs: {len(drug_idx)}, num dis: {len(dis_idx)}")
                                else:
                                    dis_idx = self.data.dis_idx_wout_val_test.numpy() if args.fly_no_val_test else np.arange(num_disease)
                                for _ in dis_idx:
                                    if args.ptrain:
                                        sampled_drug_idx = np.random.choice(drug_idx, k, replace=False)
                                        pseudo_srcs.append(sampled_drug_idx)
                                    else:
                                        pseudo_srcs.append(np.random.randint(0, num_drug, (k,)))
                                pseudo_src = np.concatenate(pseudo_srcs)
                                pseudo_dst = dis_idx.repeat(k)

                            ## Track what was train data and what was generated pseudo
                            append_is_train_mask = torch.cat([torch.ones(len(src)), torch.zeros(len(pseudo_src))]).int()
                            pseudo_etype_len[etype] = len(append_is_train_mask)
                            is_train_mask_lst.append(append_is_train_mask)
                            ## Keep the training pseudo scores
                            if args.no_ptrain_flyKD:
                                print(f"The size difference of not having pseudo train is {np.concatenate([src, pseudo_src]).shape[0] - pseudo_src.shape[0]}")
                                src = pseudo_src
                                dst = pseudo_dst
                            else:
                                src = np.concatenate([src, pseudo_src])
                                dst = np.concatenate([dst, pseudo_dst])
                            train_out[etype] = (src, dst)
                            assert pseudo_src.shape == pseudo_dst.shape
                        g_on_the_fly = dgl.heterograph(train_out, num_nodes_dict = {ntype: self.G.number_of_nodes(ntype) for ntype in self.G.ntypes}).to(self.device)
                    else:
                        if args.occasional_flyKD:
                            print(f"Occasional_flyKD counter: {epoch % args.occasional_flyKD}")
                        else:
                            print(f"Fixed flyKD mode enabled: Using the same g_on_the_fly graph")
                    ## compute positive scores now
                    pseudo_rels1, _ = distmult(g_on_the_fly, self.teacher_G, h, mode=None, pretrain_mode=False, keep_grad_for_sl=True, verbose=verbose)
                    pos_score = torch.cat([pseudo_rels1[etype] for etype in train_out.keys()])
                    pseudo_pos_scores = pos_score.clone()
                    is_train_mask = torch.cat(is_train_mask_lst).bool()

                    ## run teacher model to generate pseudo labels on the fly
                    with torch.no_grad():
                        h, _, distmult = self.teacher_model(self.teacher_G, pretrain_mode = False, mode = 'train', return_h_and_kl=True) 
                        pseudo_rels2, _ = distmult(g_on_the_fly, self.teacher_G, h, mode=None, pretrain_mode=False, keep_grad_for_sl=True, verbose=verbose)
                        pos_score = torch.cat([pseudo_rels2[i] for i in self.pseudo_dd_etypes])

                        if args.strong_scores is not None:
                            constraint = args.strong_scores
                            confidence_mask = pos_score > constraint
                            ## Keep the real train dataset
                            confidence_mask[is_train_mask] = True
                            self.soft_psuedo_prob = torch.sigmoid(pos_score[confidence_mask]) ## label
                            pseudo_pos_scores = pseudo_pos_scores[confidence_mask] ## score
                            ## get the train_mask of updated strong scores. 
                            dummy_confidence_mask = torch.empty(len(confidence_mask)).long()
                            dummy_confidence_mask[confidence_mask] = torch.arange(len(pseudo_pos_scores)) ## number the True values
                            train_idx = dummy_confidence_mask[is_train_mask] ## Obtain the idx correponsidng to train mask
                            is_train_mask = torch.zeros(len(pseudo_pos_scores)).bool()
                            is_train_mask[train_idx] = True
                        else:
                            self.soft_psuedo_prob = torch.sigmoid(pos_score)
                        if verbose:
                            print(f"Total pseudo length including train: {len(self.soft_psuedo_prob)}")
                            
                        # generate negative sampling for strong pseudo labels
                        if args.strong_scores is not None:
                            masked_train_out = {}
                            confidence_mask = confidence_mask.cpu()
                            for i, etype in enumerate(self.pseudo_dd_etypes):
                                if i==0:
                                    etype_len = pseudo_etype_len[etype]
                                    masked_src = train_out[etype][0][confidence_mask[:etype_len]]
                                    masked_dst = train_out[etype][1][confidence_mask[:etype_len]]
                                else:
                                    etype_len = pseudo_etype_len[etype]
                                    masked_src = train_out[etype][0][confidence_mask[-etype_len:]]
                                    masked_dst = train_out[etype][1][confidence_mask[-etype_len:]]                                
                                masked_train_out[etype] = (masked_src, masked_dst)
                            g_masked_on_the_fly = dgl.heterograph(masked_train_out, num_nodes_dict = {ntype: self.G.number_of_nodes(ntype) for ntype in self.G.ntypes}).to(self.device)
                            assert pseudo_rels1.keys() == pseudo_rels2.keys()
                else:
                    pseudo_pos_scores_rel, _ = distmult(self.g_pos_pseudo, self.G, h, mode=None, pretrain_mode=False, keep_grad_for_sl=True, verbose=verbose)
                    ## psuedo skips on off-label dd relation. 
                    pos_scores_rel_lst = [pseudo_pos_scores_rel[i] for i in self.pseudo_dd_etypes]
                    pseudo_pos_scores = torch.cat(pos_scores_rel_lst)
                    if verbose:                    
                        print("calling positive graph for pseudo")
                        print(f"number of labels in pseudo graph: {len(pseudo_pos_scores)}")
                    
                if args.limited_neg_pseudo_sampling or args.neg_pseudo_sampling:
                    if args.limited_neg_pseudo_sampling:
                        g_neg_pseudo = pseudo_neg_sampler(self.g_dd_train) 
                    elif args.neg_pseudo_sampling:
                        g_neg_pseudo = pseudo_neg_sampler(self.g_pos_pseudo)
                    if verbose:
                        if args.on_the_fly_KD:
                            print(f"train length: {is_train_mask.float().sum()}")
                        print("calling negative graph for pseudo")
                    pseudo_neg_scores_rel, _ = distmult(g_neg_pseudo, self.G, h, mode=None, pretrain_mode=False, keep_grad_for_sl=True, verbose=verbose)
                    pseudo_neg_scores = torch.cat([pseudo_neg_scores_rel[i] for i in self.pseudo_dd_etypes])
                    if args.on_the_fly_KD:
                        is_train_mask = torch.cat([is_train_mask, torch.ones(pseudo_neg_scores.shape).bool()])
                    if not args.curriculum3 or 1200 <= epoch:
                        pseudo_labels = torch.cat([self.soft_psuedo_prob, torch.zeros(pseudo_neg_scores.shape, device=self.device)]) 
                        pseudo_scores = torch.cat([pseudo_pos_scores, pseudo_neg_scores])
                    else:
                        pseudo_scores = pseudo_pos_scores
                        pseudo_labels = self.soft_psuedo_prob
                else:
                    pseudo_scores = pseudo_pos_scores
                    pseudo_labels = self.soft_psuedo_prob
                if args.balance_loss:
                    ## C + I = T where these represents number of relations, cind, ind, total respectively. 
                    ## n * C / I = OC / OI we want to multioply n so that the relation ratio match up. n is the multiplier, OC, OI are original cind, ind.
                    ## T / (n * c + I) * (n * C + I) = T, rescale the entire loss value to maintain the magnitude of the loss the same by T / (n * c + I)
                    C = len(pos_scores_rel_lst[0])
                    I = len(pos_scores_rel_lst[1])
                    T = C + I
                    n = self.OC / self.OI * I / C ## cind_reweight_term
                    rescale_back = T / (n * C + I)
                    print(f"ratios C:I {C/I}, ratios OC: OI {self.OC/self.OI}")
                    print(f"rescaling C by {n} then scale back everything with {rescale_back}")
                    weight_tensor = torch.ones(pseudo_labels.shape)
                    weight_tensor[:C] *= n
                    weight_tensor[T: T+C] *= n
                    weight_tensor = weight_tensor.to(self.device)
                    pseudo_loss = F.binary_cross_entropy(torch.sigmoid(pseudo_scores), pseudo_labels, weight=weight_tensor)
                    pseudo_loss *= rescale_back
                    pseudo_loss += beta_kl_loss
                elif args.no_curriculum:
                    pseudo_train_loss = F.binary_cross_entropy(torch.sigmoid(pseudo_scores[is_train_mask]), pseudo_labels[is_train_mask])
                    pseudo_random_loss = F.binary_cross_entropy(torch.sigmoid(pseudo_scores[~is_train_mask]), pseudo_labels[~is_train_mask])
                    pseudo_loss = 0.05 * loss + pseudo_train_loss + pseudo_random_loss
                    pseudo_loss /= 1 + 1 + 0.05
                    if verbose:
                        print("using no curriculum formula")
                        print(f"loss: {loss}, pseudo train loss: {pseudo_train_loss}, pseudo_random_loss: {pseudo_random_loss}")
                elif args.curriculum1:
                    assert args.on_the_fly_KD
                    pseudo_train_loss = F.binary_cross_entropy(torch.sigmoid(pseudo_scores[is_train_mask]), pseudo_labels[is_train_mask])
                    pseudo_random_loss = F.binary_cross_entropy(torch.sigmoid(pseudo_scores[~is_train_mask]), pseudo_labels[~is_train_mask])
                    lambda_og = lambda epoch: 1 if epoch <= 400 else 1.95 - .95/400 * epoch if epoch <= 800 else 0.05
                    lambda_pseudo_train = lambda epoch: 0 if epoch <= 400 else -1 + epoch/400 if epoch <= 800 else 1
                    lambda_pseudo_random = lambda epoch: 0 if epoch <= 800 else -2 + epoch/400 if epoch <= 1200 else 1
                    pseudo_loss = lambda_og(epoch)*loss + lambda_pseudo_train(epoch)*pseudo_train_loss + lambda_pseudo_random(epoch)*pseudo_random_loss
                    normalizer = lambda_og(epoch) + lambda_pseudo_train(epoch) + lambda_pseudo_random(epoch)
                    pseudo_loss /= normalizer
                    if verbose:
                        print(f"Using curriculum formula 1: L_og={lambda_og(epoch)/normalizer}, L_tr={lambda_pseudo_train(epoch)/normalizer}, \
L_r={lambda_pseudo_random(epoch)/normalizer}")
                elif args.curriculum2:
                    pseudo_train_loss = F.binary_cross_entropy(torch.sigmoid(pseudo_scores), pseudo_labels)
                    lambda_og = lambda epoch: 1 if epoch <= 600 else 1.95 - .95/600 * epoch if epoch <= 1200 else 0.05
                    lambda_pseudo_train = lambda epoch: 0 if epoch <= 600 else -1 + epoch/600 if epoch <= 1200 else 1
                    pseudo_loss = lambda_og(epoch)* loss + lambda_pseudo_train(epoch)*pseudo_train_loss
                    normalizer = lambda_og(epoch) + lambda_pseudo_train(epoch)
                    pseudo_loss /= normalizer
                    if verbose:
                        print(f"Using curriculum formula 2: L_og={lambda_og(epoch)/normalizer}, L_tr={lambda_pseudo_train(epoch)/normalizer}")
                elif args.curriculum3:
                    pseudo_random_loss = F.binary_cross_entropy(torch.sigmoid(pseudo_scores), pseudo_labels)
                    lambda_og = lambda epoch: 1 if epoch <= 600 else 1.95 - .95/600 * epoch if epoch <= 1200 else 0.05
                    lambda_pseudo_random = lambda epoch: 0 if epoch <= 600 else -1 + epoch/600 if epoch <= 1200 else 1
                    pseudo_loss = lambda_og(epoch)*loss + lambda_pseudo_random(epoch)*pseudo_random_loss
                    normalizer = lambda_og(epoch) + lambda_pseudo_random(epoch)
                    pseudo_loss /= normalizer
                    if verbose:
                        print(f"Using curriculum formula 3: L_og={lambda_og(epoch)/normalizer}, L_tr={lambda_pseudo_random(epoch)/normalizer}")
                elif args.curriculum1_stepwise:
                    assert args.on_the_fly_KD
                    pseudo_train_loss = F.binary_cross_entropy(torch.sigmoid(pseudo_scores[is_train_mask]), pseudo_labels[is_train_mask])
                    pseudo_random_loss = F.binary_cross_entropy(torch.sigmoid(pseudo_scores[~is_train_mask]), pseudo_labels[~is_train_mask])
                    lambda_og = lambda epoch: 1 if epoch <= 400 else 0.05
                    lambda_pseudo_train = lambda epoch: 0 if epoch <= 400 else 1
                    lambda_pseudo_random = lambda epoch: 0 if epoch <= 800 else 1
                    pseudo_loss = lambda_og(epoch)*loss + lambda_pseudo_train(epoch)*pseudo_train_loss + lambda_pseudo_random(epoch)*pseudo_random_loss
                    normalizer = lambda_og(epoch) + lambda_pseudo_train(epoch) + lambda_pseudo_random(epoch)
                    pseudo_loss /= normalizer
                    if verbose:
                        print(f"Using curriculum formula 4: L_og={lambda_og(epoch)/normalizer}, L_tr={lambda_pseudo_train(epoch)/normalizer}, \
L_r={lambda_pseudo_random(epoch)/normalizer}")
                else:
                    pseudo_loss = F.binary_cross_entropy(torch.sigmoid(pseudo_scores), pseudo_labels) + beta_kl_loss
                if verbose:
                    print(f"loss: {loss}, pseudo loss: {pseudo_loss}, beta kl loss: {beta_kl_loss}")
                    idx = random.randint(0, len(pseudo_scores)-1)
                    print(f'Example pseudo_scores: {pseudo_scores[idx].item(), pseudo_labels[idx].item()}')
                ## only use use_og if not using flyKD
                if self.use_og: 
                    loss = 0.05 * loss + pseudo_loss ## can try adjusting loss with weighted parameters...
                    loss /= 0.05 + 1
                else:
                    loss = pseudo_loss
                    
            ## Adding LSP loss in the end because we want to avoid 0.05 compression of loss. 
            if self.LSP:
                loss = loss + Total_LSP_loss
            optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
            optimizer.step()
            if args.curriculum1 or args.curriculum2:
                if epoch >= 1600:
                    scheduler.step(loss)
            else:
                scheduler.step(loss)
            end = time.time()
            if verbose:
                print(f'Epoch Training time: {end - strt}')

            if self.weight_bias_track:
                self.wandb.log({"Training Loss": loss})

            if epoch % train_print_per_n == 0:
                # training tracking...
                self.model.eval()
                auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc = get_all_metrics_fb(pred_score_pos, pred_score_neg, scores.reshape(-1,).detach().cpu().numpy(), labels, self.G, True)

                if self.weight_bias_track:
                    temp_d = get_wandb_log_dict(auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc, "Training")
                    temp_d.update({"LR": optimizer.param_groups[0]['lr']})
                    self.wandb.log(temp_d)

                print('Epoch: %d LR: %.5f Loss %.4f, Train Micro AUROC %.4f Train Micro AUPRC %.4f Train Macro AUROC %.4f Train Macro AUPRC %.4f' % (
                    epoch,
                    optimizer.param_groups[0]['lr'], 
                    loss.item(),
                    micro_auroc,
                    micro_auprc,
                    macro_auroc,
                    macro_auprc
                ))

                print('----- AUROC Performance in Each Relation -----')
                print_dict(auroc_rel)
                print('----- AUPRC Performance in Each Relation -----')
                print_dict(auprc_rel)
                print('----------------------------------------------')

            del pred_score_pos, pred_score_neg, scores, labels

            if (epoch) % valid_per_n == 0 or epoch == n_epoch-1:
                # validation tracking...
                print('Validation.....')
                self.model.eval()
                (auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc), loss = evaluate_fb(self.model, self.g_valid_pos, self.g_valid_neg, self.G, self.dd_etypes, self.device, mode = 'valid')

                strt = time.time()
                ## Now way <400 epoch does the best.
                if best_val_acc < macro_auroc:
                    best_val_acc = macro_auroc
                if epoch == n_epoch-1: ## Just storing the last because of some 'killed' error and the sake of time.
                # if best_val_acc < macro_auroc and epoch > int(0.8 * n_epoch): 
                    self.best_model = copy.deepcopy(self.model)
                    self.best_G = copy.deepcopy(self.G)
                print(f"time it took to deep copy the model and graph: {time.time() - strt}")

                print('Epoch: %d LR: %.5f Validation Loss %.4f,  Validation Micro AUROC %.4f Validation Micro AUPRC %.4f Validation Macro AUROC %.4f Validation Macro AUPRC %.4f (Best Macro AUROC %.4f)' % (
                    epoch,
                    optimizer.param_groups[0]['lr'], 
                    loss,
                    micro_auroc,
                    micro_auprc,
                    macro_auroc,
                    macro_auprc,
                    best_val_acc
                ))

                print('----- AUROC Performance in Each Relation -----')
                print_dict(auroc_rel)
                print('----- AUPRC Performance in Each Relation -----')
                print_dict(auprc_rel)
                print('----------------------------------------------')
                
                if sweep_wandb is not None:
                    sweep_wandb.log({'validation_loss': loss, 
                                  'validation_micro_auroc': micro_auroc,
                                  'validation_macro_auroc': macro_auroc,
                                  'validation_micro_auprc': micro_auprc, 
                                  'validation_macro_auprc': macro_auprc})
                
                
                if self.weight_bias_track:
                    temp_d = get_wandb_log_dict(auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc, "Validation")
                    temp_d.update({"Validation Loss": loss,
                                  "Validation Relation Performance": self.wandb.Table(data=to_wandb_table(auroc_rel, auprc_rel),
                                        columns = ["rel_id", "Rel", "AUROC", "AUPRC"])
                                  })

                    self.wandb.log(temp_d)
                print(f'Validation Epoch time: {time.time() - end}')

        print('Testing...')
        self.model.eval()
        (auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc), loss, pred_pos, pred_neg = evaluate_fb(self.best_model, self.g_test_pos, self.g_test_neg, self.best_G, self.dd_etypes, self.device, True, mode = 'test')

        print('Testing Loss %.4f Testing Micro AUROC %.4f Testing Micro AUPRC %.4f Testing Macro AUROC %.4f Testing Macro AUPRC %.4f' % (
            loss,
            micro_auroc,
            micro_auprc,
            macro_auroc,
            macro_auprc
        ))

        if self.weight_bias_track:
            temp_d = get_wandb_log_dict(auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc, "Testing")
            
            temp_d.update({"Testing Loss": loss,
                          "Testing Relation Performance": self.wandb.Table(data=to_wandb_table(auroc_rel, auprc_rel),
                                columns = ["rel_id", "Rel", "AUROC", "AUPRC"])
                          })

            self.wandb.log(temp_d)

        if save_name is not None:
            import pickle
            with open(save_name, 'wb') as f:
                pickle.dump(get_wandb_log_dict(auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc, "Testing"), f)
            
        print('----- AUROC Performance in Each Relation -----')
        print_dict(auroc_rel, dd_only = False)
        print('----- AUPRC Performance in Each Relation -----')
        print_dict(auprc_rel, dd_only = False)
        print('----------------------------------------------')
        
        
    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        
        if self.config is None:
            raise ValueError('No model is initialized...')
        
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
        embeddings = {}
        for ntype in self.best_G.ntypes:
            embeddings[ntype] = self.G.nodes[ntype].data['inp']
        torch.save(embeddings, os.path.join(path, 'best_G.pt'))
        torch.save(self.g_valid_pos, os.path.join(path, 'g_valid_pos.pt'))
        torch.save(self.g_valid_neg, os.path.join(path, 'g_valid_neg.pt'))
        torch.save(self.g_test_pos, os.path.join(path, 'g_test_pos.pt'))
        torch.save(self.g_test_neg, os.path.join(path, 'g_test_neg.pt'))
        torch.save(self.best_model.state_dict(), os.path.join(path, 'model.pt'))
    
    def predict(self, df):
        out = {}
        g = self.G
        df_in = df[['x_idx', 'relation', 'y_idx']]
        for etype in g.canonical_etypes:
            try:
                df_temp = df_in[df_in.relation == etype[1]]
            except:
                print(etype[1])
            src = torch.Tensor(df_temp.x_idx.values).to(self.device).to(dtype = torch.int64)
            dst = torch.Tensor(df_temp.y_idx.values).to(self.device).to(dtype = torch.int64)
            out.update({etype: (src, dst)})
        g_eval = dgl.heterograph(out, num_nodes_dict={ntype: g.number_of_nodes(ntype) for ntype in g.ntypes}, device=self.device)
        
        g = g.to(self.device)
        self.model.eval()
        pred_score_pos, pred_score_neg, pos_score, neg_score = self.model(g, 
                                                                           g_eval, 
                                                                           g_eval, 
                                                                           pretrain_mode = False, 
                                                                           mode = 'test')
        return pred_score_pos

    def retrieve_embedding(self, path = None):
        self.G = self.G.to(self.device)
        h = self.model(self.G, self.G, return_h = True)
        for i,j in h.items():
            h[i] = j.detach().cpu()
            
        if path is not None:
            with open(os.path.join(path, 'node_emb.pkl'), 'wb') as f:
                pickle.dump(h, f)
        
        return h
                      
    def retrieve_sim_diseases(self, relation, k = 5, path = None):
        if relation not in ['indication', 'contraindication', 'off-label']:
            raise ValueError("Please select the following three relations: 'indication', 'contraindication', 'off-label' !")
                      
        etypes = self.dd_etypes

        out_degrees = {}
        in_degrees = {}

        for etype in etypes:
            out_degrees[etype] = torch.where(self.G.out_degrees(etype=etype) != 0)
            in_degrees[etype] = torch.where(self.G.in_degrees(etype=etype) != 0)
        
        sim_all_etypes = self.model.pred.sim_all_etypes
        diseaseid2id_etypes = self.model.pred.diseaseid2id_etypes

        id2diseaseid_etypes = {}
        for etype, diseaseid2id in diseaseid2id_etypes.items():
            id2diseaseid_etypes[etype] = {j: i for i, j in diseaseid2id.items()}   
        
        h = self.retrieve_embedding()
        
        if relation == 'indication':
            etype = ('disease', 'rev_indication', 'drug')
        elif relation == 'contraindication':
            etype = ('disease', 'rev_contraindication', 'drug')          
        elif relation == 'off-label':
            etype = ('disease', 'rev_off-label use', 'drug')           
        
        src, dst = etype[0], etype[2]
        src_rel_idx = out_degrees[etype]
        dst_rel_idx = in_degrees[etype]
        src_h = h[src][src_rel_idx]
        dst_h = h[dst][dst_rel_idx]

        src_rel_ids_keys = out_degrees[etype]
        dst_rel_ids_keys = in_degrees[etype]
        src_h_keys = h[src][src_rel_ids_keys]
        dst_h_keys = h[dst][dst_rel_ids_keys]

        h_disease = {}              
        h_disease['disease_query'] = src_h
        h_disease['disease_key'] = src_h_keys
        h_disease['disease_query_id'] = src_rel_idx
        h_disease['disease_key_id'] = src_rel_ids_keys
        
        sim = sim_all_etypes[etype][np.array([diseaseid2id_etypes[etype][i.item()] for i in h_disease['disease_query_id'][0]])]
                      
        ## get top K most similar diseases and their similarity scores
        coef = torch.topk(sim, k + 1).values[:, 1:]
        ## normalize simialrity scores
        coef = F.normalize(coef, p=1, dim=1)
        ## get these diseases embedding
        embed = h_disease['disease_key'][torch.topk(sim, k + 1).indices[:, 1:]]
        ## augmented disease embedding
        out = torch.mul(embed.to('cpu'), coef.unsqueeze(dim = 2)).sum(dim = 1)
        
        similar_diseases = torch.topk(sim, k + 1).indices[:, 1:]
        similar_diseases = similar_diseases.apply_(lambda x: id2diseaseid_etypes[etype][x]) 
        
        if path is not None:
            with open(os.path.join(path, 'sim_diseases.pkl'), 'wb') as f:
                pickle.dump(similar_diseases, f)
                      
        return similar_diseases
                      
    def load_pretrained(self, path, legacy=False, super_legacy=False, keep_config=False):
        ## load config file
        
        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
            
        if not keep_config:
            if hasattr(config, "args"):
                config["args"].on_the_fly_KD = False
                print("turning off recursively creating more teacher models")
            self.model_initialize(**config)
            self.config = config
        
        state_dict = torch.load(os.path.join(path, 'model.pt'), map_location = torch.device('cpu'))
        if legacy: 
            state_dict_G = torch.load(os.path.join(path, 'G.pt'), map_location = torch.device('cpu'))
        else:
            state_dict_G = torch.load(os.path.join(path, 'best_G.pt'), map_location = torch.device('cpu'))
        if not super_legacy:
            self.g_valid_pos = torch.load(os.path.join(path, 'g_valid_pos.pt'))
            self.g_valid_neg = torch.load(os.path.join(path, 'g_valid_neg.pt'))
            self.g_test_pos = torch.load(os.path.join(path, 'g_test_pos.pt'))
            self.g_test_neg = torch.load(os.path.join(path, 'g_test_neg.pt'))
            
        self.model.load_state_dict(state_dict, strict=False)
        
        for ntype, embs in state_dict_G.items():
            self.G.nodes[ntype].data['inp'] = embs
            if not legacy:
                self.best_G.nodes[ntype].data['inp'] = embs

        self.model = self.model.to(self.device)
        self.G = self.G.to(self.device)
        self.best_G = self.best_G.to(self.device)
        self.best_model = self.model
        print(f"Loaded a trained model from path: {path}. Could be fully trained or only the pretrain phase")
        
    def train_graphmask(self, relation = 'indication',
                              learning_rate = 3e-4,
                              allowance = 0.005,
                              epochs_per_layer = 1000,
                              penalty_scaling = 1,
                              moving_average_window_size = 100,
                              valid_per_n = 5):
        
        self.relation = relation
        
        if relation not in ['indication', 'contraindication', 'off-label']:
            raise ValueError("Please select the following three relations: 'indication', 'contraindication', 'off-label' !")
         
        if relation == 'indication':
            etypes_train = [('drug', 'indication', 'disease'),
                            ('disease', 'rev_indication', 'drug')]
        elif relation == 'contraindication':
            etypes_train = [('drug', 'contraindication', 'disease'), 
                           ('disease', 'rev_contraindication', 'drug')]
        elif relation == 'off-label':
            etypes_train = [('drug', 'off-label use', 'disease'),
                           ('disease', 'rev_off-label use', 'drug')]
        else:
            etypes_train = dd_etypes    
        
        best_loss_sum = 100        
        
        if "graphmask_model" not in self.__dict__:
            self.graphmask_model = copy.deepcopy(self.best_model)
            self.best_graphmask_model = copy.deepcopy(self.graphmask_model)
            ## add all the parameters for graphmask
            self.graphmask_model.add_graphmask_parameters(self.G)
        else:
            print("Training from checkpoint/pretrained model...")
        
        self.graphmask_model.eval()
        disable_all_gradients(self.graphmask_model)
        
        optimizer = torch.optim.Adam(self.graphmask_model.parameters(), lr=learning_rate)
        self.graphmask_model.to(self.device)
        lagrangian_optimization = LagrangianOptimization(optimizer,
                                                         self.device,
                                                         batch_size_multiplier=None)

        f_moving_average = MovingAverage(window_size=moving_average_window_size)
        g_moving_average = MovingAverage(window_size=moving_average_window_size)

        best_sparsity = 1.01

        neg_sampler = Full_Graph_NegSampler(self.G, 1, 'fix_dst', self.device)
        loss_fct = nn.MSELoss()

        self.G = self.G.to(self.device)
        
        ## iterate over layers. One at a time!
        for layer in reversed(list(range(self.graphmask_model.count_layers()))):
            self.graphmask_model.enable_layer(layer) ## enable baselines and gates parameters

            for epoch in range(epochs_per_layer):
                self.graphmask_model.train()
                neg_graph = neg_sampler(self.G)
                original_predictions_pos, original_predictions_neg, _, _ = self.graphmask_model.graphmask_forward(self.G, self.G, neg_graph, graphmask_mode = False, only_relation = relation)

                pos_score = torch.cat([original_predictions_pos[i] for i in etypes_train])
                neg_score = torch.cat([original_predictions_neg[i] for i in etypes_train])
                original_predictions = torch.sigmoid(torch.cat((pos_score, neg_score))).to('cpu')

                updated_predictions_pos, updated_predictions_neg, penalty, num_masked = self.graphmask_model.graphmask_forward(self.G, self.G, neg_graph, graphmask_mode = True, only_relation = relation)
                pos_score = torch.cat([updated_predictions_pos[i] for i in etypes_train])
                neg_score = torch.cat([updated_predictions_neg[i] for i in etypes_train])
                updated_predictions = torch.sigmoid(torch.cat((pos_score, neg_score)))

                labels = [1] * len(pos_score) + [0] * len(neg_score)
                loss_pred = F.binary_cross_entropy(updated_predictions, torch.Tensor(labels).float().to(self.device)).item()

                original_predictions = original_predictions.to(self.device)
                loss_pred_ori = F.binary_cross_entropy(original_predictions, torch.Tensor(labels).float().to(self.device)).item()
                # loss is the divergence between updated and original predictions
                loss = loss_fct(original_predictions, updated_predictions)

                g = torch.relu(loss - allowance).mean()
                f = penalty * penalty_scaling

                lagrangian_optimization.update(f, g)

                f_moving_average.register(float(f.item()))
                g_moving_average.register(float(loss.mean().item()))

                print(
                    "Running epoch {0:n} of GraphMask training. Mean divergence={1:.4f}, mean penalty={2:.4f}, bce_update={3:.4f}, bce_original={4:.4f}, num_masked_l1={5:.4f}, num_masked_l2={6:.4f}".format(
                        epoch,
                        g_moving_average.get_value(),
                        f_moving_average.get_value(),
                        loss_pred,
                        loss_pred_ori,
                        num_masked[0]/self.G.number_of_edges(),
                        num_masked[1]/self.G.number_of_edges())
                )

                if self.weight_bias_track == 'True':
                    self.wandb.log({'divergence': g_moving_average.get_value(),
                              'penalty': f_moving_average.get_value(),
                              'bce_masked': loss_pred,
                              'bce_original': loss_pred_ori,
                              '%masked_L1': num_masked[0]/self.G.number_of_edges(),
                              '%masked_L2': num_masked[1]/self.G.number_of_edges()})

                del original_predictions, updated_predictions, f, g, loss, pos_score, neg_score, loss_pred_ori, loss_pred, neg_graph
                
                if epoch % valid_per_n == 0:
                    loss_sum = evaluate_graphmask(self.graphmask_model, self.G, self.g_valid_pos, self.g_valid_neg, relation, epoch, mode = 'validation', allowance = allowance, penalty_scaling = penalty_scaling, etypes_train = etypes_train, device = self.device, weight_bias_track = self.weight_bias_track, wandb = self.wandb)
                    
                    if loss_sum < best_loss_sum:
                        # takes the best checkpoint
                        best_loss_sum = loss_sum
                        self.best_graphmask_model = copy.deepcopy(self.graphmask_model)
            
        loss_sum, metrics = evaluate_graphmask(self.best_graphmask_model, self.G, self.g_test_pos, self.g_test_neg, relation, epoch, mode = 'testing', allowance = allowance, penalty_scaling = penalty_scaling, etypes_train = etypes_train, device = self.device, weight_bias_track = self.weight_bias_track, wandb = self.wandb)
        
        if self.weight_bias_track == 'True':
            self.wandb.log(metrics)
        return metrics
    
    def save_graphmask_model(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        
        if self.config is None:
            raise ValueError('No model is initialized...')
        
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
       
        torch.save(self.best_graphmask_model.state_dict(), os.path.join(path, 'graphmask_model.pt'))
        
    def load_pretrained_graphmask(self, path):
        ## load config file
        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
            
        self.model_initialize(**config)
        self.config = config
        if "graphmask_model" not in self.__dict__:
            self.graphmask_model = copy.deepcopy(self.best_model)
            self.best_graphmask_model = copy.deepcopy(self.graphmask_model)
            ## add all the parameters for graphmask
            self.graphmask_model.add_graphmask_parameters(self.G)
        
        state_dict = torch.load(os.path.join(path, 'graphmask_model.pt'), map_location = torch.device('cpu'))
        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        self.graphmask_model.load_state_dict(state_dict)
        self.graphmask_model = self.graphmask_model.to(self.device)
        self.best_graphmask_model = self.graphmask_model
    
    
    def retrieve_gates_scores_penalties(self):
        updated_predictions_pos, updated_predictions_neg, penalty, num_masked = self.graphmask_model.graphmask_forward(self.G, self.G, self.G, graphmask_mode = True, only_relation = self.relation, return_gates = True)
        gates = self.graphmask_model.get_gates()
        scores = self.graphmask_model.get_gates_scores()
        penalties = self.graphmask_model.get_gates_penalties()
        
        return gates, scores, penalties
    
    def retrieve_save_gates(self, path):
        _, scores, _ = self.retrieve_gates_scores_penalties()
        
        df_raw = pd.read_csv(os.path.join(self.data_folder, 'kg.csv'))
        df = self.df
        
        df_raw['x_id'] = df_raw.x_id.apply(lambda x: convert2str(x))
        df_raw['y_id'] = df_raw.y_id.apply(lambda x: convert2str(x))

        df['x_id'] = df.x_id.apply(lambda x: convert2str(x))
        df['y_id'] = df.y_id.apply(lambda x: convert2str(x))

        idx2id_all = {}
        id2name_all = {}
        for node_type in self.G.ntypes:
            idx2id = dict(df[df.x_type == node_type][['x_idx', 'x_id']].values)
            idx2id.update(dict(df[df.y_type == node_type][['y_idx', 'y_id']].values))
            id2name = dict(df_raw[df_raw.x_type == node_type][['x_id', 'x_name']].values)
            id2name.update(dict(df_raw[df_raw.y_type == node_type][['y_id', 'y_name']].values))

            idx2id_all[node_type] = idx2id
            id2name_all[node_type] = id2name
            
        all_att_df = pd.DataFrame()
        
        G = self.G.to('cpu')
        for etypes in G.canonical_etypes:
            etype = etypes[1]
            src, dst = etypes[0], etypes[2]

            df_temp = pd.DataFrame()
            df_temp['x_idx'] = G.edges(etype = etype)[0].numpy()
            df_temp['y_idx'] = G.edges(etype = etype)[1].numpy()
            df_temp['x_id'] = df_temp['x_idx'].apply(lambda x: idx2id_all[src][x])
            df_temp['y_id'] = df_temp['y_idx'].apply(lambda x: idx2id_all[dst][x])

            df_temp['x_name'] = df_temp['x_id'].apply(lambda x: id2name_all[src][x])
            df_temp['y_name'] = df_temp['y_id'].apply(lambda x: id2name_all[dst][x])

            df_temp['x_type'] = src
            df_temp['y_type'] = dst
            df_temp['relation'] = etype

            df_temp[self.relation + '_layer1_att'] = scores[0][etype].reshape(-1,)
            df_temp[self.relation + '_layer2_att'] = scores[1][etype].reshape(-1,)

            all_att_df = pd.concat([all_att_df, df_temp], ignore_index=True)
        
        all_att_df.to_pickle(os.path.join(path, 'graphmask_output_' + self.relation + '.pkl'))
        return all_att_df
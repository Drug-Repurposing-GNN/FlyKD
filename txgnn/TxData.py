import os
import math
import copy
import pickle

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import dgl
from .utils import *

from .utils import preprocess_kg, create_split, process_disease_area_split, create_dgl_graph, evaluate_graph_construct, convert2str, data_download_wrapper

import warnings
warnings.filterwarnings("ignore")


class TxData:
    
    def __init__(self, data_folder_path):
        if not os.path.exists(data_folder_path):
            os.mkdir(data_folder_path)
            
        self.data_folder = data_folder_path # the data folder, contains the kg.csv
        data_download_wrapper('https://dataverse.harvard.edu/api/access/datafile/6180626', os.path.join(self.data_folder, 'kg.csv'))
        data_download_wrapper('https://dataverse.harvard.edu/api/access/datafile/6180617', os.path.join(self.data_folder, 'node.csv'))
        data_download_wrapper('https://dataverse.harvard.edu/api/access/datafile/6180616', os.path.join(self.data_folder, 'edges.csv'))
        
        
    def prepare_split(self, split = 'complex_disease',
                     disease_eval_idx = None,
                     seed = 42,
                     no_kg = False,
                     additional_train=None,
                     create_psuedo_edges=False,
                     soft_pseudo=False,
                     pseudo_on_train = None):
        
        if split not in ['random', 'complex_disease', 'disease_eval', 'cell_proliferation', 'mental_health', 'cardiovascular', 'anemia', 'adrenal_gland', 'full_graph', 'downstream_pred']:
            raise ValueError("Please select one of the following supported splits: 'random', 'complex_disease', 'disease_eval', 'cell_proliferation', 'mental_health', 'cardiovascular', 'anemia', 'adrenal_gland'")
            
        if disease_eval_idx is not None:
            split = 'disease_eval'
            print('disease eval index is not none, use the individual disease split...')
        self.split = split
        
        if split in ['cell_proliferation', 'mental_health', 'cardiovascular', 'anemia', 'adrenal_gland']:
            kg_path = os.path.join(self.data_folder, split + '_kg', 'kg_directed.csv')
        else:
            kg_path = os.path.join(self.data_folder, 'kg_directed.csv')
            
        if os.path.exists(kg_path):
            print('Found saved processed KG... Loading...')
            df = pd.read_csv(kg_path)
        else:
            if os.path.exists(os.path.join(self.data_folder, 'kg.csv')):
                print('First time usage... Mapping TxData raw KG to directed csv... it takes several minutes...')
                preprocess_kg(self.data_folder, split)
                df = pd.read_csv(kg_path)
            else:
                raise ValueError("KG file path does not exist...")
        
        if split == 'disease_eval':
            split_data_path = os.path.join(self.data_folder, self.split + '_' + str(disease_eval_idx))
        elif split == 'downstream_pred':
            split_data_path = os.path.join(self.data_folder, self.split + '_downstream_pred')
            disease_eval_idx = [11394.,  6353., 12696., 14183., 12895.,  9128., 12623., 15129.,
                                   12897., 12860.,  7611., 13113.,  4029., 14906., 13438., 13177.,
                                   13335., 12896., 12879., 12909.,  4815., 12766., 12653.]
        elif no_kg:
            split_data_path = os.path.join(self.data_folder, self.split + '_no_kg_' + str(seed))
        else:
            split_data_path = os.path.join(self.data_folder, self.split + '_' + str(seed))
        
        if no_kg:
            sub_kg = ['off-label use', 'indication', 'contraindication']
            df = df[df.relation.isin(sub_kg)].reset_index(drop = True)        
        
        if not os.path.exists(os.path.join(split_data_path, 'train.csv')):
            if not os.path.exists(split_data_path):
                os.mkdir(split_data_path)           
            print('Creating splits... it takes several minutes...')
            df_train, df_valid, df_test = create_split(df, split, disease_eval_idx, split_data_path, seed)
        else:
            print('Splits detected... Loading splits....')
            df_train = pd.read_csv(os.path.join(split_data_path, 'train.csv'), low_memory=False)
            df_valid = pd.read_csv(os.path.join(split_data_path, 'valid.csv'), low_memory=False)
            df_test = pd.read_csv(os.path.join(split_data_path, 'test.csv'), low_memory=False)     
            
        if split not in ['random', 'complex_disease', 'disease_eval', 'full_graph', 'downstream_pred']:
            df_test = process_disease_area_split(self.data_folder, df, df_test, split)

        df_train = df_train.drop_duplicates() 
        df_valid = df_valid.drop_duplicates()       
        df_test = df_test.drop_duplicates() 
    
        if additional_train is not None:
            ## creating rev relations
            # insert pseudo data to create pseudo edges
            if create_psuedo_edges:
                print(f'total df train size before psuedo label injection: {len(df_train)}')
                df_train = df_train.append(additional_train, ignore_index=True)
                print(f'total df train size after psuedo label injection: {len(df_train)}')
                
        print('Creating DGL graph....')
        # create dgl graph
        self.G = create_dgl_graph(df_train, df) ## df is here to obtain the highest index number which is required to create a contiguous DGL graph
        self.full_G = create_dgl_graph(df, df) ## df is here to obtain the highest index number which is required to create a contiguous DGL graph

        def construct_dd_only_graph(df_, soft_pseudo_logits=None, verbose=False, return_out=False): ## only to generate psuedo graphs
            pseudo_dd_etypes = [('drug', 'contraindication', 'disease'), 
                        ('drug', 'indication', 'disease'), 
                        ] 
            out = {}
            df_in = df_
            debug_sum = 0
            for etype in pseudo_dd_etypes:
                try:
                    df_temp = df_in[df_in.relation == etype[1]]
                except:
                    print(etype[1])
                src = df_temp.x_idx.astype(int).values
                dst = df_temp.y_idx.astype(int).values
                debug_sum += len(src)
                out.update({etype: (src, dst)})
                if soft_pseudo_logits is not None:
                    soft_pseudo_logits.update({etype: torch.Tensor(df_temp['score'].values)})

            if verbose:
                print(f'total number of labels injected: {debug_sum}')
            g = dgl.heterograph(out, num_nodes_dict = {ntype: self.G.number_of_nodes(ntype) for ntype in self.G.ntypes})
            if soft_pseudo_logits is not None:
                return g, soft_pseudo_logits 
            elif return_out:
                return g, out
            else:
                return g
        ## add additional training data (self-supervised data)
        self.soft_psuedo_logits_rel = None
        self.g_pos_pseudo = None
        if additional_train is not None and create_psuedo_edges is False:
            ## new dgl to compute for 
            if soft_pseudo:
                ## create labels for dgl graph
                soft_pseudo_logits = {}
                self.g_pos_pseudo, self.soft_psuedo_logits_rel = construct_dd_only_graph(additional_train, soft_pseudo_logits, verbose=True)
            else:
                self.g_pos_pseudo = construct_dd_only_graph(additional_train, verbose=True)

        self.df, self.df_train, self.df_valid, self.df_test = df, df_train, df_valid, df_test
        ## for negative sampling of disease drug relation
        self.g_dd_train, self.dd_train_out = construct_dd_only_graph(df_train[df_train.relation.isin(["indication", "contraindication"])], return_out=True)
        valid_y_idx = df_valid[df_valid["relation"].isin(["indication", "contraindication"])].y_idx
        test_y_idx = df_test[df_test["relation"].isin(["indication", "contraindication"])].y_idx
        valid_test_y_idx = torch.tensor(pd.concat([valid_y_idx, test_y_idx]).unique())
        diseases = torch.arange(self.G.num_nodes("disease"))
        dis_idx_wout_val_test = diseases[~torch.isin(diseases, valid_test_y_idx)]
        assert len(dis_idx_wout_val_test) == self.G.num_nodes("disease") - len(valid_test_y_idx)
        self.dis_idx_wout_val_test = dis_idx_wout_val_test
        self.drug_idx_ptrain = df_train[df_train["relation"].isin(["indication", "contraindication"])].x_idx.value_counts()
        self.dis_idx_ptrain = df_train[df_train["relation"].isin(["indication", "contraindication"])].y_idx.value_counts()
        self.drug_idx_ptrain /= self.drug_idx_ptrain.sum()
        self.dis_idx_ptrain /= self.dis_idx_ptrain.sum()
        self.drug_rel_idx_ptrain = {}
        self.dis_rel_idx_ptrain = {}
        self.drug_rel_idx_ptrain["indication"] = df_train[df_train["relation"] == "indication"].x_idx.value_counts()
        self.drug_rel_idx_ptrain["contraindication"] = df_train[df_train["relation"] == "contraindication"].x_idx.value_counts()
        self.dis_rel_idx_ptrain["indication"] = df_train[df_train["relation"] == "indication"].y_idx.value_counts()
        self.dis_rel_idx_ptrain["contraindication"] = df_train[df_train["relation"] == "contraindication"].y_idx.value_counts()
        self.drug_rel_idx_ptrain["indication"] /= self.drug_rel_idx_ptrain["indication"].sum()
        self.drug_rel_idx_ptrain["contraindication"] /= self.drug_rel_idx_ptrain["contraindication"].sum()
        self.dis_rel_idx_ptrain["indication"] /= self.dis_rel_idx_ptrain["indication"].sum()
        self.dis_rel_idx_ptrain["contraindication"] /= self.dis_rel_idx_ptrain["contraindication"].sum()

        self.disease_eval_idx = disease_eval_idx
        self.no_kg = no_kg
        self.seed = seed
        print('Done!')
        
        
    def retrieve_id_mapping(self):
        df = self.df
        df['x_id'] = df.x_id.apply(lambda x: convert2str(x))
        df['y_id'] = df.y_id.apply(lambda x: convert2str(x))

        idx2id_drug = dict(df[df.x_type == 'drug'][['x_idx', 'x_id']].drop_duplicates().values)
        idx2id_drug.update(dict(df[df.y_type == 'drug'][['y_idx', 'y_id']].drop_duplicates().values))

        idx2id_disease = dict(df[df.x_type == 'disease'][['x_idx', 'x_id']].drop_duplicates().values)
        idx2id_disease.update(dict(df[df.y_type == 'disease'][['y_idx', 'y_id']].drop_duplicates().values))

        df_ = pd.read_csv(os.path.join(self.data_folder, 'kg.csv'))
        df_['x_id'] = df_.x_id.apply(lambda x: convert2str(x))
        df_['y_id'] = df_.y_id.apply(lambda x: convert2str(x))

        id2name_disease = dict(df_[df_.x_type == 'disease'][['x_id', 'x_name']].drop_duplicates().values)
        id2name_disease.update(dict(df_[df_.y_type == 'disease'][['y_id', 'y_name']].drop_duplicates().values))

        id2name_drug = dict(df_[df_.x_type == 'drug'][['x_id', 'x_name']].drop_duplicates().values)
        id2name_drug.update(dict(df_[df_.y_type == 'drug'][['y_id', 'y_name']].drop_duplicates().values))
        
        return {'id2name_drug': id2name_drug,
                'id2name_disease': id2name_disease,
                'idx2id_disease': idx2id_disease,
                'idx2id_drug': idx2id_drug
               }
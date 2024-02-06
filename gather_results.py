# %%
import numpy as np

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import font_manager
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

font_dirs = ["./"]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

sns.set(rc={'figure.figsize':(10,6)})
sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font = "Helvetica", font_scale=1.5)
plt.rc('axes', unicode_minus=False)
from sklearn.metrics import average_precision_score, roc_auc_score
from txgnn import TxData

import pickle
df_res_all = []

from txgnn import TxData
txdata = TxData(data_folder_path = './data')
txdata.prepare_split(split = 'random', seed = 1)
id_mapping = txdata.retrieve_id_mapping()

drug_neg_dict_splits = {}
for split in ['autoimmune', 
               'metabolic_disorder', 
               'diabetes', 'neurodigenerative','random', 'complex_disease', 
               'cell_proliferation', 'mental_health',
               'cardiovascular', 'anemia', 
               'adrenal_gland']:
    drug_neg_dict = {}

    for rel in ['indication', 'contraindication']:
        seed_list = [1,2,3,4,5]
        drug_neg_dict[rel] = {}

        for seed in tqdm(seed_list):
            txdata.prepare_split(split = split, seed = seed)
            drug_neg_index = txdata.df_train[txdata.df_train.relation == 'rev_' + rel].y_idx.unique().astype(int)
            drug_neg_dict[rel][seed] = drug_neg_index
    drug_neg_dict_splits[split] = drug_neg_dict

# 'contraindication'
for rel in ['indication', 'contraindication']:
    for split in tqdm(['autoimmune', 
                       'metabolic_disorder', 
                       'diabetes', 'neurodigenerative','random', 'complex_disease', 
                       'cell_proliferation', 'mental_health',
                       'cardiovascular', 'anemia', 
                       'adrenal_gland']):
        
        seed_list = [1,2,3,4,5]
            
        for seed in seed_list:
            
            #txdata.prepare_split(split = split, seed = seed)
            #drug_neg_index = txdata.df_train[txdata.df_train.relation == 'rev_' + rel].y_idx.unique().astype(int)
            drug_neg_index = drug_neg_dict_splits[split][rel][seed]
            name = rel + '_' + split + '_' + str(seed)

            with open('output/diffusion/diffusion_' + name + '.pkl', 'rb') as f:
                disease2res = pickle.load(f)

            with open('output/HGT_' + name + '.pkl', 'rb') as f:
                disease2res_hgt = pickle.load(f)

            with open('output/HAN_' + name + '.pkl', 'rb') as f:
                disease2res_han = pickle.load(f)
            
            with open('output/ClinicalBioBERT_' + name + '.pkl', 'rb') as f:
                disease2res_bert = pickle.load(f)
            
            #if rel == 'indication':
            with open('output/proximity/proximity_' + name + '.pkl', 'rb') as f:
                disease2res_prox = pickle.load(f)
            
            with open('data/GNN_'+ str(seed)+ '_' + split + '_eval', 'rb') as f:
                disease2res_gnn = pickle.load(f)

            with open('data/TxGNN_'+ str(seed)+ '_' + split + '_eval', 'rb') as f:
                disease2res_txgnn = pickle.load(f)

            for i,j in disease2res.items():
                disease2res[i]['HGT'] = disease2res_hgt[i]
                disease2res[i]['HAN'] = disease2res_han[i]
                disease2res[i]['BioBERT'] = disease2res_bert[i]
                #if rel == 'indication':
                disease2res[i]['Proximity'] = disease2res_prox[i]
                d_id = id_mapping['idx2id_disease'][i]
                disease2res[i]['RGCN'] = {'y_pred': np.array(list(disease2res_gnn['rev_' + rel].loc[d_id].Prediction.values()))}
                disease2res[i]['TxGNN'] = {'y_pred': np.array(list(disease2res_txgnn['rev_' + rel].loc[d_id].Prediction.values()))}

            disease2idx = {}
            for d in disease2res.keys():
                np.random.seed(42)
                y = disease2res[d]['DSD-min']['y']
                y_pos_idx = np.where(y == 1)[0]
                num_hits = len(y_pos_idx)
                query_neg_options = np.intersect1d(np.where(y != 1)[0], drug_neg_index)
                y_neg_idx = np.random.choice(query_neg_options, num_hits)
                disease2idx[d] = (y_pos_idx, y_neg_idx)
            
            
            #method_list = ['DSD-min', 'KL-med', 'KL-min', 'JS-med', 'JS-min', 'Proximity' , 'HGT', 'HAN', 'RGCN', 'BioBERT', 'TxGNN']
            method_list = ['DSD-min', 'KL-min', 'JS-min', 'HGT', 'HAN', 'RGCN', 'BioBERT', 'TxGNN']
            
            method_list.append('Proximity') 
            
            for method in method_list:
                pos_pred_all = []
                neg_pred_all = []
                recall_100_all = []
                precision_100_all = []
                precision_10_all = []
                precision_K_all = []

                avg_rank_all = []
                for d in disease2res.keys():
                    pos_pred = disease2res[d][method]['y_pred'][disease2idx[d][0]].tolist()
                    neg_pred = disease2res[d][method]['y_pred'][disease2idx[d][1]].tolist()
                    pos_pred_all += pos_pred
                    neg_pred_all += neg_pred
                    recall_100_all.append(len(np.intersect1d(np.argsort(disease2res[d][method]['y_pred'])[::-1][:100], disease2idx[d][0]))/len(disease2idx[d][0]))
                    precision_100_all.append(len(np.intersect1d(np.argsort(disease2res[d][method]['y_pred'])[::-1][:100], disease2idx[d][0]))/100)
                    precision_10_all.append(len(np.intersect1d(np.argsort(disease2res[d][method]['y_pred'])[::-1][:10], disease2idx[d][0]))/10)
                    precision_K_all.append(len(np.intersect1d(np.argsort(disease2res[d][method]['y_pred'])[::-1][:len(disease2idx[d][0])], disease2idx[d][0]))/len(disease2idx[d][0]))
                    avg_rank_all.append(np.mean(np.argsort(np.argsort(disease2res[d][method]['y_pred']))[disease2idx[d][0]])/len(disease2res[d][method]['y_pred']))
                    
                auprc = average_precision_score([1]*len(pos_pred_all) + [0] * len(neg_pred_all), pos_pred_all + neg_pred_all)
                auroc = roc_auc_score([1]*len(pos_pred_all) + [0] * len(neg_pred_all), pos_pred_all + neg_pred_all)
                df_res_all.append((method, auprc, seed, split, rel, 'AUPRC'))
                df_res_all.append((method, auroc, seed, split, rel, 'AUROC'))
                df_res_all.append((method, np.mean(recall_100_all), seed, split, rel, 'Recall@100'))
                df_res_all.append((method, np.mean(avg_rank_all), seed, split, rel, 'Avg Rank'))
                df_res_all.append((method, np.mean(precision_100_all), seed, split, rel, 'Precision@100'))
                df_res_all.append((method, np.mean(precision_10_all), seed, split, rel, 'Precision@10'))
                df_res_all.append((method, np.mean(precision_K_all), seed, split, rel, 'Precision@K'))


# %%
split_to_name = {
    'random': 'Random Disease Split', 
    'complex_disease': 'Zero-shot Disease Split', 
    'cell_proliferation': 'Disease Area Split: Cell Proliferation', 
    'mental_health': 'Disease Area Split: Mental Health', 
    'cardiovascular': 'Disease Area Split: Cardiovascular', 
    'anemia': 'Disease Area Split: Anemia', 
    'adrenal_gland': 'Disease Area Split: Adrenal Gland',
    'autoimmune': 'Disease Area Split: Autoimmune', 
    'metabolic_disorder': 'Disease Area Split: Metabolic Disorder', 
    'diabetes': 'Disease Area Split: Diabetes', 
    'neurodigenerative': 'Disease Area Split: Neurodegenerative', 
}

df_res_all_pd = pd.DataFrame(df_res_all).rename(columns = {0: 'Method', 1: 'Metric', 2: 'Seed', 3: 'Split', 4: 'Task', 5: 'Metric Name'})

df_res_all_pd.to_csv('result_more_metrics.csv', index = False)

sns.set(rc={'figure.figsize':(7,4)})
sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font = "Helvetica", font_scale=1.5)
plt.rc('axes', unicode_minus=False)

rel = 'indication'
for split in ['random', 'complex_disease', 'cell_proliferation', 'mental_health', 
              'cardiovascular', 'anemia', 'adrenal_gland', 'autoimmune', 
              'metabolic_disorder', 'diabetes', 'neurodigenerative']:
    df_res_all_pd['Method'] = df_res_all_pd.Method.apply(lambda x: x.split('-')[0] if len(x.split('-')) > 1 else x)
    df_metric = df_res_all_pd[(df_res_all_pd.Task == rel) & (df_res_all_pd.Split == split)]
    print(split_to_name[split])
    sns.stripplot(data = df_metric[df_metric['Metric Name'] == 'AUPRC'], x = 'Method', y = 'Metric', hue = 'Method', 
                  order = ['KL', 'JS', 'DSD', 'Proximity', 'RGCN', 'HGT', 'HAN', 'BioBERT', 'TxGNN'], alpha = 0.3)
    g = sns.pointplot(data = df_metric[df_metric['Metric Name'] == 'AUPRC'], x = 'Method', y = 'Metric', hue = 'Method', 
                  order = ['KL', 'JS', 'DSD', 'Proximity', 'RGCN', 'HGT', 'HAN', 'BioBERT', 'TxGNN'])
    g.set(xlabel = '', ylabel = 'AUPRC',  ylim = (0,1.05), title = rel.capitalize() + ' - ' + split_to_name[split])
    sns.despine()
    g.legend_.remove()
    plt.xticks(rotation=30)
    plt.show()


sns.set(rc={'figure.figsize':(7,4)})
sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font = "Helvetica", font_scale=1.5)
plt.rc('axes', unicode_minus=False)

rel = 'contraindication'
for split in ['random', 'complex_disease', 'cell_proliferation', 'mental_health', 
              'cardiovascular', 'anemia', 'adrenal_gland', 'autoimmune', 
              'metabolic_disorder', 'diabetes', 'neurodigenerative']:
    df_res_all_pd['Method'] = df_res_all_pd.Method.apply(lambda x: x.split('-')[0] if len(x.split('-')) > 1 else x)
    df_metric = df_res_all_pd[(df_res_all_pd.Task == rel) & (df_res_all_pd.Split == split)]
    print(split_to_name[split])
    sns.stripplot(data = df_metric[df_metric['Metric Name'] == 'AUPRC'], x = 'Method', y = 'Metric', hue = 'Method', 
                  order = ['KL', 'JS', 'DSD', 'Proximity', 'RGCN', 'HGT', 'HAN', 'BioBERT', 'TxGNN'], alpha = 0.3)
    g = sns.pointplot(data = df_metric[df_metric['Metric Name'] == 'AUPRC'], x = 'Method', y = 'Metric', hue = 'Method', 
                  order = ['KL', 'JS', 'DSD', 'Proximity', 'RGCN', 'HGT', 'HAN', 'BioBERT', 'TxGNN'])
    g.set(xlabel = '', ylabel = 'AUPRC',  ylim = (0,1.05), title = rel.capitalize() + ' - ' + split_to_name[split])
    sns.despine()
    g.legend_.remove()
    plt.xticks(rotation=30)
    plt.show()



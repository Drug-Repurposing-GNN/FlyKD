















## Things to try:
## use KL divergence loss instead of CE
## try evaluation (valid/test) without DPM

## Running
nohup python -u noisy_student.py --student_size 120 > ./logs/120/reference_proper.txt
# nohup python -u noisy_student.py --reparam_mode MLP --student_size 120 > ./logs/120/e2000_0.95lr_MLP.txt
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og+random100.csv --soft_pseudo --student_size 120 --use_og > ./logs/120/soft_pseudo_inog+random100_pneg_wog_e2000_0.95lr_MLP.txt
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og+random100.csv --soft_pseudo --student_size 120 --use_og --reparam_mode MLP > ./logs/120/soft_pseudo_inog+random100_pneg_wog_e2000_0.95lr_MLP.txt


## Try random+100 too, since it seems like having restrained or not doesn't really matter.
# nohup python -u noisy_student.py > ./logs/reference_performances/restrained_1_e2000__0.95lr.txt
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og+random100.csv --soft_pseudo --student_size 100 --use_og > ./logs/soft_pseudo_inog+random100_pneg_wog_e2000_fromrestr_0.95lr.txt

# nohup python -u noisy_student.py > ./logs/reference_performances/restrained_1_e1000__0.9lr.txt
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_restrained_inog_dataset.csv --neg_pseudo_sampling --soft_pseudo --student_size 100 --use_og > ./logs/soft_pseudo_inog__pneg_wog_e1000_fromrestr_0.9lr.txt

# nohup python -u noisy_student.py --psuedo_label_fname pscores_all_restrained_saveG.csv --deg inf > ./logs/psuedo_scores_all_restrained_saveG.txt 2>&1 ## generating ALL psuedo_labels
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og_dataset.csv --neg_pseudo_sampling --soft_pseudo --student_size 100 --use_og > ./logs/test.txt
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og+random100.csv --soft_pseudo --student_size 100 --use_og > ./logs/soft_pseudo_inog+random100_pneg_wog_e1000_NoDPMforood.txt
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og+random100.csv --soft_pseudo --student_size 100 --use_og > ./logs/soft_pseudo_inog+random100_pneg_wog_e1000_allweakDPM.txt

# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og_dataset.csv --neg_pseudo_sampling --soft_pseudo --student_size 100 --use_og > ./logs/soft_pseudo_inog_pneg_wog_e2000.txt
# nohup python -u noisy_student.py > ./logs/reference_performances/restrained_1_e2000.txt
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og+random100.csv --soft_pseudo --student_size 100 --use_og > ./logs/soft_pseudo_inog+random100_pneg_wog_e2000.txt

# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og+random80.csv --soft_pseudo --student_size 100 --use_og > ./logs/test.txt

# nohup python -u noisy_student.py --train_then_generate --psuedo_label_fname psuedo_scores_500.csv --deg inf --k_top_candidates 500 --set_seed 1 > ./logs/psuedo_scores_500.txt 2>&1 ## generating psuedo_labels

# nohup python -u noisy_student.py --fix_split_random_seed > ./logs/reference_performances/restrained_fix_split_random_seed_finetune.txt
# nohup python -u noisy_student.py --fix_split_random_seed > ./logs/reference_performances/restrained_fix_split_random_seed4_finetune.txt
# nohup python -u noisy_student.py --fix_split_random_seed > ./logs/reference_performances/restrained_fix_split_random_seed5_finetune.txt

# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og_dataset.csv --neg_pseudo_sampling --soft_pseudo --student_size 100 --fix_split_random_seed --use_og > ./logs/reference_performances/restrained_random_finetune_pseudo4.txt
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og_dataset.csv --neg_pseudo_sampling --soft_pseudo --student_size 100 --fix_split_random_seed --use_og > ./logs/reference_performances/restrained_random_finetune_pseudo5.txt

# nohup python -u noisy_student.py --student_size 100 --LSP RBF --LSP_size partial --use_og > ./logs/reference_performances/restrained_1_finetune_savesim_RBF_partial_wog.txt

# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og_dataset.csv --neg_pseudo_sampling --soft_pseudo --student_size 100 --random_seed --use_og > ./logs/reference_performances/restrained_random_finetune_pseudo.txt
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og_dataset.csv --neg_pseudo_sampling --soft_pseudo --student_size 100 --random_seed --use_og > ./logs/reference_performances/restrained_random_finetune_pseudo2.txt
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og_dataset.csv --neg_pseudo_sampling --soft_pseudo --student_size 100 --random_seed --use_og > ./logs/reference_performances/restrained_random_finetune_pseudo3.txt

# nohup python -u noisy_student.py --save_name seed_1_restrained_saveG > ./logs/save_pre_restrained_save_G.txt ## want to make sure the sum of the positive matches (reproduced).

## generates all indication and contraindication
# nohup python -u noisy_student.py --psuedo_label_fname softpseudo_restrained_saveG.csv > ./logs/generate_pseudo_restrained_1.txt ## want to make sure the sum of the positive matches (reproduced).
# nohup python -u noisy_student.py --psuedo_label_fname pseudo_valid_restrained_saveG.csv --generate_indication > ./logs/generate_validation_restrained_1.txt ## want to make sure the sum of the positive matches (reproduced).

# nohup python -u noisy_student.py > ./logs/reference_performances/restrained_1_finetune.txt
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og_dataset.csv --neg_pseudo_sampling --soft_pseudo --student_size 100 --set_seed 1 --use_og > ./logs/softpsuedo_inog_wpneg_Finetune_CE_wDPM_T2_wog.txt

# nohup python -u noisy_student.py --student_size 100 --set_seed 1 --LSP RBF --LSP_size full > ./logs/reference_performances/restrained_1_finetune_savesim_RBF_full.txt
# nohup python -u noisy_student.py --student_size 100 --set_seed 1 --LSP cosine --LSP_size partial > ./logs/reference_performances/restrained_1_finetune_savesim_cosine_partial.txt

# nohup python -u noisy_student.py --student_size 100 --set_seed 1 > ./logs/reference_performances/restrained_1_savesim.txt
# nohup python -u noisy_student.py --student_size 100 --set_seed 1 --LSP > ./logs/LSP_all.txt
# nohup python -u noisy_student.py --psuedo_label_fname toy_pseudo_scores_train.csv --neg_pseudo_sampling --soft_pseudo --student_size 100 --set_seed 1 > ./logs/softpsuedo_toytrain_wpneg_Finetune_CE_wDPM.txt

# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og_dataset.csv --neg_pseudo_sampling --soft_pseudo --student_size 100 --set_seed 1 > ./logs/softpsuedo_inog_wpneg_Finetune_CE_wDPM.txt
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og_dataset.csv --use_og --neg_pseudo_sampling --soft_pseudo --student_size 100 --set_seed 1 > ./logs/softpsuedo_inog_wpneg_Finetune_CE_wDPM_wog.txt
# nohup python -u noisy_student.py --psuedo_label_fname toy_pseudo_scores_train.csv --neg_pseudo_sampling --soft_pseudo --student_size 100 --set_seed 1 --kl > ./logs/softpsuedo_toytrain_wpneg_Finetune_KL_wDPM.txt
# python -u validation_test.py --save_dir seed_1_restrained_saveG
# nohup python -u validation_test.py --save_dir seed_1_restrained_saveG > ./logs/proper_seed_1_restrained_saveG.txt
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og_dataset.csv --neg_pseudo_sampling --use_og --soft_pseudo --student_size 100 --set_seed 1 --kl > ./logs/test.txt
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og+random80.csv --neg_pseudo_sampling --use_og --soft_pseudo --student_size 100 --set_seed 1 --kl > ./logs/soft_psuedo_scores_in_og+random80_KL_Finetune_no_rev_w_neg_w_og.txt 2>&1
## ---
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og_dataset.csv --neg_pseudo_sampling --soft_pseudo --student_size 100 --set_seed 1 --kl > ./logs/soft_pseudo_labels_og_dataset_KL_Finetune_no_rev_w_neg_w_ogloss.txt 2>&1
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og_dataset.csv --neg_pseudo_sampling --soft_pseudo --student_size 100 --set_seed 1 --kl --no_dpm > ./logs/soft_pseudo_labels_og_dataset_KL_Finetune_no_rev_w_neg_wout_DP.txt 2>&1
# nohup python -u noisy_student.py --generate_contraindication --psuedo_label_fname psuedo_scores_all_contraindication.csv --deg inf --set_seed 1 --k_top_candidates -1 > ./logs/psuedo_scores_all_contraindication.txt 2>&1 ## generating ALL psuedo_labels
# nohup python -u noisy_student.py --generate_contraindication --psuedo_label_fname psuedo_scores_all_contraindication.csv --deg inf --set_seed 1 --k_top_candidates -1 > ./logs/psuedo_scores_all_contraindication.txt 2>&1 ## generating ALL psuedo_labes

# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_top_and_bottom_20.csv --soft_pseudo --student_size 100 --set_seed 1 --kl > ./logs/soft_pseudo_labels_top_and_bottom_20_KL_Finetune_no_rev.txt 2>&1
# nohup python -u noisy_student.py --generate_contraindication --psuedo_label_fname psuedo_scores_all_contraindication.csv --deg inf --set_seed 1 --k_top_candidates -1 > ./logs/psuedo_scores_all_contraindication.txt 2>&1 ## generating ALL psuedo_labels
# nohup python -u E_Demo.py > ./logs/no_rev_normal_finetune.txt 2>&1 ## 

## Fine-tuning only \
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_top20.csv --soft_pseudo --student_size 100 --set_seed 1 > ./logs/high_degree_soft_pseudo_labels_top100_no_neg_sampling_CE_Finetune_no_rev.txt 2>&1
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_top20.csv --soft_pseudo --student_size 100 --set_seed 1 --kl --neg_pseudo_sampling > ./logs/soft_pseudo_labels_top20_KL_Finetune_no_rev.txt 2>&1
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_top20.csv --soft_pseudo --student_size 100 --set_seed 1 --kl > ./logs/soft_pseudo_labels_top20_KL_Finetune_no_rev.txt 2>&1
# nohup python -u noisy_student.py --train_then_generate --psuedo_label_fname psuedo_scores_500.csv --deg inf --k_top_candidates 500 --set_seed 1 > ./logs/psuedo_scores_500.txt 2>&1 ## generating psuedo_labels
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_top50.csv --soft_pseudo --student_size 100 --set_seed 1 > ./logs/soft_pseudo_labels_top50_no_neg_sampling_CE.txt 2>&1
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_top20.csv --soft_pseudo --student_size 100 --set_seed 1 --kl > ./logs/soft_pseudo_labels_ONLY_top20_no_neg_sampling_KL.txt 2>&1
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_top50.csv --deg inf --k_top_candidates 50 > ./testing/generating.txt 2>&1 ## generating top 50 psuedo scores
# nohup python -u E_Demo.py --seed 42 > ./logs/normal_42_2.txt 2>&1 ## generating top 50 psuedo scores

# nohup python -u noisy_student.py --psuedo_label_fname psuedo_labels_75000.csv --psuedo_edge > ./logs/75000_psuedo_edges2.txt 2>&1 ## running psuedo edge GNN
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_labels_75000.csv > ./logs/75000_psuedo_labels.txt 2>&1 ## running psuedo edge GNN
# nohup python -u E_Demo.py > training_normal.txt ## this should match the original score

# nohup python -u E_Demo.py  --psuedo > testing_training.txt ## hope this at least matches the original score
# nohup python -u E_Demo.py --psuedo > testing_psuedo_training.txt

# nohup python -u noisy_student.py --psuedo_label_fname psuedo_labels_75000.csv > ./logs/full_75000_psuedo_labels.txt 2>&1 ## generating psuedo_labels

# Note: 2>&1 means log the error as well

## Noisy Student
# nohup python -u noisy_student.py > noisy_student2.txt 2>&1 & 

## Regular Test
# nohup python -u noisy_student.py --dropout 0.1 > ./logs/noisy_student_dropout.txt 2>&1  ## dropout
# nohup python -u noisy_student.py --psuedo_edges > ./logs/noisy_student_p_edges.txt 2>&1  ## dropout
# nohup python -u noisy_student.py --reparam_mode MLP > ./logs/noisy_student_MLP.txt 2>&1  ## VGAE MLP
# nohup python -u noisy_student.py --reparam_mode RMLP > ./logs/noisy_student_RMLP.txt 2>&1  ## VGAE RMLP
# nohup python -u noisy_student.py --reparam_mode MPNN > ./logs/noisy_student_MPNN.txt 2>&1  ## VGAE MPNN

# nohup python -u noisy_student.py --psuedo_edges --k_top_candidates 1  --psuedo_label_fname top1_cand_psuedo > ./logs/top1_psuedo_edges.txt 2>&1  ## VGAE MPNN
# nohup python -u noisy_student.py --weight_decay  1e-4 --psuedo_label_fname top1_cand_psuedo > ./logs/weight_decay.txt 2>&1  ## VGAE MPNN

## fname 'psuedo_labels_75000.csv'

# nohup python -u noisy_student.py --k_top_candidates 10 > ./logs/top10_cand.txt 2>&1  ## top 10
# nohup python -u noisy_student.py --k_top_candidates 20 > ./logs/top20_cand.txt 2>&1  ## top 20

# nohup python -u noisy_student.py --student_size 100 --train_from_scratch  > ./logs/same_size_student.txt 2>&1 ## Testing
# nohup python -u noisy_student.py --train_from_scratch > ./logs/noisy_student2.txt 2>&1 ## Testing
# nohup python -u noisy_student.py --three_iter_from_scratch > ./logs/three_iter_from_scrach.txt 2>&1 ## Testing

# nohup python -u E_Demo.py > ./logs/testing2.txt 2>&1 ## observe that there is no discrepency between these two models.
# nohup python -u E_Demo.py > ./logs/testing3.txt 2>&1 ## observe that there is no discrepency between these two models.
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_labels_75000.csv --student_size 100 > ./logs/same_size_student.txt 2>&1 ## Observe that the knowledge transfer does not exist and gives the same performance as the previous two.

# nohup python -u noisy_student.py --psuedo_label_fname psuedo_labels_15000.csv --k_top_candidates 1 > ./logs/15000_psuedo_edges.txt 2>&1 ## generating psuedo_labels
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_labels_15000.csv --psuedo_edges > ./logs/15000_psuedo_edges.txt 2>&1 ## running psuedo edge GNN

# nohup python -u noisy_student.py --psuedo_label_fname psuedo_labels_75000.csv > ./logs/75000_psuedo_edges.txt 2>&1 ## generating psuedo_labels
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_labels_75000.csv --psuedo_edges > ./logs/75000_psuedo_edges.txt 2>&1 ## running psuedo edge GNN

# nohup python -u noisy_student.py --psuedo_label_fname psuedo_labels_15000.csv --student_size 100 > ./logs/15000_psuedo_labels.txt 2>&1 ## running psuedo edge GNN
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_labels_75000.csv --student_size 100 > ./logs/75000_psuedo_labels.txt 2>&1 ## running psuedo edge GNN

# nohup python -u noisy_student.py --psuedo_label_fname pl_least_score_5.5.csv --least_score 5.5 > ./logs/generating.txt 2>&1 ## generating psuedo_labels
# nohup python -u noisy_student.py --psuedo_label_fname pl_least_score_6.csv --least_score 6 > ./logs/generating2.txt 2>&1 ## generating psuedo_labels

# nohup python -u noisy_student.py --psuedo_label_fname pl_least_score_5.5.csv --student_size 100 > ./logs/least_score5.5_label.txt 2>&1 ## running psuedo edge GNN
# nohup python -u noisy_student.py --psuedo_label_fname pl_least_score_5.5.csv --student_size 100 --psuedo_edges > ./logs/least_score5.5_edges.txt 2>&1 ## running psuedo edge GNN
# nohup python -u noisy_student.py --psuedo_label_fname pl_least_score_6.csv --student_size 100 > ./logs/least_score6_label.txt 2>&1 ## running psuedo edge GNN
# nohup python -u noisy_student.py --psuedo_label_fname pl_least_score_6.csv --student_size 100 --psuedo_edges > ./logs/least_score6_edges.txt 2>&1 ## running psuedo edge GNN


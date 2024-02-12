















## Things to try:
## use KL divergence loss instead of CE
## try evaluation (valid/test) without DPM

## Running
## No negative sampling
nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_top20.csv --soft_pseudo --student_size 100 --set_seed 1 > ./logs/soft_pseudo_labels_top20_no_neg_sampling_KL.txt 2>&1
## ---
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


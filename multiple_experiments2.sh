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

# nohup python -u noisy_student.py --train_from_scratch > ./logs/noisy_student.txt 2>&1 ## Testing

nohup python -u noisy_student.py --train_from_scratch --deg inf > ./logs/all_disease_studento.txt 2>&1 ## Testing

nohup python -u noisy_student.py --use_diff_savedir --three_iter_from_scratch > ./logs/three_iter_from_scrach2o.txt 2>&1 ## Testing



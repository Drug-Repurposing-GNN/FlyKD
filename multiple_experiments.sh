######################################### Script Examples #########################################
## student model baseline (130, 80) without KD
python -u noisy_student.py proper_130_e2000_90 --set_seed 45 --teacher_size 130 --epochs 2000 --save_model
python -u noisy_student.py proper_80_e2000_90 --set_seed 45 --teacher_size 80 --epochs 2000 --save_model

## OG FlyKD on seed 45
python -u noisy_student.py NS_130to80_soft_pseudo_flyKD_curr1 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
    --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --epochs 2000 --curriculum1

## FlyKD (configuration: decrease entropy of probability distribution of Random Graph)
# python -u noisy_student.py NS_130to80_soft_pseudo_flyKD_curr1_modprob --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#     --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --epochs 2000 --curriculum1 --modified_multinomial

## FlyKD (configuration: step-wise Curriculum Learning)
# python -u noisy_student.py NS_130to80_soft_pseudo_flyKD_curr1_stepwise --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#     --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --epochs 2000 --curriculum1_stepwise

## FlyKD (configuration: only maintain confident pseudo labels - (> logit score 2))
# python -u noisy_student.py NS_130to80_soft_pseudo_flyKD_curr1_str2 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1 --limited_neg_pseudo_sampling \
#     --soft_pseudo --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --epochs 2000 --curriculum1 --strong_scores 2

## FlyKD (configuration: Occasional 5)
# python -u noisy_student.py NS_130to80_soft_pseudo_flyKD_curr1_occasional5 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1 --limited_neg_pseudo_sampling \
#     --soft_pseudo --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --epochs 2000 --curriculum1 --occasional_flyKD 5

## Basic KD
# python -u noisy_student.py NS_130to80_soft_pseudo_no_curr --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#     --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --epochs 2000

## Basic KD (with Curriculum Learning)
# python -u noisy_student.py NS_130to80_soft_pseudo_curr2 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#     --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --epochs 2000 --curriculum2

## FlyKD (no pseudo labels on train dataset)
# python -u noisy_student.py NS_130to80_soft_pseudo_flyKD_no_ptrain --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#     --limited_neg_pseudo_sampling --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --epochs 2000 --curriculum3 --no_ptrain_flyKD

## FlyKD (configuration: No Curriculum Learning)
# python -u noisy_student.py NS_130to80_soft_pseudo_flyKD_no_curr --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#     --limited_neg_pseudo_sampling --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --epochs 2000 --no_curriculum

## FlyKD (configuration: Fixed Random Graph)
# python -u noisy_student.py NS_130to80_soft_pseudo_fixed_flyKD_curr1 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#     --limited_neg_pseudo_sampling --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --epochs 2000 --curriculum1 --fixed_flyKD
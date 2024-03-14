

######################################### Script Examples #########################################
## OG FlyKD on seed 45
python -u noisy_student.py test/NS_130to80_soft_pseudo_flyKD_curr1 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
    --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --epochs 2000 --curriculum1

## student model baseline without KD
python -u noisy_student.py test/proper_80_e1200_90 --set_seed 45 --teacher_size 80

##
######################################### Below are my scratch work for reference if needed #########################################
#### Test out different Noise -- choose configuration depending on the results####
## weight decay (only on the student model)
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_flyKD+random100_ptrain_weight_decay --set_seed 45 --teacher_size 130 --student_size 150 --iter 2\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --on_the_fly_KD --ptrain --weight_decay 5e-4
## dropout
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_flyKD+random100_ptrain_dropout0.2 --set_seed 45 --teacher_size 130 --student_size 150 --iter 2\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --on_the_fly_KD --ptrain --dropout 0.2
## vgae+dropout (only on the student model)
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_flyKD+random100_ptrain_MLP+dropout --set_seed 45 --teacher_size 130 --student_size 150 --iter 2\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --on_the_fly_KD --ptrain --reparam_mode MLP --dropout 0.1
## Sanity check. 
# noisy_student_130to80_soft_pseudo_flyKD+random100_ptrain should match noisy_student_130to80_soft_pseudo_flyKD+random100_ptrain_test
# or check random100 and see if anything used pretrain phase ckpt and then compare whether they had the same score.
#------
# # # 1
# python -u noisy_student.py NS_130to80_soft_pseudo_fixed_flyKD_nocurr --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#     --limited_neg_pseudo_sampling --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --epochs 2000 --no_curriculum --fixed_flyKD
# # # 2
# python -u noisy_student.py NS_130to80_soft_pseudo_fixed_flyKD_nocurr --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#     --limited_neg_pseudo_sampling --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --epochs 2000 --no_curriculum --fixed_flyKD
# python -u noisy_student.py NS_130to80_soft_pseudo_fixed_flyKD_nocurr --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#     --limited_neg_pseudo_sampling --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --epochs 2000 --no_curriculum --fixed_flyKD
# # 3
# python -u noisy_student.py whoeverrunsthiscodeisstupid --set_seed 48 --teacher_size 130 --student_size 80 --iter 1\
#     --limited_neg_pseudo_sampling --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --epochs 2000 --no_curriculum --fixed_flyKD
# python -u noisy_student.py NS_130to80_soft_pseudo_fixed_flyKD_nocurr123123 --set_seed 49 --teacher_size 130 --student_size 80 --iter 1\
#     --limited_neg_pseudo_sampling --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --epochs 2000 --no_curriculum --fixed_flyKD
# # # 4
# # # # # 5
# # # 6

## --------------------------------------------------------------------------------------------------------------------------------------------------------------##
## --------------------------------------------------------------------------------------------------------------------------------------------------------------##
## --------------------------------------------------------------------------------------------------------------------------------------------------------------##
# python -u noisy_student.py NS_130to80_soft_pseudo_flyKD_curr1_modprob --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#     --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --epochs 2000 --curriculum1 --modified_multinomial
# python -u noisy_student.py NS_130to80_soft_pseudo_flyKD_curr1_stepwise --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#     --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --epochs 2000 --curriculum1_stepwise
# python -u noisy_student.py NS_130to80_soft_pseudo_flyKD_curr1_str2 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1 --limited_neg_pseudo_sampling \
#     --soft_pseudo --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --epochs 2000 --curriculum1 --strong_scores 2
# python -u noisy_student.py NS_130to80_soft_pseudo_flyKD_curr1_occasional5 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1 --limited_neg_pseudo_sampling \
#     --soft_pseudo --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --epochs 2000 --curriculum1 --occasional_flyKD 5
# python -u noisy_student.py NS_130to80_soft_pseudo_flyKD_curr1 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#     --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --epochs 2000 --curriculum1
# python -u noisy_student.py NS_130to80_soft_pseudo_no_curr --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#     --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --epochs 2000
# python -u noisy_student.py NS_130to80_soft_pseudo_curr2 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#     --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --epochs 2000 --curriculum2
# python -u noisy_student.py NS_130to80_soft_pseudo_flyKD_no_ptrain --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#     --limited_neg_pseudo_sampling --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --epochs 2000 --curriculum3 --no_ptrain_flyKD
# python -u noisy_student.py NS_130to80_soft_pseudo_flyKD_no_curr --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#     --limited_neg_pseudo_sampling --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --epochs 2000 --no_curriculum
# python -u noisy_student.py NS_130to80_soft_pseudo_fixed_flyKD_curr1 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#     --limited_neg_pseudo_sampling --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --epochs 2000 --curriculum1 --fixed_flyKD
# python -u noisy_student.py NS_130to80_soft_pseudo_ptrain_rel_multinomial_100000_curr1 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#             --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --epochs 2000 --curriculum1 
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain_curr2 --set_seed 48 --teacher_size 130 --student_size 80 --iter 1\
#                             --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --curriculum2 --epochs 2000
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                             --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --epochs 2000
# python -u noisy_student.py NS_130to80_soft_pseudo_ptrain_rel_multinomial_100000 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#             --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --epochs 2000
# python -u noisy_student.py ptrain130 --set_seed 45 --teacher_size 130 --save_model --epochs 2000
# python -u noisy_student.py ptrain80 --set_seed 48 --teacher_size 80 --force_finetune_iter0 --epochs 2000
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain_rel_multinomial_100000_curr1_adjlr2 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --curriculum1 
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain_curr2_adjlr2 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                             --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --curriculum2
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random100 --set_seed 48 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD_rel_multinomial_100000_str3 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#     --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --strong_scores 3 --balance_train_random 1
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain_rel_multinomial_100000_balance_random0.5 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --balance_train_random 0.5
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain_bneg --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                             --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --scale_neg_loss
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_rel_multinomial_ptrain100000 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_rel_multinomial_ptrain100000 --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_rel_multinomial_ptrain100000 --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_rel_multinomial_ptrain500000 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 500000 --on_the_fly_KD --rel_multinomial_ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_rel_multinomial_ptrain500000 --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 500000 --on_the_fly_KD --rel_multinomial_ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_rel_multinomial_ptrain500000 --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 500000 --on_the_fly_KD --rel_multinomial_ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_rel_multinomial_ptrain30000 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 30000 --on_the_fly_KD --rel_multinomial_ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_rel_multinomial_ptrain30000 --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 30000 --on_the_fly_KD --rel_multinomial_ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_rel_multinomial_ptrain30000 --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 30000 --on_the_fly_KD --rel_multinomial_ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_multinomial_ptrain30000 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 30000 --on_the_fly_KD --multinomial_ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_multinomial_ptrain30000 --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 30000 --on_the_fly_KD --multinomial_ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_multinomial_ptrain30000 --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 30000 --on_the_fly_KD --multinomial_ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_rel_ptrain --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --rel_ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_rel_ptrain --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --rel_ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_rel_ptrain --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --rel_ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_rel_ptrain --set_seed 48 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --rel_ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_rel_ptrain --set_seed 49 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --rel_ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random200_ptrain --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 200 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random200_ptrain --set_seed 48 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 200 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random200_ptrain --set_seed 49 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 200 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_ptrain --set_seed 49 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random200 --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                             --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 200 --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random100_ptrain --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random100_ptrain --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random100_ptrain --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random100_ptrain --set_seed 48 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random100_ptrain --set_seed 49 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_str3 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --strong_scores 3
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_str3 --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --strong_scores 3
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_str3 --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --strong_scores 3
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_flyKD+random50_ptrain_RMLP --set_seed 47 --teacher_size 130 --student_size 150 --iter 2\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --ptrain --reparam_mode RMLP
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_flyKD+random100_ptrain_RMLP --set_seed 45 --teacher_size 130 --student_size 150 --iter 2\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --on_the_fly_KD --ptrain --reparam_mode RMLP
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_flyKD+random100_ptrain_RMLP --set_seed 46 --teacher_size 130 --student_size 150 --iter 2\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --on_the_fly_KD --ptrain --reparam_mode RMLP
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_flyKD+random100_ptrain_RMLP --set_seed 47 --teacher_size 130 --student_size 150 --iter 2\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --on_the_fly_KD --ptrain --reparam_mode RMLP
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_flyKD+random100_ptrain_MLP --set_seed 45 --teacher_size 130 --student_size 150 --iter 2\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --on_the_fly_KD --ptrain --reparam_mode MLP
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_flyKD+random100_ptrain_MLP --set_seed 46 --teacher_size 130 --student_size 150 --iter 2\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --on_the_fly_KD --ptrain --reparam_mode MLP
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_flyKD+random100_ptrain_MLP --set_seed 47 --teacher_size 130 --student_size 150 --iter 2\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --on_the_fly_KD --ptrain --reparam_mode MLP
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_flyKD+random100_ptrain_dropout --set_seed 45 --teacher_size 130 --student_size 150 --iter 2\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --on_the_fly_KD --ptrain --dropout 0.1
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_flyKD+random100_ptrain_dropout --set_seed 46 --teacher_size 130 --student_size 150 --iter 2\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --on_the_fly_KD --ptrain --dropout 0.1
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_flyKD+random100_ptrain_dropout --set_seed 47 --teacher_size 130 --student_size 150 --iter 2\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --on_the_fly_KD --ptrain --dropout 0.1
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random200_ptrain --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 200 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random200_ptrain --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 200 --on_the_fly_KD --ptrain

# python -u noisy_student.py proper_150_e1200_90 --set_seed 48 --teacher_size 150
# python -u noisy_student.py proper_150_e1200_90 --set_seed 49 --teacher_size 150
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_ptrain --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_ptrain --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_ptrain --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_ptrain --set_seed 48 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_ptrain --set_seed 49 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --ptrain
# python -u noisy_student.py save_pretrain/proper_150_e1200_90 --set_seed 45 --teacher_size 150
# python -u noisy_student.py save_pretrain/proper_150_e1200_90 --set_seed 46 --teacher_size 150
# python -u noisy_student.py save_pretrain/proper_150_e1200_90 --set_seed 47 --teacher_size 150
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain --set_seed 48 --teacher_size 130 --student_size 80 --iter 1\
#                             --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain --set_seed 49 --teacher_size 130 --student_size 80 --iter 1\
#                             --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                             --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                             --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                             --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_novaltest --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --fly_no_val_test
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_novaltest --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --fly_no_val_test
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_novaltest --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --fly_no_val_test
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_novaltest --set_seed 48 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --fly_no_val_test
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random25_novaltest --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 25 --on_the_fly_KD --fly_no_val_test
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random25_novaltest --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 25 --on_the_fly_KD --fly_no_val_test
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random25_novaltest --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 25 --on_the_fly_KD --fly_no_val_test
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random50 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                             --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random50 --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                             --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random100 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                             --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random100 --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                             --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --ptrain
# python -u noisy_student.py ptrain80 --set_seed 45 --teacher_size 80 --save_model
# python -u noisy_student.py ptrain80 --set_seed 46 --teacher_size 80 --save_model
# python -u noisy_student.py ptrain80 --set_seed 47 --teacher_size 80 --save_model
# python -u noisy_student.py ptrain80 --set_seed 48 --teacher_size 80 --save_model
# python -u noisy_student.py ptrain80 --set_seed 49 --teacher_size 80 --save_model
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_str3 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --strong_scores 3
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_str3 --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --strong_scores 3
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_str3 --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --strong_scores 3
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_flyKD+random50_ptrain_MLP_test --set_seed 45 --teacher_size 130 --student_size 150 --iter 2\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --ptrain --reparam_mode MLP
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_ptrain --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_ptrain --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_ptrain --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_ptrain --set_seed 48 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_ptrain --set_seed 49 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random100_ptrain --set_seed 48 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random100_ptrain --set_seed 49 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random200_ptrain --set_seed 48 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 200 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random200_ptrain --set_seed 49 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 200 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random50 --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random100 --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --ptrain
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_flyKD+random50_ptrain --set_seed 46 --teacher_size 130 --student_size 150 --iter 2\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_flyKD+random50_ptrain --set_seed 47 --teacher_size 130 --student_size 150 --iter 2\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_flyKD+random50_ptrain --set_seed 48 --teacher_size 130 --student_size 150 --iter 2\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_flyKD+random50_ptrain --set_seed 49 --teacher_size 130 --student_size 150 --iter 2\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_flyKD+random50_ptrain --set_seed 45 --teacher_size 130 --student_size 150 --iter 2\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --ptrain
# python -u noisy_student.py proper_150_e1200_95_MLP --set_seed 45 --teacher_size 150 --force_reparam --force_iter0 --reparam_mode MLP
# python -u noisy_student.py proper_150_e1200_95_MLP --set_seed 46 --teacher_size 150 --force_reparam --force_iter0 --reparam_mode MLP
# python -u noisy_student.py proper_150_e1200_95_MLP --set_seed 47 --teacher_size 150 --force_reparam --force_iter0 --reparam_mode MLP
# python -u noisy_student.py proper_150_e1200_95_MLP --set_seed 48 --teacher_size 150 --force_reparam --force_iter0 --reparam_mode MLP
# python -u noisy_student.py proper_150_e1200_95_MLP --set_seed 49 --teacher_size 150 --force_reparam --force_iter0 --reparam_mode MLP
## Quick Run:
# 1
# python -u noisy_student.py save_pretrain/proper_80_e1200_90 --set_seed 45 --teacher_size 80 --force_iter0 --only_pretrain
# 2
# python -u noisy_student.py save_pretrain/proper_80_e1200_90 --set_seed 46 --teacher_size 80 --force_iter0 --only_pretrain
# 3
# 4
# python -u noisy_student.py save_pretrain/proper_80_e1200_90 --set_seed 47 --teacher_size 80 --force_iter0 --only_pretrain
# 5
# python -u noisy_student.py ptrain80 --set_seed 48 --teacher_size 80 --force_iter0
# 6
# python -u noisy_student.py ptrain80 --set_seed 49 --teacher_size 80 --force_iter0

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random50 --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random100 --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_str3 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --strong_scores 3
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_str3 --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --strong_scores 3
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_str3 --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --strong_scores 3
## Running
# 1
# python -u noisy_student.py save_pretrain/proper_130_e1200_90 --set_seed 45 --teacher_size 130 --force_iter0 --only_pretrain
# 2
# python -u noisy_student.py save_pretrain/proper_130_e1200_90 --set_seed 46 --teacher_size 130 --force_iter0 --only_pretrain
# 3
# python -u noisy_student.py save_pretrain/proper_130_e1200_90 --set_seed 47 --teacher_size 130 --force_iter0 --only_pretrain
# 4
# python -u noisy_student.py save_pretrain/proper_150_e1200_90 --set_seed 45 --teacher_size 150 --force_iter0 --only_pretrain
# 5
# python -u noisy_student.py save_pretrain/proper_150_e1200_90 --set_seed 46 --teacher_size 150 --force_iter0 --only_pretrain
# 6
# python -u noisy_student.py save_pretrain/proper_150_e1200_90 --set_seed 47 --teacher_size 150 --force_iter0 --only_pretrain
## -----------------
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_all --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random100_ptrain --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_novaltest --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --fly_no_val_test
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random200_ptrain --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 200 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random200_ptrain --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 200 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random200_ptrain --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 200 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random100_ptrain --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random100_ptrain --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100 --on_the_fly_KD --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_all --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_novaltest --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --fly_no_val_test
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_novaltest --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --debug --fly_no_val_test 
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_novaltest --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --debug --fly_no_val_test

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_ptrain --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --debug --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_ptrain --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --debug --ptrain
# python -u noisy_student.py proper_80_95 --set_seed 46 --teacher_size 80 --debug --force_iter0
# python -u noisy_student.py proper_80_95 --set_seed 47 --teacher_size 80 --debug --force_iter0

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_str2_masked_neg --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --debug --strong_scores 2

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_str3_masked_neg --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --debug --strong_scores 3

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_ptrain --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --debug

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_ptrain --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --debug

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --debug

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50 --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --debug

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_str3_masked_eneg --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
                # --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --debug --extra_neg_sampling --strong_scores 3

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_str2_masked_eneg --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --debug --extra_neg_sampling --strong_scores 2
                
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_str3_masked --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --debug --strong_scores 3

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_str4_masked --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --debug --strong_scores 4

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_str4_masked_eneg --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                 --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --debug --strong_scores 4 --extra_neg_sampling

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_str1_eneg --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --debug --extra_neg_sampling

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_str2_full_eneg --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --debug --extra_neg_sampling --exlucde_valid_test

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_flyKD+random50_str3_full_eneg --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --on_the_fly_KD --debug --extra_neg_sampling --exlucde_valid_test


# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random50 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 50 --ptrain --debug

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_all+random50 --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --random_pseudo_k 50 --on_the_fly_KD

#------------------------------------------

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_all+random50 --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --random_pseudo_k 50

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_all+random50 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --random_pseudo_k 50
## below not running
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+pretrain --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --pseudo_pretrain

# python -u noisy_student.py noisy_student_130to80_soft_pseudo+random50+pretrain --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --random_pseudo_k 50 --pseudo_pretrain

# python -u noisy_student.py noisy_student_130to80_soft_pseudo+random50+pretrain --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --random_pseudo_k 50 --pseudo_pretrain

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_all+random50 --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --random_pseudo_k 50
## below not running
# python -u noisy_student.py noisy_student_130to80_soft_pseudo+random50+pretrain --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --random_pseudo_k 50 --pseudo_pretrain

# python -u noisy_student.py noisy_student_130to80_soft_pseudo+random50 --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --random_pseudo_k 50
# python -u noisy_student.py noisy_student_130to80_soft_pseudo+random200 --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --random_pseudo_k 200


#checkout
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+pretrain --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --pseudo_pretrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+pretrain --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --pseudo_pretrain

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_strongscores8 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --strong_scores 8
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_strongscores8 --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --strong_scores 8
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_strongscores8 --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --strong_scores 8
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_strongscores8_bl --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --strong_scores 8 --balance_loss
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_strongscores8_bl --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --strong_scores 8 --balance_loss
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_strongscores8_bl --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --strong_scores 8 --balance_loss

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random200 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --random_pseudo_k 200
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random200 --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --random_pseudo_k 200
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random200 --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --random_pseudo_k 200
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random200_bl --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --random_pseudo_k 200 --balance_loss
## check below 2
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random200_bl --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --random_pseudo_k 200 --balance_loss
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random200_bl --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --random_pseudo_k 200 --balance_loss

## Running
## include_all_1000 does not work, because it creates too many labels
# python -u noisy_student.py noisy_student_130to80_soft_pseudo+random200 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --random_pseudo_k 200
# python -u noisy_student.py noisy_student_130to80_soft_pseudo+random200 --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --random_pseudo_k 200
# python -u noisy_student.py noisy_student_130to80_soft_pseudn+random1000 --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --random_pseudo_k 1000
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random1000 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --random_pseudo_k 1000
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random1000 --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --random_pseudo_k 1000
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random1000 --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --random_pseudo_k 1000                                      

# python -u noisy_student.py noisy_student_130to100_soft_pseudo+random50 --set_seed 45 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --random_pseudo_k 50
## Run all below
# python -u noisy_student.py noisy_student_130to100_soft_pseudo+random50 --set_seed 46 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --random_pseudo_k 50
# python -u noisy_student.py noisy_student_130to100_soft_pseudo+random50 --set_seed 47 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --random_pseudo_k 50
# python -u noisy_student.py noisy_student_130to80_soft_pseudo+random50 --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --random_pseudo_k 50
# python -u noisy_student.py noisy_student_130to80_soft_pseudo+random50 --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --random_pseudo_k 50

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain_strongscores8 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --strong_scores 8
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain_strongscores8 --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --strong_scores 8
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain_strongscores8 --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --strong_scores 8
## check  2 below
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain_strongscores8_bl --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --strong_scores 8 --balance_loss
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain_strongscores8_bl --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --strong_scores 8 --balance_loss
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain_strongscores8_bl --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --strong_scores 8 --balance_loss

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain_strongscores9 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --strong_scores 9
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain_strongscores9 --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --strong_scores 9
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain_strongscores9 --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --strong_scores 9
## check below 2
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain_strongscores9_bl --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --strong_scores 9 --balance_loss
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain_strongscores9_bl --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --strong_scores 9 --balance_loss
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain_strongscores9_bl --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --strong_scores 9 --balance_loss



# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain_strongscores6 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --ptrain --strong_scores 6

## Dummy training: should be around the corresponding training
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain_bl --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --ptrain --balance_loss
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random200_bl --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --random_pseudo_k 200 --balance_loss

## include all pseudo allows relatioNS between any disease and drug entities. ptrain on the other hand, only allows pseudo scores between disease and drug entities that have labels during training. 
# python -u noisy_student.py noisy_student_130to80_soft_pseudo+random50 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --include_all_pseudo --random_pseudo_k 50
## Need to run
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_strongscores8 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --include_all_pseudo --strong_scores 8

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random200_bl --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --random_pseudo_k 200 --balance_loss 
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random200 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --random_pseudo_k 200 

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_strongscores7 --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --include_all_pseudo --strong_scores 7
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_strongscores7 --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --include_all_pseudo --strong_scores 7

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random50 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --random_pseudo_k 50
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random50 --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --random_pseudo_k 50 
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain+random50 --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --limited_neg_pseudo_sampling --soft_pseudo --ptrain --random_pseudo_k 50 


# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain_strongscores6 --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --ptrain --strong_scores 6
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain_strongscores6 --set_seed 47 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --ptrain --strong_scores 6

# python -u noisy_student.py ptrain150 --set_seed 47 --teacher_size 150 --iter 0

# python -u noisy_student.py noisy_student_130to100_soft_pseudo_ptrain_strongscores10 --set_seed 45 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --include_ptrain --strong_scores 10
# python -u noisy_student.py noisy_student_130to100_soft_pseudo_ptrain_strongscores10 --set_seed 46 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --include_ptrain --strong_scores 10

# python -u noisy_student.py noisy_student_130to100_soft_pseudo_ptrain_strongscores10 --set_seed 47 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --include_ptrain --strong_scores 10
# python -u noisy_student.py ptrain80 --set_seed 47 --teacher_size 80 --iter 0

# python -u noisy_student.py noisy_student_130to150_soft_pseudo --set_seed 45 --teacher_size 130 --student_size 150 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --ptrain
# python -u noisy_student.py noisy_student_130to150_soft_pseudo --set_seed 46 --teacher_size 130 --student_size 150 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --ptrain
# python -u noisy_student.py noisy_student_130to150_soft_pseudo --set_seed 47 --teacher_size 130 --student_size 150 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --ptrain


# python -u noisy_student.py ptrain80 --set_seed 45 --teacher_size 80 --iter 0
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain_RBF --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --ptrain --LSP RBF --LSP_size partial --all_layers_LSP --sigmas 3

# python -u noisy_student.py ptrain80 --set_seed 46 --teacher_size 80 --iter 0
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain_RBF --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --ptrain --LSP RBF --LSP_size partial --all_layers_LSP --sigmas 3

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain_DRBF --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --ptrain --LSP RBF --LSP_size partial --all_layers_LSP --sigmas 0.2 3 --all_layers_LSP

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain_DRBF --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --ptrain --LSP RBF --LSP_size partial --all_layers_LSP --sigmas 0.2 3 --all_layers_LSP

# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --ptrain
# python -u noisy_student.py noisy_student_130to80_soft_pseudo_ptrain --set_seed 46 --teacher_size 130 --student_size 80 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --ptrain

# python -u noisy_student.py ptrain100 --set_seed 45 --teacher_size 100 --iter 0
# python -u noisy_student.py ptrain100 --set_seed 46 --teacher_size 100 --iter 0
# python -u noisy_student.py ptrain100 --set_seed 47 --teacher_size 100 --iter 0


# python -u noisy_student.py ptrain_noisy_student_130to100_soft_pseudo_double_RBF_partial --set_seed 45 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --ptrain --LSP RBF --LSP_size partial --all_layers_LSP

# python -u noisy_student.py ptrain_noisy_student_130to100_soft_pseudo_double_RBF_partial --set_seed 46 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --ptrain --LSP RBF --LSP_size partial --all_layers_LSP

# python -u noisy_student.py ptrain_noisy_student_130to100_soft_pseudo_double_RBF_partial --set_seed 47 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --ptrain --LSP RBF --LSP_size partial --all_layers_LSP

# python -u noisy_student.py ptrain_noisy_student_130to100_no_pseudo_RBF_partial --set_seed 45 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --no_pseudo --ptrain --LSP RBF --LSP_size partial
# python -u noisy_student.py ptrain_noisy_student_130to100_no_pseudo_RBF_partial --set_seed 46 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --no_pseudo --ptrain --LSP RBF --LSP_size partial

# python -u noisy_student.py ptrain_noisy_student_130to100_no_pseudo_RBF_partial --set_seed 47 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --no_pseudo --ptrain --LSP RBF --LSP_size partial
# python -u noisy_student.py ptrain_noisy_student_130to100_soft_pseudo_strongscores10 --set_seed 45 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --include_ptrain --strong_scores 10

# python -u noisy_student.py ptrain_noisy_student_130to100_soft_pseudo_strongscores10 --set_seed 46 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --include_ptrain --strong_scores 10
# python -u noisy_student.py ptrain_noisy_student_130to100_soft_pseudo+random50 --set_seed 45 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --include_ptrain --random_pseudo_k 50 
# python -u noisy_student.py ptrain_noisy_student_130to100_soft_pseudo+random50 --set_seed 46 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --include_ptrain --random_pseudo_k 50



# python -u noisy_student.py ptrain/noisy_student_130to100_soft_pseudo_RBF_partial --set_seed 45 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --ptrain --LSP RBF --LSP_size partial

# python -u noisy_student.py ptrain/noisy_student_130to100_soft_pseudo_RBF_partial --set_seed 46 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --ptrain --LSP RBF --LSP_size partial



# python -u noisy_student.py noisy_student_130to100_soft_pseudo_inog_train --set_seed 45 --teacher_size 130 --student_size 100 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo --ptrain
# python -u noisy_student.py noisy_student_130to100_soft_pseudo_inog_train --set_seed 46 --teacher_size 130 --student_size 100 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo --ptrain
# python -u noisy_student.py noisy_student_130to100_soft_pseudo_inog_train --set_seed 47 --teacher_size 130 --student_size 100 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo --ptrain

# python -u noisy_student.py noisy_student_130to150_soft_pseudo_inog_train --set_seed 45 --teacher_size 130 --student_size 150 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo --ptrain
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_inog_train --set_seed 46 --teacher_size 130 --student_size 150 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo --ptrain

# python -u noisy_student.py noisy_student_130to150_soft_pseudo_inog_train --set_seed 47 --teacher_size 130 --student_size 150 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo --ptrain
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_inog_train_MLP --set_seed 45 --teacher_size 130 --student_size 150 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo --ptrain --reparam_mode MLP

# python -u noisy_student.py noisy_student_130to150_soft_pseudo_inog_train_MLP --set_seed 46 --teacher_size 130 --student_size 150 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo --ptrain --reparam_mode MLP
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_inog_train_MLP --set_seed 47 --teacher_size 130 --student_size 150 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo --ptrain --reparam_mode MLP

## Errored
# python -u noisy_student.py noisy_student_130to100_soft_pseudo_inog --set_seed 48 --teacher_size 130 --student_size 100 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo
# python -u noisy_student.py noisy_student_130to100_soft_pseudo_inog --set_seed 49 --teacher_size 130 --student_size 100 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo
# python -u noisy_student.py proper_150_95 --set_seed 46 --teacher_size 150
# python -u noisy_student.py proper_150_95 --set_seed 47 --teacher_size 150 
# python -u noisy_student.py proper_150_95_MLP --set_seed 46 --teacher_size 150 --force_reparam --reparam_mode MLP
# python -u noisy_student.py proper_150_95_MLP --set_seed 47 --teacher_size 150 --force_reparam --reparam_mode MLP
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_inog --set_seed 46 --teacher_size 130 --student_size 150 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo
# python -u noisy_student.py proper_100_95 --set_seed 48 --teacher_size 100 
# python -u noisy_student.py proper_100_95 --set_seed 49 --teacher_size 100 


# python -u noisy_student.py noisy_student_130to100_soft_pseudo_inog --set_seed 47 --teacher_size 130 --student_size 100 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_inog --set_seed 47 --teacher_size 130 --student_size 150 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_inog_MLP --set_seed 45 --teacher_size 130 --student_size 150 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo --reparam_mode MLP
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_inog_MLP --set_seed 46 --teacher_size 130 --student_size 150 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo --reparam_mode MLP
## ----slightly more correct(?) negative sampling above this line--------------
# python -u noisy_student.py KD130_to_100_soft_pseudo_inog_partial_RBF --set_seed 45 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --LSP RBF --LSP_size partial --generate_inog
# python -u noisy_student.py KD130_to_100_soft_pseudo_inog_partial_RBF+random50 --set_seed 45 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --LSP RBF --LSP_size partial --random_pseudo_k 50
# python -u noisy_student.py KD130_to_100_soft_pseudo_inog_partial_RBF+random100 --set_seed 45 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --LSP RBF --LSP_size partial --random_pseudo_k 100
# python -u noisy_student.py KD130_to_100_soft_pseudo_inog+strong_scores5 --set_seed 45 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --strong_scores 5
# python -u noisy_student.py noisy_student_130to100_soft_pseudo_inog_pneg --set_seed 45 --teacher_size 130 --student_size 100 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_inog --set_seed 45 --teacher_size 130 --student_size 150 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo
# python -u noisy_student.py noisy_student_130to100_soft_pseudo_inog --set_seed 46 --teacher_size 130 --student_size 100 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_inog --set_seed 46 --teacher_size 130 --student_size 150 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo
## ----------proper pseudo labels above this line------------
# python -u noisy_student.py KD130_to_100_soft_pseudo_inog_partial_RBF --set_seed 46 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --LSP RBF --LSP_size partial --generate_inog


# python -u noisy_student.py KD130_to_100_soft_pseudo_inog_partial_RBF --set_seed 47 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --LSP RBF --LSP_size partial --generate_inog


# python -u noisy_student.py KD130_to_100_soft_pseudo_inog+top_bottom10 --set_seed 45 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --k_top_bottom_candidates 10



# python -u noisy_student.py proper_150_95 --set_seed 45 --teacher_size 150 --force_reparam --reparam_mode MLP
# python -u noisy_student.py proper_150_95 --set_seed 45 --teacher_size 150
# python -u noisy_student.py proper_150_95_MLP --set_seed 45 --teacher_size 150 --force_reparam --reparam_mode MLP
# python -u noisy_student.py KD130_to_100_soft_pseudo_inog+random100 --set_seed 46 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100

# python -u noisy_student.py KD130_to_100_soft_pseudo_inog+random150 --set_seed 45 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --random_pseudo_k 150

# python -u noisy_student.py KD130_to_100_soft_pseudo_inog+random200 --set_seed 45 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --random_pseudo_k 200

# python -u noisy_student.py proper_100_95 --set_seed 46 --teacher_size 100
# python -u noisy_student.py proper_100_95 --set_seed 47 --teacher_size 100

# python -u noisy_student.py KD130_to_100_soft_pseudo_inog --set_seed 45 --teacher_size 130 --student_size 100 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo
# python -u noisy_student.py KD130_to_100_soft_pseudo_inog --set_seed 46 --teacher_size 130 --student_size 100 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo
# python -u noisy_student.py KD130_to_100_soft_pseudo_inog --set_seed 47 --teacher_size 130 --student_size 100 --iter 1\
                                        # --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo

# python -u noisy_student.py KD130_to_100_soft_pseudo_inog+random20 --set_seed 45 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --random_pseudo_k 20
# python -u noisy_student.py KD130_to_100_soft_pseudo_inog+random20 --set_seed 46 --teacher_size 130 --student_size 100 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --random_pseudo_k 20
# python -u noisy_student.py KD130_to_100_soft_pseudo_inog+random20 --set_seed 47 --teacher_size 130 --student_size 100 --iter 1\
                                        # --use_og --neg_pseudo_sampling --soft_pseudo --random_pseudo_k 20

# python -u noisy_student.py noisy_student_130to150_soft_pseudo_inog --set_seed 45 --teacher_size 130 --student_size 150 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_inog --set_seed 46 --teacher_size 130 --student_size 150 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_inog --set_seed 47 --teacher_size 130 --student_size 150 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo

# python -u noisy_student.py noisy_student_130to150_soft_pseudo_inog+random20 --set_seed 45 --teacher_size 130 --student_size 150 --iter 1\
                                        # --use_og --neg_pseudo_sampling --soft_pseudo --random_pseudo_k 20
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_inog+random20 --set_seed 46 --teacher_size 130 --student_size 150 --iter 1\
                                        # --use_og --neg_pseudo_sampling --soft_pseudo --random_pseudo_k 20
# python -u noisy_student.py noisy_student_130to150_soft_pseudo_inog+random20 --set_seed 47 --teacher_size 130 --student_size 150 --iter 1\
#                                         --use_og --neg_pseudo_sampling --soft_pseudo --random_pseudo_k 20

##--------------------

# python -u noisy_student.py KD128_to_100 --set_seed 45 --teacher_size 130 --student_size 100 --iter 1
# python -u noisy_student.py KD128_to_100 --set_seed 46 --teacher_size 130 --student_size 100 --iter 1
# python -u noisy_student.py KD128_to_100 --set_seed 47 --teacher_size 130 --student_size 100 --iter 1

# python -u noisy_student.py proper_130 --set_seed 28 --teacher_size 130
# python -u noisy_student.py proper_130 --set_seed 63 --teacher_size 130
# python -u noisy_student.py proper_130 --set_seed 86 --teacher_size 130


# python -u noisy_student.py KD128_to_100 --set_seed 45 --teacher_size 130 --student_size 100 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo
# python -u noisy_student.py KD128_to_100 --set_seed 46 --teacher_size 130 --student_size 100 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo
# python -u noisy_student.py KD128_to_100 --set_seed 47 --teacher_size 130 --student_size 100 --iter 1\
#                                         --generate_inog --use_og --neg_pseudo_sampling --soft_pseudo


# python -u noisy_student.py noisy_student_130 --set_seed 54 --teacher_size 130 --iter 0 

# python -u noisy_student.py proper_80 --repeat 3 --iter 0 --generate_inog \
#         --use_og --neg_pseudo_sampling --soft_pseudo --teacher_size 80
# python -u noisy_student.py KD80_soft_pseudo_inog_pneg --repeat 3 --iter 1 --generate_inog \
#         --use_og --neg_pseudo_sampling --soft_pseudo --teacher_size 100 --student_size 80

# python -u noisy_student.py KD80_soft_pseudo_inog_pneg_random20 --repeat 3 --iter 1 --generate_inog \
#         --use_og --neg_pseudo_sampling --soft_pseudo --teacher_size 100 --student_size 80 --random_pseudo_k 20

# python -u noisy_student.py soft_pseudo_inog_pneg --repeat 3 --iter 2 --generate_inog \
#         --use_og --neg_pseudo_sampling --soft_pseudo --teacher_size 100 --student_size 120

# python -u noisy_student.py soft_pseudo_inog+random20_pneg --repeat 3 --iter 2 --generate_inog \
#         --use_og --neg_pseudo_sampling --soft_pseudo --teacher_size 100 --student_size 120 --random_pseudo_k 20
# -------------

# python -u noisy_student.py proper_100_95 --repeat 3 --iter 0 --teacher_size 100
# python -u noisy_student.py soft_pseudo_inog_pneg_wog_95 --repeat 1 --iter 2 --generate_inog \
#         --use_og --neg_pseudo_sampling --soft_pseudo --student_size 120 --teacher_size 100

# python -u noisy_student.py soft_pseudo_inog_pneg_wog_95_MLP --repeat 1 --iter 2 --generate_inog \
#         --use_og --neg_pseudo_sampling --soft_pseudo --student_size 120 --teacher_size 100 --reparam_mode MLP

# python -u noisy_student.py soft_pseudo_inog_pneg_wog_95_RMLP --repeat 3 --iter 2 --generate_inog \
#         --use_og --neg_pseudo_sampling --soft_pseudo --student_size 120 --reparam_mode RMLP 

# python -u noisy_student.py proper_100_95 --repeat 3 --iter 0 --teacher_size 100

# python -u noisy_student.py proper_120_95 --repeat 2 --iter 0 --teacher_size 120
# python -u noisy_student.py force_proper_120_95_MLP --repeat 3 --iter 0 --teacher_size 120 --reparam_mode MLP

# nohup python -u noisy_student.py --reparam_mode MLP --student_size 120 > ./logs/120/0.95lr_MLP.txt
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og_dataset.csv --neg_pseudo_sampling --soft_pseudo --student_size 120 --use_og > ./logs/120/soft_pseudo_inog_pneg_wog_0.95lr.txt
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og_dataset.csv --neg_pseudo_sampling --soft_pseudo --student_size 120 --use_og --reparam_mode MLP > ./logs/120/soft_pseudo_inog_pneg_wog_0.95lr_MLP.txt
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og+random100.csv --soft_pseudo --student_size 120 --use_og > ./logs/120/soft_pseudo_inog+random100_pneg_wog_0.95lr.txt
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og+random100.csv --soft_pseudo --student_size 120 --use_og > ./logs/120/soft_pseudo_inog+random100_pneg_wog_0.95lr_MLP.txt

# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og_dataset.csv --neg_pseudo_sampling --LSP RBF --LSP_size partial --soft_pseudo --student_size 100 --use_og > ./logs/LSP/soft_pseudo_inog_pneg_wog_0.95lr_RBF.txt
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og+random100.csv --LSP RBF --LSP_size partial --soft_pseudo --student_size 120 --use_og > ./logs/LSP/soft_pseudo_inog+random100_pneg_wog_0.95lr_RBF.txt

# nohup python -u noisy_student.py --student_size 120 > ./logs/120/reference_proper2.txt
## Try random+100 too, since it seems like having restrained or not doesn't really matter.
# nohup python -u noisy_student.py > ./logs/reference_performances/restrained_1__0.95lr.txt
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og+random100.csv --soft_pseudo --student_size 100 --use_og > ./logs/soft_pseudo_inog+random100_pneg_wog_fromrestr_0.95lr.txt

# nohup python -u noisy_student.py > ./logs/reference_performances/restrained_1_e1000__0.9lr.txt
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_restrained_inog_dataset.csv --neg_pseudo_sampling --soft_pseudo --student_size 100 --use_og > ./logs/soft_pseudo_inog__pneg_wog_e1000_fromrestr_0.9lr.txt

# nohup python -u noisy_student.py --psuedo_label_fname pscores_all_restrained_saveG.csv --deg inf > ./logs/psuedo_scores_all_restrained_saveG.txt 2>&1 ## generating ALL psuedo_labels
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og_dataset.csv --neg_pseudo_sampling --soft_pseudo --student_size 100 --use_og > ./logs/test.txt
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og+random100.csv --soft_pseudo --student_size 100 --use_og > ./logs/soft_pseudo_inog+random100_pneg_wog_e1000_NoDPMforood.txt
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og+random100.csv --soft_pseudo --student_size 100 --use_og > ./logs/soft_pseudo_inog+random100_pneg_wog_e1000_allweakDPM.txt

# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og_dataset.csv --neg_pseudo_sampling --soft_pseudo --student_size 100 --use_og > ./logs/soft_pseudo_inog_pneg_wog_txt
# nohup python -u noisy_student.py > ./logs/reference_performances/restrained_1_txt
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_scores_in_og+random100.csv --soft_pseudo --student_size 100 --use_og > ./logs/soft_pseudo_inog+random100_pneg_wog_txt

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

# nohup python -u noisy_student.py --save_model seed_1_restrained_saveG > ./logs/save_pre_restrained_save_G.txt ## want to make sure the sum of the positive matches (reproduced).

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

# Note: 2>&1 meaNS log the error as well

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
# nohup python -u noisy_student.py --psuedo_label_fname psuedo_labels_75000.csv --student_size 100 > ./logs/same_size_student.txt 2>&1 ## Observe that the knowledge traNSfer does not exist and gives the same performance as the previous two.

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


######################################### Script Examples #########################################
## OG FlyKD on seed 45
python -u noisy_student.py test/NS_130to80_soft_pseudo_flyKD_curr1 --set_seed 45 --teacher_size 130 --student_size 80 --iter 1\
    --limited_neg_pseudo_sampling --soft_pseudo --random_pseudo_k 100000 --on_the_fly_KD --rel_multinomial_ptrain --epochs 2000 --curriculum1

## student model baseline without KD
python -u noisy_student.py test/proper_80_e1200_90 --set_seed 45 --teacher_size 80
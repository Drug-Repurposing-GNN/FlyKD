from txgnn import TxData, TxGNN, TxEval
from accelerate.utils import set_seed as rng_set_seed
import numpy as np
import time
import pandas as pd
from noisy_student_utils import *
from txgnn.utils import *
import pickle
import sys
import pprint
import wandb
import argparse
import os
import random
import datetime
'''
    Let's first try one iteration to increase performance.
'''
if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', default=0.0, type=float, help="add dropout at layer 1")
    parser.add_argument('--reparam_mode', default=False, help='choose from {MLP, RMLP, MPNN}')
    parser.add_argument('--psuedo_label_fname', default=None, help='choose from {psuedo_labels_75000.csv, }') ## is default None? 
    parser.add_argument('--split', default='complex_disease', help='choose from {complex_disease, ...}')
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--k_top_bottom_candidates', type=float)
    parser.add_argument('--k_top_candidates', type=float)
    parser.add_argument('--psuedo_edges', action='store_true')
    parser.add_argument('--student_size', default=120, type=int)
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--deg', default=float('inf'), type=float) ## 'inf' for all diseases?
    parser.add_argument('--epochs', type=int) ## 'inf' for all diseases?
    parser.add_argument('--set_seed', default=1, type=int)
    parser.add_argument('--soft_pseudo', action='store_true')
    parser.add_argument('--train_then_generate', action='store_true')
    parser.add_argument('--use_high_degree', action='store_true')
    parser.add_argument('--neg_pseudo_sampling', action='store_true')
    parser.add_argument('--generate_indication', action='store_true')
    parser.add_argument('--generate_contraindication', action='store_true')
    parser.add_argument('--no_dpm', action='store_true', help="disables disease pooling mechanism during evaluation")
    parser.add_argument('--use_og', action='store_true', help="use original loss as additional loss in pseudo training")
    parser.add_argument('--LSP', type=str, help="additional Local Sturcture (similarity vector) loss using KL div. Mode: {cosine, L2, RBF, Poly}")
    parser.add_argument('--LSP_size', type=str, help="use full or partial relations to compute the LS vector?")
    parser.add_argument('--T', type=float, default=1, help="temperature for pseudo loss")
    parser.add_argument("--generate_inog", action="store_true", help="Generate only on existing relations (train, val, data) for quicker time")
    parser.add_argument("--repeat", type=int, default=1, help="repeat n many times with random seed to create confidence interval")
    parser.add_argument("--iter", type=int, default=0, help="iteration for noisy-student framework. 0 means only train teacher method")
    parser.add_argument("--teacher_size", type=int, default=100, help="the name of the filename or directory for eval_perf")
    parser.add_argument("fname", type=str, help="the name of the filename or directory for eval_perf")
    parser.add_argument("--debug", action="store_true", help="send files to debug folder")
    parser.add_argument("--debug2", action="store_true", help="send files to debug folder")
    parser.add_argument("--random_pseudo_k", type=int, help="number of pseudo labels on random relations. (Uses disease pooling mechanism)")
    parser.add_argument("--no_wandb", action="store_true", help="turn off wandb")
    parser.add_argument('--save_model', action="store_true", help="save the trained model to reuse as teahcer model in the future")
    parser.add_argument('--force_reparam', action="store_true", help="Needed to turn on reparam mode on the first iteration")
    parser.add_argument('--strong_scores', type=float, help="extra pseudo labels on relations that has high confidence: abs(value) > strong_scores")
    parser.add_argument('--ptrain', action="store_true", help="generate pseudo labels only on training set")
    parser.add_argument('--no_pseudo', action="store_true", help="disable pseudo labels")
    parser.add_argument('--all_layers_LSP', action="store_true", help="compute auxilary loss on LSP on all aggregation layers")
    parser.add_argument('--include_all_pseudo', action="store_true", help="When you are generating pseudo labels from ANY disease, this ensures relation between unseen entities\
can be retained")
    parser.add_argument('--sigmas', nargs='+', type=float, help="[0.2, 3] seems to decrease the entropy of similarity score the most")
    parser.add_argument('--limited_neg_pseudo_sampling', action='store_true', help="allows negative sampling to only happen to existing relation")
    parser.add_argument("--balance_loss", action="store_true", help="Balance out the loss from pseudo labels based on existing ratio between ind vs cind")
    parser.add_argument("--pseudo_pretrain", action="store_true", help="pretrain with pseudo labels as well") 
    parser.add_argument("--exclude_valid_test", action="store_true", help="do not use valid/test dataset at all")
    parser.add_argument("--on_the_fly_KD", action="store_true", help="use teacher model on the fly to generate different pseudo label each time")
    parser.add_argument("--force_iter0", action="store_true", help="")
    parser.add_argument("--fly_no_val_test", action="store_true", help="")
    parser.add_argument("--extra_neg_sampling", action="store_true", help="")
    parser.add_argument("--only_pretrain", action="store_true", help="used to store pretraining phase of a model")
    parser.add_argument("--rel_ptrain", action="store_true", help="puts relation specific ptrain constraint")
    parser.add_argument("--rel_multinomial_ptrain", action="store_true", help="uses multinomial distribution based on entities degrees to sample random graph")
    parser.add_argument("--multinomial_ptrain", action="store_true", help="uses multinomial distribution based on entities degrees to sample random graph")
    parser.add_argument("--scale_neg_loss", action="store_true", help="scales neg loss according to the positiveness of soft pseudo")
    parser.add_argument("--no_curriculum", action="store_true", help="ratio between og loss, random pseudo loss, and train pseudo loss")
    parser.add_argument("--curriculum1", action="store_true", help="Curriculum Learning: Original + Pseudo + Random")
    parser.add_argument("--curriculum2", action="store_true", help="Curriculum Learning: Original + Pseudo")
    parser.add_argument("--curriculum3", action="store_true", help="probably takes out pseduo train data")
    parser.add_argument("--curriculum1_stepwise", action="store_true", help="Step wise version of curriculum1")
    parser.add_argument("--force_finetune_iter0", action="store_true", help="force finetuning phase of iter0")
    parser.add_argument("--fixed_flyKD", action="store_true", help="fix flykd's random generation for ablation study")
    parser.add_argument("--no_ptrain_flyKD", action="store_true", help="Only use FlyKD's random graph")
    parser.add_argument("--occasional_flyKD", type=int, help="How frequent should you generate random graph")
    parser.add_argument("--modified_multinomial", action="store_true", help="modify (lower for now) the entropy of multinomial prob distribution")

    ## passing in args
    args = parser.parse_args()
    args.date = datetime.datetime.now()
    if args.debug:
        args.fname = f"debug/{args.fname}"
    elif args.debug2:
        args.fname = f"debug2/{args.fname}"
        args.debug = True
        
    ## to get notified when training is finished
    if args.no_wandb is None:
        proj_name = 'TxGNN'
        exp_name = 'TxGNN'
        wandb.init(project=proj_name, name=exp_name)  
        wandb.config.update(vars(args))

    ## This for loop is deprecated
    for _ in range(args.repeat):
        ## setting seed for rng and data split
        seed = args.set_seed
        rng_set_seed(seed)
        for i, _ in enumerate(range(args.iter + 1)):
            save_dir, log_file = init_logfile(i, seed, args)

            diff_epochs = f"e{args.epochs}" if args.epochs else ""
            pretrained_path = f"./pretrained_models/{args.teacher_size}_{diff_epochs}/{seed}" 
            ith_model_ckpt = f"{save_dir}{i}_model_ckpt/" 
                
            args.i = i
            if i == 0:
                ## only train if the pretrained model does not exist
                if not os.path.exists(pretrained_path) or args.force_iter0 or args.force_finetune_iter0:
                    if args.iter == 0 and not args.save_model:
                        pretrained_path = None
                    ## Use teacher size for first model (teacher model)
                    size = args.teacher_size 
                    ## training function
                    print("Training *without* pseudo labels")
                    train_w_psuedo_labels(None, None, seed, pretrained_path, args, size=size, i=i)
                else:
                    print_val_test_auprc_w_path(pretrained_path, split=args.split, seed=seed)
            else:
                prev_trained_dir = pretrained_path if i == 1 and os.path.exists(pretrained_path) else f"{save_dir}{i-1}_model_ckpt/"
                size = args.teacher_size if i == 1 else args.student_size
                if args.no_pseudo:
                    pseudo_df = None
                elif args.on_the_fly_KD:
                    args.prev_trained_dir = prev_trained_dir
                    pseudo_df, pretrain_scores_dict = None, None
                else:
                    txdata, results, pretrain_scores_dict = generate_pseudo_labels(prev_trained_dir, size, seed, args)
                    pseudo_df = turn_into_df(results, txdata, args,) ## saves the pseudo csv file
                    del results
                
                print(f"Saving model is enabeled: {ith_model_ckpt if args.iter > i else None}")
                ith_model_ckpt = ith_model_ckpt if args.iter > i else None
                train_w_psuedo_labels(pseudo_df, pretrain_scores_dict, seed, ith_model_ckpt, args, i=i)
                del pseudo_df
            log_file.close()
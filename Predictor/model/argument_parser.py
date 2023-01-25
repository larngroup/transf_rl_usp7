# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 09:05:32 2022

@author: tiago
"""
import argparse
import os

def argparser():
    """
    Argument Parser Function
    Outputs:
    - FLAGS: arguments object
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--option',
        type=str,
        default='grid_search',
        help='Select: grid_search or train_model')
    
    parser.add_argument(
        '--model',
        type=str,
        default='standard',
        help='Select Transformer: mlm or standard')
    
    parser.add_argument(
        '--data_path',
        type=dict,
        default={},
        help='Data Path')

    parser.add_argument(
        '--bi_rnn',
        type=bool,
        nargs='+',
        help='Bi-directional RNN')
    
    parser.add_argument(
        '--finetuning',
        type=bool,
        default=False,
        help='Finetuning with USP7 data')
    
    parser.add_argument(
        '--max_str_len',
        type=int,
        default=100,
        help='SMILES Sequences Max Length')
    
    parser.add_argument(
        '--n_splits',
        type=int,
        default=5,
        help='Number of dataset splits')
    
    parser.add_argument(
        '--normalization_strategy',
        type=str,
        nargs='+',
        help='Normalization strategy (min_max, percentile or robust')
    
    parser.add_argument(
        '--optimizer_fn',
        type=str,
        nargs='+',
        action='append',
        help='Optimizer Function Parameters')
    
    parser.add_argument(
        '--reduction_lr',
        type=str,
        nargs='+',
        action='append',
        help='Reduce learning rate parameters (reduction factor, patience and minimum lr')
    
    parser.add_argument(
        '--rnn_1',
        type=int,
        nargs='+',
        help='Number of units first RNN layer')
    
    parser.add_argument(
        '--rnn_2',
        type=int,
        nargs='+',
        help='Number of units second RNN layer')
    
    parser.add_argument(
        '--n_layers',
        type=int,
        nargs='+',
        help='Number of FC layers')
    
    parser.add_argument(
        '--units',
        type=int,
        nargs = '+',
        help='Number of units first layer')
    
    parser.add_argument(
        '--activation_fc',
        type=str,
        nargs = '+',
        help='Activation function dense layers')

    parser.add_argument(
        '--dropout_rnn',
        type=float,
        nargs='+',
        help='Dropout RNN')
    
    parser.add_argument(
        '--rnn_type',
        type=str,
        nargs='+',
        help='Type of RNN layer, gru or lstm')
    
    parser.add_argument(
        '--dropout_fc',
        type=float,
        nargs='+',
        help='Dropout rate fully-connected neural network') 
    
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Epochs without progression')    
    
    parser.add_argument(
        '--n_iterations',
        type=int,
        default=75,
        help='Number of iterations')
    
    parser.add_argument(
        '--n_iterations_ft',
        type=int,
        default=50,
        help='Number of finetuning iterations')
    
    parser.add_argument(
        '--print_result',
        type=bool,
        default=False,
        help='Print the loss function evolution')
    
    parser.add_argument(
        '--model_size_std',
        type=int,
        default=256,
        help='Model size standard')  
   
    parser.add_argument(
        '--model_size_mlm',
        type=int,
        default=256,
        help='Model size mlm')
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size')    
    
    parser.add_argument(
        '--min_delta',
        type=float,
        default=0.001,
        help='Minimal loss improvement')
    
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='',
        help='Directory for checkpoint weights'
    )

    parser.add_argument(
        '--log_dir',
        type=str,
        default='',
        help='Directory for log data.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS


def logging(msg, FLAGS):
    """
    Logging function to update the log file
    Args:
    - msg [str]: info to add to the log file
    - FLAGS: arguments object
    """

    fpath = os.path.join(FLAGS.log_dir, "log.txt")

    with open(fpath, "a") as fw:
        fw.write("%s\n" % msg)
    print("------------------------//------------------------")
    print(msg)
    print("------------------------//------------------------")
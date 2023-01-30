# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 17:05:42 2021

@author: tiago
"""
# internal
from model.generation import generation_process
from model.argument_parser import *

# External
import tensorflow as tf
import warnings
import time
import os

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    
    """Runs the model according to the input selections"""
    
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
          
    FLAGS = argparser()
    FLAGS.log_dir = os.getcwd() + '/logs/' + time.strftime("%d_%m_%y_%H_%M", time.gmtime()) + "/"
    FLAGS.checkpoint_path = os.getcwd() + '/checkpoints/' + time.strftime("%d_%m_%y_%H_%M", time.gmtime()) + "/"
    FLAGS.path_generated_mols = os.getcwd() + '/sampled_mols/' + time.strftime("%d_%m_%y_%H_%M", time.gmtime()) + "/"
    
    FLAGS.models_path = {'generator_unbiased_path': 'models//generator//unbiased_generator.hdf5',
                       'generator_biased_path': 'models//generator//biased_generator.hdf5',
                       'transformer_standard_path': 'models//transformer//transformer_standard.h5',
                       'transformer_mlm_path': 'models//transformer//transformer_mlm.h5',
                       'predictor_standard': 'models//predictor//predictor_standard.h5',
                       'predictor_mlm': 'models//predictor//predictor_mlm.h5',
                       'generator_data_path': 'data/train_chembl_22_clean_1576904_sorted_std_final.smi',
                       'usp7_path_1': 'data/usp_inhibitors_1.xlsx',
                       'usp7_path_2': 'data/usp_inhibitors_2.xlsx',
                       'predictor_data_path': 'data/usp7_new.csv'} 

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    if not os.path.exists(FLAGS.checkpoint_path):
        os.makedirs(FLAGS.checkpoint_path)
    if not os.path.exists(FLAGS.path_generated_mols):
        os.makedirs(FLAGS.path_generated_mols)
        
    logging(str(FLAGS), FLAGS)
    
    # Implementation of the generation dynamics
    conditional_generation = generation_process(FLAGS)
   
    if FLAGS.option == 'unbiased':
        conditional_generation.samples_generation()
    
    elif FLAGS.option == 'standard' or FLAGS.option == 'mlm' or FLAGS.option == 'mlm_exp1' or FLAGS.option == 'mlm_exp2' or FLAGS.option == 'standard_exp1' or FLAGS.option == 'standard_exp2':
        conditional_generation.policy_gradient()
        conditional_generation.compare_models()

    else: 
        raise Exception("Parameter 'option' can only be: 'unbiased', 'standard', 'mlm', 'experiment1' or 'experiment2")

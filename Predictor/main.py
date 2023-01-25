# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 17:05:42 2021

@author: tiago
"""
# internal
from model.predictor import predictor_constructor
import tensorflow as tf
from model.argument_parser import argparser,logging

# external
import warnings
import time
import os

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    
    """Loads and combines the required models into the dynamics of generating
    novel molecules"""
    
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
          
    FLAGS = argparser()
    FLAGS.log_dir = os.getcwd() + '/logs/' + time.strftime("%d_%m_%y_%H_%M", time.gmtime()) + "/"
    FLAGS.checkpoint_path = os.getcwd() + '/checkpoints/' + time.strftime("%d_%m_%y_%H_%M", time.gmtime()) + "/"
    
    FLAGS.paths = {'transformer_standard_path': 'models//transformer//model.h5',
                       'transformer_mlm_path': 'models//transformer//model_mlm.h5',
                       'predictor_standard': 'models//predictor//predictor_standard.hdf5',
                       'predictor_mlm': 'models//predictor//predictor_mlm.hdf5',
                       'scaler_path': 'data//scaler.save',
                       'predictor_data_path': 'data/usp7_new.csv'}

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    if not os.path.exists(FLAGS.checkpoint_path):
        os.makedirs(FLAGS.checkpoint_path)
        
    logging(str(FLAGS), FLAGS)
    
    # Implementation of the Predictor dynamics
    predictor_model = predictor_constructor(FLAGS)
   
    if FLAGS.option == 'train_model':
        predictor_model.train_best()
    
    elif FLAGS.option == 'grid_search':
        predictor_model.grid_search_cv()

    else: 
        raise Exception("Parameter 'option' can only be: 'train_model' or 'grid_search'")

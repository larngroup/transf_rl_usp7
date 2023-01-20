# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 10:07:51 2021

@author: tiago
"""

# Internal 
from model.transformer import Transformer
from model.argument_parser import *
from dataloader.dataloader import DataLoader

# External
import warnings
import tensorflow as tf
# import tensorflow_addons as tfa
import time
import os
import itertools
import gc
warnings.filterwarnings('ignore')


def run_train_model(FLAGS):
    
    # Define the parameters
    n_epochs = FLAGS.n_epochs    
    batch_size = FLAGS.batchsize
    lr_scheduler = FLAGS.lr_scheduler    
    lr_WarmUpSteps = FLAGS.lr_WarmUpSteps    
    drop_rate = FLAGS.dropout[0]    
    optimizer_fn = FLAGS.optimizer_fn[0]   
    min_delta = FLAGS.min_delta    
    patience = FLAGS.patience    
    d_model = FLAGS.d_model[0]
    n_layers = FLAGS.n_layers[0]
    n_heads = FLAGS.n_heads[0]
    activation_func = FLAGS.activation_func[0]
    ff_dim = FLAGS.ff_dim[0]
    
 	# Initialize the Transformer model
    transformer_model = Transformer(FLAGS)
    
    print('\nLoading data...')
    raw_dataset = DataLoader().load_smiles(FLAGS)
    
    print('\nPre-processing data...')
    processed_dataset_train,processed_dataset_test = DataLoader.pre_process_data(raw_dataset,transformer_model,FLAGS)
    
    print('\nTraining the model...')
    transformer_model.train(processed_dataset_train,FLAGS, n_epochs, batch_size, lr_scheduler,
                                          lr_WarmUpSteps, min_delta, patience,
                                          optimizer_fn, drop_rate, d_model, n_layers,
                                          n_heads, activation_func, ff_dim)
    
    print('\nEvaluating the model...')
    loss,acc = transformer_model.evaluate(processed_dataset_test)
    
    logging("Test set - " + (" Loss = %0.3f, ACC = %0.3f" %(loss,acc)),FLAGS)

def run_grid_search(FLAGS):	
    """
    Run Grid Search function
    ----------
    FLAGS: arguments object
    """
    
    n_epochs = FLAGS.n_epochs    
    batch_size = FLAGS.batchsize
    lr_scheduler = FLAGS.lr_scheduler    
    lr_WarmUpSteps = FLAGS.lr_WarmUpSteps    
    drop_rate_set = FLAGS.dropout    
    optimizer_fn = FLAGS.optimizer_fn    
    min_delta = FLAGS.min_delta    
    patience = FLAGS.patience    
    d_model = FLAGS.d_model
    n_layers = FLAGS.n_layers
    n_heads = FLAGS.n_heads
    activation_func = FLAGS.activation_func
    ff_dim = FLAGS.ff_dim
    
    
 	# Initialize the Transformer model
    transformer_model = Transformer(FLAGS)
    
    raw_dataset = DataLoader().load_smiles(FLAGS)
    
    processed_dataset_train,processed_dataset_test = DataLoader.pre_process_data(raw_dataset,transformer_model,FLAGS)
    
    logging("--------------------Grid Search-------------------", FLAGS)
    
    for params in itertools.product(optimizer_fn, drop_rate_set, d_model,                                    
                                    n_layers, n_heads, activation_func, ff_dim):
        
        p1, p2, p3, p4, p5, p6, p7 = params
      
        results = []
        transformer_model = Transformer(FLAGS)
        # for fold_idx in range(len(folds)):            
        # index_train = list(itertools.chain.from_iterable([folds[i] for i in range(len(folds)) if i != fold_idx]))
        # index_val = folds[fold_idx]
        # data_train = [tf.gather(i, index_train) for i in data]
        # data_val = [tf.gather(i, index_val) for i in data]

        encoder = transformer_model.train(processed_dataset_train,FLAGS, n_epochs, batch_size, lr_scheduler,
                                          lr_WarmUpSteps, min_delta, patience,
                                          p1, p2, p3, p4, p5, p6, p7)
                                                                   
        loss,acc = transformer_model.evaluate(processed_dataset_test)             
        
        results.append((loss,acc))
        logging(("Epochs = %d, Batch size= %d, Lr scheduler = %s, Warmup steps = %d, "
                 "Minimum delta = %d, Patience = %d, Optimizer = %s,  Dropout= %d, " +                     
                 "Model dimension = %d, Number of Layers= %d, Number of heads= %d, " +                     
                 "Activation function= %s, Fully-connected dimension = %d, " +                     
                 "SCCE = %0.3f, ACC= %0.3f") %                    
                (n_epochs, batch_size, lr_scheduler,lr_WarmUpSteps, min_delta, patience,
                                                  p1, p2, p3, p4, p5, p6, p7, loss,acc), FLAGS)
        
        del encoder
        gc.collect()
        # logging("Mean - " + (" SCCE = %0.3f, ACC = %0.3f" % (np.mean(results, axis=0)[0], np.mean(results, axis=0)[1]), FLAGS)
        logging("Mean - " + (" SCCE = %0.3f, ACC = %0.3f" %
                             (loss,acc)), FLAGS)

def run():    
    """Loads the parameters, builds the model, trains and evaluates it"""
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0],True)

    FLAGS = argparser()
    FLAGS.log_dir = os.getcwd() + '/logs/' + time.strftime("d_%m_%y_%H_%M", time.gmtime())+"/"
    FLAGS.checkpoint_path = os.getcwd() + '/checkpoints/' + time.strftime("d_%m_%y_%H_%M", time.gmtime())+"/"

    if not os.path.exists(FLAGS.log_dir):
    	os.makedirs(FLAGS.log_dir)
    if not os.path.exists(FLAGS.checkpoint_path):
    	os.makedirs(FLAGS.checkpoint_path)
   
    logging(str(FLAGS), FLAGS)

    if FLAGS.option == 'train':
    	run_train_model(FLAGS)
    if FLAGS.option == 'validation':
    	run_grid_search(FLAGS)

    
if __name__ == '__main__':
    run()

 
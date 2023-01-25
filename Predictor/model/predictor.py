# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:49:39 2021

@author: tiago
"""

# internal
from .base_model import BaseModel
from dataloader.dataloader import DataLoader
from utils.utils import Utils
from model.build_models import Build_models
from model.prediction import Predictor
from model.transformer import Transformer
from model.transformer_mlm import Transformer_mlm
from model.build_best_models import rnn_model,fc_model
from model.argument_parser import logging

# external
import time
import joblib
import itertools
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler,MinMaxScaler

class predictor_constructor(BaseModel):
    """Predictor Model Class initialization"""
    
    def __init__(self, FLAGS):
        super().__init__(FLAGS)

        self.FLAGS = FLAGS
        
        # Transformer model initialization
        if self.FLAGS.model == 'standard':
            self.transformer_model = Transformer(self.FLAGS)
        elif self.FLAGS.model == 'mlm':
            self.transformer_model = Transformer_mlm(self.FLAGS)
            
        # Load the table of possible tokens
        self.token_table = Utils().table 
        
        # Load data: SMILES + raw pIC50 values 
        self.raw_smiles,self.raw_labels = DataLoader().load_dataset(self.FLAGS)
        
        # Split data into training, validation and testing sets
        self.pre_process_data()
        
        # Optimizer definition
        self.FLAGS.optimizer_fn = self.FLAGS.optimizer_fn[0]
        self.FLAGS.reduction_lr = self.FLAGS.reduction_lr[0]

    def pre_process_data(self):
        """ 
        Splits the data into train-validation and test sets and defines the 
        indexes to split for the different folds of cross-validation
        """
        # Split data into train_validation and test sets
        self.train_val_data,self.test_data,self.train_val_labels,self.test_labels = Utils().data_division(self.raw_smiles,self.raw_labels)

        # Set up the cross-validation
        if self.FLAGS.option == 'grid_search':
            self.idxs_cv = Utils().cv_split(self.train_val_data,self.train_val_labels,self.FLAGS)
            
    def metrics(self,y_true,predictions):
        """ Computes the Predictor's evaluation metrics
    
           Args
           ----------
               y_true (tensor): True values
               predictions (tensor): Model predictions
    
           Returns
           -------
               Root-mean squared (RMSE), coefficient of determination (Q2) and 
               concordance correlation coefficient (CCC)
        """   
        rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, predictions))))
        
        SS_res =  tf.reduce_sum(tf.square(tf.subtract(y_true, predictions)))
        SS_tot = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
        q2 = tf.subtract(1.0, tf.divide(SS_res, SS_tot))
        
        num = 2*tf.reduce_sum(tf.subtract(y_true,tf.reduce_mean(y_true))*tf.subtract(predictions,tf.reduce_mean(predictions)))
        den = tf.reduce_sum(tf.square(tf.subtract(y_true,tf.reduce_mean(y_true)))) + tf.reduce_sum(tf.square(tf.subtract(predictions,tf.reduce_mean(predictions)))) + predictions.shape[-1]*tf.square(tf.subtract(tf.reduce_mean(y_true),tf.reduce_mean(predictions)))
        ccc = num/den
        return rmse,q2,ccc
        
    def grid_search_cv(self):
        """ 
        Implements the grid-search method to define the best architecture for 
        both the standard and mlm-based Predictors
        """
        
        logging("--------------------Grid Search-------------------", self.FLAGS)
        
        if self.FLAGS.model == 'standard':
            itrs = itertools.product(self.FLAGS.rnn_1,self.FLAGS.rnn_2,
                                            self.FLAGS.dropout_rnn,self.FLAGS.bi_rnn,
                                            self.FLAGS.rnn_type,self.FLAGS.normalization_strategy)
        elif self.FLAGS.model == 'mlm':
            itrs = itertools.product(self.FLAGS.n_layers,self.FLAGS.units,
                                            self.FLAGS.dropout_fc,self.FLAGS.activation_fc,
                                            self.FLAGS.normalization_strategy)
            
        for params in itrs:
            
            if self.FLAGS.model == 'standard': 
                p1,p2,p3,p4,p5,p6 = params               
                self.FLAGS.rnn_1 = p1
                self.FLAGS.rnn_2 = p2
                self.FLAGS.dropout_rnn= p3
                self.FLAGS.bi_rnn = p4
                self.FLAGS.rnn_type = p5
                self.FLAGS.normalization_strategy = p6

        
            elif self.FLAGS.model == 'mlm':
                p1,p2,p3,p4,p5 = params
                self.FLAGS.n_layers = p1
                self.FLAGS.units = p2
                self.FLAGS.dropout_fc= p3
                self.FLAGS.activation_fc = p4
                self.FLAGS.normalization_strategy = p5
        
            # Normalization methods definition
            scaler = None
            if self.FLAGS.normalization_strategy == 'min_max':
                scaler = MinMaxScaler().fit(np.array(self.train_val_labels).reshape(-1, 1))
                
            elif self.FLAGS.normalization_strategy == 'robust':
                scaler = RobustScaler().fit(np.array(self.train_val_labels).reshape(-1, 1))
                
            i = 1
     
            for split in self.idxs_cv:
                print('\nCross validation, fold number ' + str(i) + ' in progress...')
                data_i = []
                train_idxs, val_idxs = split
                
                X_train = [m for idx,m in enumerate(self.train_val_data) if idx in train_idxs]
                y_train = self.train_val_labels[train_idxs]
                X_val = [m for idx,m in enumerate(self.train_val_data) if idx in val_idxs]
                y_val = self.train_val_labels[val_idxs]
                X_test = self.test_data
                y_test = self.test_labels 
    
                # # Apply the Multi-Head Attention to extract contextual embeddings
                X_train_embeds = self.transformer_model.predict_step(X_train)
                X_val_embeds = self.transformer_model.predict_step(X_val)
                X_test_embeds = self.transformer_model.predict_step(X_test)

                data_i.append(X_train_embeds)
                data_i.append(y_train)
                data_i.append(X_val_embeds)
                data_i.append(y_val)
                data_i.append(X_test_embeds)
                data_i.append(y_test)
      
                data_i = Utils().normalize(self.FLAGS,data_i,self.train_val_labels,scaler)
                
                self.build_models_obj = Build_models(data_i,self.FLAGS,i)
                
                
                if self.FLAGS.model == 'standard': 
                    self.build_models_obj.build_tri_rnn_am()
                elif self.FLAGS.model == 'mlm': 
                    self.build_models_obj.build_tri_rnn_am_mlm()
                    
    
                self.build_models_obj.train_dl_model()
                self.build_models_obj.save_dl_model()
                i+=1
            
            predictor= Predictor(self.build_models_obj,self.token_table,self.FLAGS)
            metrics_test,metrics_val = predictor.evaluator_cv(data_i,scaler) 
            
            print('\n\nTest set, Validation set')
            print("\nMean_squared_error: ",metrics_test[0],', ',metrics_val[0], "\nRoot mean squared: ",metrics_test[1],', ',metrics_val[1],"\nQ_squared: ", metrics_test[2],', ',metrics_val[2], "\nCCC: ",metrics_test[3],', ',metrics_val[3])
            
            if self.FLAGS.model == 'standard': 
                logging(("Model = %s, Iterations = %d,  Batch size = %d, RNN_1 units= %d, " +
                         "RNN_2 units = %d,  Dropout rate = %0.2f, Bi-directional = %s, " +
                         "RNN type = %s, Normalization type = %s, MSE_val = %0.3f, MSE_test = %0.3f, "+
                         "RMSE_val = %0.3f, RMSE_test = %0.3f, Q2_val = %0.3f, "+
                         "Q2_test = %0.3f, CCC_val = %0.3f, CCC_test = %0.3f") %
                        (self.FLAGS.model, self.FLAGS.n_iterations, self.FLAGS.batch_size,
                         p1, p2, p3, p4, p5, p6, metrics_test[0],metrics_val[0],
                         metrics_test[1],metrics_val[1],metrics_test[2],metrics_val[2],
                        metrics_test[3],metrics_val[3]),self.FLAGS)
                
            elif self.FLAGS.model == 'mlm': 
                logging(("Model = %s, Iterations = %d,  Batch size = %d, Number of layers = %d, " +
                         "Units per layer = %d,  Dropout rate = %f, Activation function = %s, Normalization type = %s " +
                         "MSE_val = %0.3f, MSE_test = %0.3f, RMSE_val = %0.3f, "+
                         "RMSE_test = %0.3f, Q2_val = %0.3f, Q2_test = %0.3f, CCC_val = %0.3f, CCC_test = %0.3f") %
                        (self.FLAGS.model, self.FLAGS.n_iterations, self.FLAGS.batch_size,
                         p1, p2, p3, p4, p5, metrics_test[0],metrics_val[0],
                         metrics_test[1],metrics_val[1],metrics_test[2],metrics_val[2],
                        metrics_test[3],metrics_val[3]),self.FLAGS)
     
            

    def loss_func(self,y_true, y_pred):
        """ 
        Computes the mean-squared error loss
        Args
        ----------
            y_true (tensor): True values
            y_pred (tensor): Model predictions
 
        Returns
        -------
            Mean-squared error between the predictions and the real pIC50
        """
        loss = self.mse_loss(y_true, y_pred)
        return loss
           
    
    @tf.function
    def train_step(self,source_seq, target_y):
        """ 
        Performs the learning step of the training process
        Args
        ----------
            source_seq (tensor): Input SMILES
            target_y (tensor): True pIC50
 
        Returns
        -------
            loss, rmse, r2, and ccc
        """
        
        with tf.GradientTape() as tape:
            
            pred = self.predictor(source_seq)
 
            loss = self.loss_func(target_y, pred)
            rmse,r2,ccc = self.metrics(target_y, pred)
    
        variables = self.predictor.trainable_variables 
        gradients = tape.gradient(loss, variables)
  
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss,rmse,r2,ccc
    
    # @tf.function
    def train_best(self):
        """ 
        Implements the best Predictor using all available data  
        """

        x_data = self.raw_smiles
        y_data_raw = self.raw_labels
        self.FLAGS.normalization_strategy = self.FLAGS.normalization_strategy[0]
        
        # Convert smiles to embeddings
        X_embeds = self.transformer_model.predict_step(x_data)

        # Normalization methods 
        scaler = None
        if self.FLAGS.normalization_strategy == 'min_max':
            scaler = MinMaxScaler().fit(np.array(self.raw_labels).reshape(-1, 1))
            
        elif self.FLAGS.normalization_strategy == 'robust':
            scaler = RobustScaler().fit(np.array(self.raw_labels).reshape(-1, 1))
        
        joblib.dump(scaler,self.FLAGS.checkpoint_path+'scaler_predictor.save')

        y_data_normalized = Utils().normalize(self.FLAGS,y_data_raw,None,scaler,True)
        y_data_reshaped = np.reshape(y_data_normalized,(len(y_data_normalized),1))
        
        self.tensor_dataset = tf.data.Dataset.from_tensor_slices(
            (X_embeds, y_data_reshaped))
        self.tensor_dataset = self.tensor_dataset.shuffle(len(x_data)).batch(self.FLAGS.batch_size)
        
        if self.FLAGS.model == 'standard': 
            self.predictor = rnn_model(self.FLAGS)
            sequence_in = tf.constant([list(np.ones((1,256)))])
        elif self.FLAGS.model == 'mlm': 
            sequence_in = tf.constant([list(np.ones((1,256)))])
            self.predictor = fc_model(self.FLAGS)
        
        
        prediction_test = self.predictor(sequence_in)
        
        self.mse_loss = tf.keras.losses.MeanSquaredError()
                
        
        if self.FLAGS.optimizer_fn[0] == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=float(self.FLAGS.optimizer_fn[1]),
                                                beta_1=float(self.FLAGS.optimizer_fn[2]),
                                                beta_2=float(self.FLAGS.optimizer_fn[3]),
                                                epsilon=float(self.FLAGS.optimizer_fn[4]))   
        
        last_loss = {'epoch':0,'value':1000}
        
        for epoch in range(self.FLAGS.n_iterations):
            print(f'Epoch {epoch+1}/{self.FLAGS.n_iterations}')
            start = time.time()
            loss_epoch = []
            rmse_epoch = []
            r2_epoch = []
            ccc_epoch = []
            
            for batch, (seq_x, y_train) in enumerate(self.tensor_dataset.take(-1)):
                loss_batch,rmse_batch,r2_batch,ccc_batch = self.train_step(seq_x, y_train)
                loss_epoch.append(loss_batch)
                rmse_epoch.append(rmse_batch)
                r2_epoch.append(r2_batch)
                ccc_epoch.append(ccc_batch)
                
                if batch == len(list(self.tensor_dataset.take(-1)))-1:
                    print(f'{batch+1}/{len(self.tensor_dataset.take(-1))} - {round(time.time() - start)}s - loss: {np.mean(loss_epoch):.4f} - rmse: {np.mean(rmse_epoch):.4f} - r2: {np.mean(r2_epoch):.4f} - ccc: {np.mean(ccc_epoch):.4f}')
                
            if (last_loss['value'] - np.mean(loss_epoch)) >= self.FLAGS.min_delta:
                last_loss['value'] = np.mean(loss_epoch)
                last_loss['epoch'] = epoch+1 
                print('Saving model...')
                self.predictor.save_weights(self.FLAGS.checkpoint_path + 'predictor_' + self.FLAGS.option+'.h5') #save_path+'trained_model.h5py')
            if ((epoch+1) - last_loss['epoch']) >= self.FLAGS.patience:
                break 
            

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:49:39 2021

@author: tiago
"""

# internal
from .base_model import BaseModel
from utils.utils import Utils
from model.encoder import Encoder
from model.warm_up_decay_schedule import WarmupThenDecaySchedule

# external
import tensorflow as tf
# import tensorflow_addons as tfa
import numpy as np
import time
import random


class Transformer(BaseModel):
    """Transformer general Class"""
    def __init__(self, FLAGS):
        super().__init__(FLAGS)
        
        # Implementation parameters
        self.FLAGS = FLAGS
        
        # Load the table of possible tokens
        self.token_table = Utils().voc_table 
        self.vocab_size = len(self.token_table)
        
        # Dictionary that makes the correspondence between each token and unique integers
        self.tokenDict = Utils.smilesDict(self.token_table)
        self.inv_tokenDict = {v: k for k, v in self.tokenDict.items()}
        
        # Positional encoding
        self.max_length = self.FLAGS.max_strlen
        self.model_size_best = 256
        self.pes = []
        for i in range(self.max_length):
            self.pes.append(Utils.positional_encoding(i, self.model_size_best))
        self.pes = np.concatenate(self.pes, axis=0)
        self.pes = tf.constant(self.pes, dtype=tf.float32)

    
    def loss_function(self,y_true,y_pred, mask):
        """ Calculates the loss (sparse categorical crossentropy) only for 
            the masked tokens

        Args
        ----------
            y_true (array): true label
            y_pred (array): model's predictions
            mask (array): mask of 1's (masked tokens) and 0's (rest)
            
        Returns
        -------
           loss
        """
        loss_ = self.crossentropy(y_true,y_pred)
        mask = tf.cast(mask,dtype=loss_.dtype)
        loss_ *= mask
    
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

        
    def accuracy_function(self, y_true,y_pred, mask):
        """ Calculates the accuracy of masked tokens that were well predicted 
            by the model

        Args
        ----------
            y_true (array): true label
            y_pred (array): model's predictions
            mask (array): mask of 1's (masked tokens) and 0's (rest)
            
        Returns
        -------
           accuracy
        """
        accuracies = tf.equal(y_true, tf.cast(tf.argmax(y_pred, axis=2), dtype=tf.float64))
        mask_b = mask > 0
        accuracies = tf.math.logical_and(mask_b, accuracies)
        accuracies = tf.cast(accuracies, dtype=tf.float32) 
        mask = tf.cast(mask, dtype=tf.float32) 
        return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)
    
    def masking(self, inp, fgs, threshold_min):
        """ Performs the masking of the input smiles. 20% of the tokens are 
            masked (10% from FG and 10% from other regions of the molecule)

        Args
        ----------
            inp (list): List of input mols
            fgs (list): List with indices of FG
            threshold_min (int): minimum number of FGs where masking considers 
                                 FGs'
            
        Returns
        -------
            x (list): SMILES with masked tokens
            masked_positions (list): Sequences of the same length of the smiles
                                     with 1's on the masked positions
        """
        masked_positions = []
        x = [i.copy() for i in inp]
        
        for smile_index in range(len(x)):
            fg = fgs[smile_index]
            not_fg = [indx for indx in range(len(x[smile_index])) if (indx not in fg) and (x[smile_index][indx] not in [self.token_table.index('<CLS>'),
			self.token_table.index('<PAD>'), self.token_table.index('<SEP>'), self.token_table.index('<MASK>') ])] 
            
            # from the 20% of tokens that will be masked, half will be from the fg and the other half from the rest of the schleton
            p_fg = 0.1
            p_not_fg = 0.1 
            
            if len(fg) < threshold_min: 
                p_not_fg = 0.15
                
            num_mask_fg = max(1, int(round(len(fg) * p_fg))) 
            num_mask_not_fg = max(1, int(round(len(not_fg) * p_not_fg))) 
            shuffle_fg = random.sample(range(len(fg)), len(fg)) 
            shuffle_not_fg = random.sample(range(len(not_fg)), len(not_fg))
			
            fg_temp = [fg[n] for n in shuffle_fg[:num_mask_fg]] 
            not_fg_temp = [not_fg[n] for n in shuffle_not_fg[:num_mask_not_fg]] 
			
            mask_index = fg_temp + not_fg_temp#fg[shuffle_fg[:num_mask_fg]]+ shuffle_not_fg[:num_mask_not_fg] 
            # print('sequence: ',len(x[smile_index]))
            # print('masked tokens: ',len(mask_index))
            masked_pos =[0]*len(x[smile_index])
			
            for pos in mask_index:
                masked_pos[pos] = 1
                rd = random.random()
                if rd <= 0.8: 
                    # print('first')
                    x[smile_index][pos] = self.token_table.index('<MASK>')
                elif rd > 0.8: 
                    # print('second')
                    index = random.randint(1, self.token_table.index('<CLS>')-1) 
                    x[smile_index][pos] = index
            masked_positions.append(masked_pos) 
            
            
            # for pos in mask_index:
            #     masked_pos[pos] = 1
            #     if random.random() < 0.8:
            #         x[smile_index][pos] = self.token_table.index('<MASK>')
            #     elif random.random() < 0.15:
            #         index = random.randint(1, self.token_table.index('<CLS>')-1)
            #         x[smile_index][pos] = index
            # masked_positions.append(masked_pos)
                    
        return x, masked_positions
   
    @tf.function
    def train_step(self, x, target, masked_positions): 
 
        with tf.GradientTape() as tape: 
            predictions, align= self.encoder(x)
            loss = self.loss_function(target, predictions, masked_positions) 
        
        gradients = tape.gradient(loss, self.encoder.trainable_variables) 
        self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables)) 
        acc = self.accuracy_function(target, predictions, masked_positions) 
        return loss, acc

    
    def train(self,data,FLAGS,n_epochs, batchsize, lr_scheduler, lr_WarmUpSteps,                                   
              min_delta, patience, optimizer_fn, dropout, d_model, n_layers, 
              n_heads, activation_func, ff_dim):
        
        """Builds and trains the model"""
     
        self.encoder = Encoder(self.vocab_size, d_model, n_layers, n_heads,
                               dropout, activation_func,ff_dim,self.pes)
        
        sequence_in = tf.constant([[1, 2, 3, 0, 0]])
        encoder_output, _ = self.encoder(sequence_in)
        print(encoder_output.shape)
            
        self.crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction='none')
        
        lr = WarmupThenDecaySchedule(d_model,lr_WarmUpSteps)
        

        if optimizer_fn[0] == 'radam':            
            self.optimizer = tfa.optimizers.RectifiedAdam(learning_rate=float(lr), beta_1=float(optimizer_fn[1]),                                               
                                               beta_2=float(optimizer_fn[2]), epsilon=float(optimizer_fn[3]))        
        elif optimizer_fn[0] == 'adam':  

            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=float(optimizer_fn[2]),                                               
                                                   beta_2=float(optimizer_fn[3]), epsilon=float(optimizer_fn[4]))
        elif optimizer_fn[0] == 'adamw':            
            self.optimizer = tfa.optimizers.AdamW(learning_rate=float(optimizer_fn[1]), beta_1=float(optimizer_fn[1]),                                               
                                               beta_2=float(optimizer_fn[2]), epsilon=float(optimizer_fn[3]))
     
        
        last_loss = {'epoch':0,'value':1000} 
        for epoch in range(n_epochs): 
            print(f'Epoch {epoch+1}/{n_epochs}') 
            start = time.time() 
            loss_epoch = [] 
            acc_epoch = [] 
            for num, ((x_train, fgs_train), y_train) in enumerate(data): 
                loss_batch, acc_batch = self.train_step(x_train, y_train, fgs_train) 
                loss_epoch.append(loss_batch) 
                acc_epoch.append(acc_batch) 
                if num == len(data)-1: 
                    # print(f'{round(time.time() - start)}s ')    
                    print(f'{num+1}/{len(data)} - {round(time.time() - start)}s - loss: {np.mean(loss_epoch):.4f} - accuracy: {np.mean(acc_epoch):.4f}')    
					
            if (last_loss['value'] - tf.math.reduce_mean(loss_epoch)) >= self.FLAGS.min_delta: 
                last_loss['value'] = tf.math.reduce_mean(loss_epoch) 
                last_loss['epoch'] = epoch+1
                print('Saving model...') 
                self.encoder.save_weights(self.FLAGS.checkpoint_path + 'model.h5')  
                
            if ((epoch+1) - last_loss['epoch']) >= self.FLAGS.patience: 
                break 
            
        
        
    def evaluate(self,processed_dataset_test):
        """Predicts resuts for the test dataset"""
        print('\nPredicting on test set...') 
        
        
        test_loss = []
        test_acc = []
        for num, (smiles,smiles_masked,fgs_test) in enumerate(processed_dataset_test):
            pred, _= self.encoder(tf.constant(np.array(smiles_masked)),training=False) 
            loss_batch = self.loss_function(smiles,pred, fgs_test) 
            acc_batch = self.accuracy_function(smiles,pred, np.array(fgs_test)) 
            
            test_loss.append(loss_batch.numpy())
            test_acc.append(acc_batch.numpy())
            
        print("Loss, Accuracy: ", np.mean(test_loss), np.mean(test_acc))
        
        return np.mean(test_loss), np.mean(test_acc)

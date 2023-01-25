# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:49:39 2021

@author: tiago
"""

# internal
from .base_model import BaseModel
from utils.utils import Utils
from model.encoder_mlm import Encoder
from model.warm_up_decay_schedule import WarmupThenDecaySchedule

# external
import tensorflow as tf
import numpy as np
import time
import random

class Transformer_mlm(BaseModel):
    """Transformer general Class"""
    def __init__(self, FLAGS):
        super().__init__(FLAGS)

        # Implementation parameters
        self.FLAGS = FLAGS
        
        # Compute the dictionary that makes the correspondence between 
        # each token and unique integers
        self.token_table = Utils().voc_table_mlm # Load the table of possible tokens
        self.vocab_size = len(self.token_table)
        self.tokenDict = Utils().smilesDict(self.token_table)
        self.inv_tokenDict = {v: k for k, v in self.tokenDict.items()}
        
        # Model architecture
        self.n_heads = 4
        self.n_layers = 4
        self.model_size = 256
        
        # Build model and load pre-trained weights
        self.pre_processing()
        self.build_models()
        
    def pre_processing(self):
        """ Computes the positional encoding
        """     
                
        # Positional encoding
        self.max_length = self.FLAGS.max_str_len
        self.pes = []
        for i in range(self.max_length):
            self.pes.append(Utils.positional_encoding(i, self.model_size ))
        self.pes = np.concatenate(self.pes, axis=0)
        self.pes = tf.constant(self.pes, dtype=tf.float32)


    def build_models(self):
        """ Builds the Transformer architecture"""        

        self.encoder = Encoder(self.vocab_size, self.model_size, self.n_layers, self.n_heads,self.pes)
        sequence_in = tf.constant([[1, 2, 3, 0, 0]])
        encoder_output, _ = self.encoder(sequence_in)
        
        self.encoder.load_weights(self.FLAGS.paths['transformer_mlm_path'])  
        
    
    def loss_function(self,y_true,y_pred, mask):
        """ Calculates the loss (sparse categorical crossentropy) for 
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
        mask_b = tf.constant(mask > 0)
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
            
            # from the 2% of tokens that will be masked, half will be from the fg and the other half from the rest of the schleton
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
			
            mask_index = fg_temp + not_fg_temp
            masked_pos =[0]*len(x[smile_index])
			
            for pos in mask_index:
                masked_pos[pos] = 1
                if random.random() < 0.8: 
                    x[smile_index][pos] = self.token_table.index('<MASK>')
                elif random.random() < 0.15: 
                    index = random.randint(1, self.token_table.index('<CLS>')-1) 
                    x[smile_index][pos] = index
            masked_positions.append(masked_pos) 
                    
        return x, masked_positions
   
        
    def predict_step(self,sequences_in):
        """ 
        Computes the Predictor input descriptors from the input sequences

        Args
        ----------
            sequences_in (list): Predictor input SMILES
 
        Returns
        -------
            encoder_cls (array): CLS token of each input molecule
        """
        
        print('padding...')
        smiles_padded,smiles_filtered = Utils.tokenize_and_pad(self.FLAGS,sequences_in,self.token_table)
        print('padding done')
        # Identify the functional groups of each molecule
        fgs = Utils.identify_fg(smiles_filtered)
                
        # Transforms each token to the respective integer, according to 
        # the previously computed dictionary
        seq_idxs_input = Utils().smiles2idx(self.FLAGS,smiles_padded,self.tokenDict)
        
        # 'Minimumm number of functional groups (FGs) where masking considers FGs'
        threshold_min = 6
        
        # masking mlm
        input_masked, masked_positions = self.masking(seq_idxs_input, fgs, threshold_min)  
        
        print(input_masked[0].shape)
        
        encoder_last_layer,encoder_all = self.encoder(tf.constant(input_masked), training=False)
        
        # Extract just the [CLS] token vector (batch,256)
        encoder_cls = encoder_all.numpy()[:,0,:]
        print(encoder_cls.shape)
        return encoder_cls


        
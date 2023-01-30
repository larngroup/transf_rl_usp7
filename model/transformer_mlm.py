# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:49:39 2021

@author: tiago
"""

# internal
from .base_model import BaseModel
from utils.utils import Utils
from model.encoder_mlm import Encoder

# external
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from random import randrange

class Transformer_mlm(BaseModel):
    """Transformer general Class"""
    def __init__(self, FLAGS):
        super().__init__(FLAGS)

        self.FLAGS = FLAGS
        self.token_table_general = Utils().tokens # Load the table of possible tokens
        self.token_table = Utils().voc_table_mlm # Load the table of possible tokens
        self.vocab_size = len(self.token_table)
        # Compute the dictionary that makes the correspondence between each token and unique integers
        self.tokenDict = Utils.smilesDict(self.token_table)
        self.inv_tokenDict = {v: k for k, v in self.tokenDict.items()}
        
        
        # Best parameters definition
        self.max_length = self.FLAGS.max_str_len
        self.model_size = 256
        self.n_heads = 4
        self.n_layers = 4
        self.dropout = 0.1
        self.activation_func = 'relu'
        self.ff_dim = 1024
        
        self.pre_process_data()
        self.build_models()
        

    def pre_process_data(self):
        """ Pre-processes the dataset of molecules including padding, 
            tokenization and transformation of tokens into integers.
    
        Returns
        -------
            pre_processed_dataset (list): List with pre-processed training and 
                                          testing sets
        """     
    
        # Positional encoding
        self.pes = []
        for i in range(self.max_length):
            self.pes.append(Utils.positional_encoding(i, self.model_size ))
        self.pes = np.concatenate(self.pes, axis=0)
        self.pes = tf.constant(self.pes, dtype=tf.float32)

    
    def build_models(self):
        """ Builds the Transformer architecture"""
        self.encoder = Encoder(self.vocab_size, self.model_size, self.n_layers, 
                               self.n_heads, self.dropout, self.activation_func, 
                               self.ff_dim, self.pes)
        
        sequence_in = tf.constant([[1, 2, 3, 0, 0]])
        encoder_output, _,_ = self.encoder(sequence_in)
        
        print(encoder_output.shape)
        
        self.encoder.load_weights(self.FLAGS.models_path['transformer_mlm_path'])  
        
    
    def loss_function(self,y_true,y_pred, mask):
        """ Calculates the loss (sparse categorical crossentropy) just for 
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
			
            mask_index = fg_temp + not_fg_temp#fg[shuffle_fg[:num_mask_fg]]+ shuffle_not_fg[:num_mask_not_fg] 
            # print('sequence: ',len(x[smile_index]))
            # print('masked tokens: ',len(mask_index))
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
        
        
    def predict_step(self,sequences):
        """
        Parameters
        ----------
        sequence_in (str): Input SMILES sequence
        trajectory_ints (list) : Set of tokens that represent the SMILES in the 
                                Generator language
    
        Returns
        -------
        scores: attention scores
        idxs_final: ids of the most important tokens
        contextual_embedding : embeddings of the input sequences
        tokens : tokens computed in Transformer pre-processing step
    
        """

        tokens,smiles_filtered = Utils.tokenize_and_pad(self.FLAGS,sequences,self.token_table,'mlm')   
 
        # Identify the functional groups of each molecule
        fgs = Utils.identify_fg(smiles_filtered)
                 
        # Transforms each token to the respective integer, according to 
        # the previously computed dictionary
        input_raw = Utils.smiles2idx(tokens,self.tokenDict)
                     
        # 'Minimumm number of functional groups (FGs) where masking considers FGs'
        threshold_min = 6
         
        # masking mlm
        self.input_masked, self.masked_positions = self.masking(input_raw, fgs, threshold_min)   
                 
        # Transformer encoder: maps the input tokens to the contextual embeddings and attention scores
        tokens_probs, contextual_embedding,alignments = self.encoder(tf.constant( self.input_masked),training=False)

       # Extract just the [CLS] token vector (batch,256)
        encoder_cls = contextual_embedding.numpy()[:,0,:]

        return encoder_cls 

    def predict_step_rl(self,sequence_in,trajectory_ints):
        """
        Parameters
        ----------
        sequence_in (str): Input SMILES sequence
        trajectory_ints (list) : Set of tokens that represent the SMILES in the 
                                Generator language

        Returns
        -------
        scores: attention scores
        idxs_final: ids of the most important tokens
        contextual_embedding : embeddings of the input sequences
        tokens : tokens computed in Transformer pre-processing step

        """
        # print(sequence_in, trajectory_ints)
        
        # Tokenize - transform the SMILES strings into lists of tokens 
        tokens,smiles_filtered = Utils.tokenize_and_pad(self.FLAGS,[sequence_in],self.token_table,'mlm')   

        # Identify the functional groups of each molecule
        fgs = Utils.identify_fg(smiles_filtered)
                 
        # Transforms each token to the respective integer, according to 
        # the previously computed dictionary
        input_raw = Utils.smiles2idx(tokens,self.tokenDict)
                     
        # 'Minimumm number of functional groups (FGs) where masking considers FGs'
        threshold_min = 6
         
        # masking mlm
        self.input_masked, self.masked_positions = self.masking(input_raw, fgs, threshold_min)   
        
        # Transformer encoder: maps the input tokens to the contextual embeddings and attention scores
        tokens_probs, contextual_embedding,alignments = self.encoder(tf.constant( self.input_masked),training=False)

       # Extract just the [CLS] token vector (batch,256)
        encoder_cls = contextual_embedding.numpy()[:,0,:]

        # Extract only the last layer 
        attention_scores = alignments[-1].numpy()
       
        head = randrange(self.n_heads)
        selected_head = attention_scores[0,head,:,:]        
       
        # Extract the importance of each specific token
        importance_all = []

        size_h = len(selected_head)
        for c in range(0,size_h):

            importance_element = []
            importance_element.append(selected_head[c,c])
            
            for v in range(0,size_h):
                if v!=c:
                    element = (selected_head[c,v] + selected_head[v,c])/2
                    importance_element.append(element)
        
            importance_all.append(importance_element)
        

        importance_tokens = [np.mean(l) for l in importance_all]
     
        scores = Utils.softmax(importance_tokens)
        
        # Sort keeping indexes
        sorted_idxs = np.argsort(-scores)
        # print('idxs ordered: ', sorted_idxs )
        
        # Identify most important tokens
        number_tokens = int((self.FLAGS.top_tokens_rate)*len(sequence_in))
        filtered_idxs = sorted_idxs[0:number_tokens-1] 
       
        idxs_final = []
        search_gap = 0 
        
        for idx_t,t in enumerate(tokens[0]):

            if idx_t in filtered_idxs:

                if len(t)<3:
                    idxs_final.append(idx_t+search_gap)
                elif len(t)>=3  and t not in ['<PAD>','<CLS>']:
                    try:
                        index_start = trajectory_ints.index(17, idx_t+search_gap, idx_t+search_gap+3)
                        index_end = trajectory_ints.index(18, index_start+search_gap, index_start+search_gap+4)
                    except:
                        print(idx_t,t)    
                    for ii in range(index_start,index_end+1):
                        idxs_final.append(ii)
                                       
                    
            if len(t)>2 and t not in ['<PAD>','<CLS>']:
                t_len = len(t)
                search_gap += t_len-1

        
        if self.FLAGS.plot_attention_scores:
            
            # Define the attention score threshold above which the important tokens are considered
            threshold = scores[sorted_idxs[number_tokens]]
        
            # Plot the important tokens        
            plt_1 = plt.figure(figsize=(15,7))
            plt.axhline(y = threshold, color = 'r', linestyle = '-')
            plt.plot(scores,linestyle='dashed')
            ax = plt.gca()
            ax.set_xticks(range(len(sequence_in)))
            ax.set_xticklabels(sequence_in)
            plt.xlabel('Sequence')
            plt.ylabel('Attention weights')
            plt.show()
            

        return scores,idxs_final,encoder_cls,tokens,alignments


        
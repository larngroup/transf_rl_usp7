# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:49:39 2021

@author: tiago
"""

# internal
from .base_model import BaseModel
from utils.utils import Utils
from model.encoder import Encoder

# external
import tensorflow as tf
import numpy as np
import joblib

class Transformer(BaseModel):
    """Transformer general Class"""
    def __init__(self, FLAGS):
        super().__init__(FLAGS)

        self.FLAGS = FLAGS
        
        # Compute the dictionary that makes the correspondence between 
        # each token and unique integers
        self.token_table = Utils().voc_table # Load the table of possible tokens
        self.tokenDict = Utils().smilesDict(self.token_table)
        self.inv_tokenDict = {v: k for k, v in self.tokenDict.items()}
        self.vocab_size = len(self.token_table)
        
        # Model architecture
        self.model_size = 256    
        self.n_heads = 4
        self.n_layers = 4
        self.dropout = 0.1
        self.activation_func = 'relu'
        self.ff_dim = 1024

        # Build model and load pre-trained weights
        self.pre_processing()
        self.build_models()
        
        # Load scaler
        self.scaler = joblib.load(FLAGS.paths['scaler_path']) 

            
    def pre_processing(self):
        """ Computes the positional encoding
        """     
        self.max_length = self.FLAGS.max_str_len+3
        self.pes = []
        for i in range(self.max_length):
            self.pes.append(Utils.positional_encoding(i, self.model_size ))
        self.pes = np.concatenate(self.pes, axis=0)
        self.pes = tf.constant(self.pes, dtype=tf.float32)

    
    def loss_func(self,targets, logits):
        """Computes the loss function computation
        """
        mask = tf.math.logical_not(tf.math.equal(targets, 0))
        mask = tf.cast(mask, dtype=tf.int64)
        loss = self.crossentropy(targets, logits, sample_weight=mask)

        return loss
        

    def build_models(self):
        """ 
        Builds the Transformer-encoder architecture and loads the trained 
        weights
        """

        self.encoder = Encoder(self.vocab_size, self.model_size, self.n_layers, 
                               self.n_heads, self.dropout, self.activation_func, 
                               self.ff_dim, self.pes)

        sequence_in = tf.constant([[1, 2, 3, 0, 0]])
        properties_in = tf.constant([[1, 2, 3]])
        encoder_output, _ = self.encoder(sequence_in,properties_in)

        self.encoder.load_weights(self.FLAGS.paths['transformer_standard_path'])   

    # @tf.function
    def predict_step(self,sequences_in):
        
        smiles_padded = Utils.pad_seq_transformer(self.FLAGS,sequences_in,'encoder_in',self.token_table)
            
    	# tokenize - transform the SMILES strings into lists of tokens 
        tokens = Utils.tokenize_transformer(self.FLAGS,smiles_padded,self.token_table,True)   
    
        # Transforms each token to the respective integer, according to 
        # the previously computed dictionary
        seq_idxs_input = Utils().smiles2idx(self.FLAGS,tokens,self.tokenDict)
        
        properties_in = Utils.compute_properties(sequences_in,self.scaler)
        
        encoder_output, alignments = self.encoder(tf.constant(seq_idxs_input),tf.constant(properties_in),training=False)

        return encoder_output.numpy()


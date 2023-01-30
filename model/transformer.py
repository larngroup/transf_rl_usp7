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
import joblib
from random import randrange
import numpy as np
import matplotlib.pyplot as plt
import time

class Transformer(BaseModel):
    """Transformer general Class"""
    def __init__(self, FLAGS):
        super().__init__(FLAGS)

        self.FLAGS = FLAGS
        self.token_table = Utils().voc_table # Load the table of possible tokens
        self.vocab_size = len(self.token_table)
        # Compute the dictionary that makes the correspondence between 
        # each token and unique integers
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
    
        
        # build model and load pre-trained weights
        self.pre_process_data()
        self.build_models()
        
        # Load scaler
        self.scaler = joblib.load('data//scaler_transformer.save') 
        
        
    def pre_process_data(self):
        """ Pre-processes the dataset of molecules including padding, 
            tokenization and transformation of tokens into integers.
    
        Returns
        -------
            pre_processed_dataset (list): List with pre-processed training and 
                                          testing sets
        """      

        self.pes = []
        for i in range(self.max_length+3):
            self.pes.append(Utils.positional_encoding(i, self.model_size ))

        self.pes = np.concatenate(self.pes, axis=0)
        self.pes = tf.constant(self.pes, dtype=tf.float32)


        print(self.pes.shape)

    
    def loss_func(self,targets, logits):
        """ Transformer loss function 

        Parameters
        ----------
        targets : True value
        logits : Model predictions

        Returns
        -------
        loss : outputs the sparse categorical crossentropy loss between the 
               predicted and the true value 

        """
        mask = tf.math.logical_not(tf.math.equal(targets, 0))
        mask = tf.cast(mask, dtype=tf.int64)
        loss = self.crossentropy(targets, logits, sample_weight=mask)

        return loss
    

    @tf.function
    def train_step(self,source_seq, target_seq_in, target_seq_out):
        """ Execute one training step (forward pass + backward pass)
    
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)
        
        Returns:
            The loss value of the current pass
        """
        with tf.GradientTape() as tape:
            encoder_mask = 1 - tf.cast(tf.equal(source_seq, 0), dtype=tf.float32)
            # encoder_mask has shape (batch_size, source_len)
            # we need to add two more dimensions in between
            # to make it broadcastable when computing attention heads
            encoder_mask = tf.expand_dims(encoder_mask, axis=1)
            encoder_mask = tf.expand_dims(encoder_mask, axis=1)
            encoder_output, _ = self.encoder(source_seq, encoder_mask=encoder_mask)
    
            decoder_output, _, _ = self.decoder(
                target_seq_in, encoder_output, encoder_mask=encoder_mask)
    
            loss = self.loss_func(target_seq_out, decoder_output)
    
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
    
        return loss

    # @tf.function
    def train(self):
        """Compiles and trains the model"""
        
        self.crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        

        lr = WarmupThenDecaySchedule(self.model_size)
        self.optimizer = tf.keras.optimizers.Adam(lr,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)        
            
        self.n_epochs = 2
         
        starttime = time.time()
         
        for e in range(self.n_epochs):
            for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(self.tensor_dataset.take(-1)):
                loss = self.train_step(source_seq, target_seq_in,
                                  target_seq_out)
                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Elapsed time {:.2f}s'.format(
                        e + 1, batch, loss.numpy(), time.time() - starttime))
                    starttime = time.time()

            try:
                self.predict()
            except Exception as e:
                print(e)
            continue
        

    def build_models(self):
        """ Builds the Transformer architecture and loads the pre-trained 
            weights
        """

        vocab_size = len(self.token_table)
        self.encoder = Encoder(self.vocab_size, self.model_size, self.n_layers, 
                               self.n_heads, self.dropout, 
                               self.activation_func, self.ff_dim, self.pes)  
 
             
        sequence_in = tf.constant([[1, 2, 3, 0, 0]])
        properties_in = tf.constant([[1, 2, 3]])
        encoder_output, _ = self.encoder(sequence_in,properties_in,training=False)
        print(encoder_output.shape)
        
        self.encoder.load_weights(self.FLAGS.models_path['transformer_standard_path'])   

        
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
        
        tokens,_ = Utils.tokenize_and_pad(self.FLAGS,[sequence_in],self.token_table,'encoder_in')
       
        # Transforms each token into the respective integer, according to 
        # the previously computed dictionary
        seq_idxs_input = Utils.smiles2idx(tokens,self.tokenDict)
      
        properties_in = Utils.compute_properties([sequence_in],self.scaler)

        # Transformer encoder: maps the input tokens to the contextual embeddings and attention scores
        encoder_output, alignments = self.encoder(tf.constant(seq_idxs_input),tf.constant(properties_in),training=False)
        contextual_embedding = encoder_output.numpy()
        contextual_embedding = np.array(contextual_embedding,'float64')
        
        # Extract only the last layer 
        attention_scores = alignments[-1].numpy()
       
        head = randrange(4)
        selected_head = attention_scores[0,head,:,:]        

        # Extract the importance of each specific token
        importance_all = []
       
        size_fake = len(selected_head)
        for c in range(0,size_fake):

            importance_element = []
            importance_element.append(selected_head[c,c])
            
            for v in range(0,size_fake):
                if v!=c:
                    element = (selected_head[c,v] + selected_head[v,c])/2
                    importance_element.append(element)
        
            importance_all.append(importance_element)
        

        importance_tokens = [np.mean(l) for l in importance_all]
            
        scores = Utils.softmax(importance_tokens)
        
        # Sort keeping indexes
        sorted_idxs = np.argsort(-scores)
        
        # Identify most important tokens
        number_tokens = int((self.FLAGS.top_tokens_rate)*len(sequence_in))
        filtered_idxs = sorted_idxs[0:number_tokens-1] 

        idxs_final = []
        search_gap = 0 
        
        for idx_t,t in enumerate(tokens[0]):
            if idx_t in filtered_idxs:
                if len(t)<3:
                    idxs_final.append(idx_t+search_gap+1)
                elif len(t)>=3  and t not in ['[Padd]','[Start]','[End]']:        
                    try:
                        index_start = trajectory_ints.index(17, idx_t+search_gap, idx_t+search_gap+3)
                        index_end = trajectory_ints.index(18, index_start+search_gap, index_start+search_gap+4)
                    except:
                        print(idx_t,t)    
                    for ii in range(index_start,index_end+1):
                        idxs_final.append(ii)
                        
            if len(t)>2 and t not in ['[Padd]','[Start]','[End]']:
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
        
        return scores,idxs_final,contextual_embedding,tokens,alignments
    
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
        
        tokens,_ = Utils.tokenize_and_pad(self.FLAGS,sequences,self.token_table,'encoder_in')
    
        # Transforms each token into the respective integer, according to 
        # the previously computed dictionary
        seq_idxs_input = Utils.smiles2idx(tokens,self.tokenDict)
        
        properties_in = Utils.compute_properties(sequences,self.scaler)
        
        # Transformer encoder: maps the input tokens to the contextual embeddings and attention scores
        encoder_output, alignments = self.encoder(tf.constant(seq_idxs_input),tf.constant(properties_in),training=False)
        
        return encoder_output


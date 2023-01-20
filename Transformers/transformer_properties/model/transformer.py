# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:49:39 2021

@author: tiago
"""

# internal
from .base_model import BaseModel
from utils.utils import Utils
from model.encoder import Encoder
from model.decoder import Decoder
from model.warm_up_decay_schedule import WarmupThenDecaySchedule



# external  sdf 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from math import log

class Transformer(BaseModel):
    """Transformer general Class"""
    
    def __init__(self, FLAGS):
        super().__init__(FLAGS)
        # Implementation parameters
        self.FLAGS = FLAGS
        
        # Load the table of possible tokens
        self.token_table = Utils().voc_table
        self.vocab_size = len(self.token_table)
        
        # Dictionary that makes the correspondence between 
        # each token and unique integers
        self.tokenDict = Utils.smilesDict(self.token_table)
        
        self.inv_tokenDict = {v: k for k, v in self.tokenDict.items()}
        
        # Compute the positional encoding
        self.max_length = self.FLAGS.max_strlen +3
        self.model_size_best = 256
        self.pes = []
        for i in range(self.max_length):
            self.pes.append(Utils.positional_encoding(i, self.model_size_best))
        self.pes = np.concatenate(self.pes, axis=0)
        self.pes = tf.constant(self.pes, dtype=tf.float32)
        

    def loss_func(self,targets, logits):
        """ Loss function masking the padding characters"""
        # print(targets.shape)
        # print(logits.shape)
        mask = tf.math.logical_not(tf.math.equal(targets, 0))
        mask = tf.cast(mask, dtype=tf.int64)
        loss = self.crossentropy(targets, logits, sample_weight=mask)

        return loss
        

    def beam_search(self,k=3):
        """ Employes beam-search to generate k molecules instead of the 
            greedy strategy of selecting only the best option
    
        Args:
            k (int): Number of beams to follow when generating the molecules
        
        Returns:
            K sampled molecules
        """
       
        rnd_idx = np.random.choice(len(self.encoder_in_train))
        properties_sample = self.properties_set_train[rnd_idx,:]
        properties_sample = np.reshape(properties_sample,[1,3])
        sample_test_text_encoder = self.encoder_in_train[rnd_idx]
       
    
        sample_test_text_enc = sample_test_text_encoder[sample_test_text_encoder!=0]
         
        sample_test_idxs_enc = np.reshape(sample_test_text_enc,(1,len(sample_test_text_enc))).tolist()
   
        print('\nInitial scaffold',sample_test_idxs_enc)
        
        en_output, en_alignments = self.encoder(tf.constant(sample_test_idxs_enc), tf.constant(properties_sample), training=False)
        
        start_idx = self.token_table.index('[Start]')
        de_input = tf.constant([[start_idx]], dtype=tf.int64)
        
        de_output, _, _ = self.decoder(de_input, en_output, tf.constant(properties_sample), training=False)
  
        sequences = [[list(),0]]
        counter = 0 
        
        end_idx = self.token_table.index('[End]')
        sequences_final = []
        while k>0 and counter <= 75: 
    
            all_candidates = list()
            for i in range(len(sequences)):
                seq,score = sequences[i]
                
                input_ints = seq.copy()
                input_ints.insert(0,start_idx)
                x = np.reshape(input_ints,(1,len(input_ints)))
                de_output, _, _ = self.decoder(tf.constant(x,tf.int64), en_output, tf.constant(properties_sample), training=False)
                probs_all = de_output.numpy()[-1,-1,:]
            
 
                for j in range(self.vocab_size):
                    candidate = [seq+ [j], score - log(probs_all[j]+1e-13)]
                    
                    all_candidates.append(candidate)

            ordered = sorted(all_candidates, key=lambda tup:tup[1])
            sequences= ordered[:k]

            
            sequences_copy = sequences.copy()
            for idx,s in enumerate(sequences_copy):
                if end_idx in s[0]:
                    sequences_final.append(s[0][:-1])
                    sequences.remove(s)
                    k-=1
                    
            counter+=1
            
        if len(sequences_final) < k:
            sequences_final = sequences_final + sequences
            
        print('\nResult beam search:')
        # print(sequences)
        [print(str(s)+'\n') for s in sequences_final]
        return sequences_final
 

    @tf.function
    def train_step(self,source_seq,properties_set, target_seq_in, target_seq_out):
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
     
            encoder_mask_properties = tf.zeros((encoder_mask.shape[0],3),dtype=tf.float32)
          
            encoder_mask_properties = tf.expand_dims(encoder_mask_properties, axis=1)
            encoder_mask_properties = tf.expand_dims(encoder_mask_properties, axis=1)
            src_mask_encoder = tf.concat([encoder_mask_properties,encoder_mask], axis=3)
            
            # decoder_mask_properties = tf.ones((encoder_mask.shape[0],3),dtype=tf.float32)
            # decoder_mask_properties = tf.expand_dims(decoder_mask_properties, axis=1)
            # decoder_mask_properties = tf.expand_dims(decoder_mask_properties, axis=1)
            # print(decoder_mask_properties.shape)
            # print(decoder_mask_properties.shape)
            src_mask_decoder = tf.concat([encoder_mask_properties,encoder_mask_properties,encoder_mask], axis=3)
            # print(src_mask_decoder.shape)
                        
            encoder_output, _ = self.encoder(source_seq,properties_set, encoder_mask=src_mask_encoder)
    
            decoder_output, _, _ = self.decoder(
                target_seq_in, encoder_output, properties_set,encoder_mask=src_mask_decoder)
    
            loss = self.loss_func(target_seq_out, decoder_output)
    
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
    
        return loss

    # @tf.function
    def train(self,data,FLAGS,n_epochs, batchsize, lr_scheduler, lr_WarmUpSteps,                                   
              min_delta, patience, optimizer_fn, dropout, d_model, n_layers, 
              n_heads, activation_func, ff_dim):
        
        print("\nBuilding model...")    

        self.encoder = Encoder(self.vocab_size, d_model, n_layers, n_heads,
                               dropout, activation_func,ff_dim,self.pes)
 
        sequence_in = tf.constant([[1, 2, 3, 0, 0]])
        properties_in = tf.constant([[1, 2, 3]])
        encoder_output, _ = self.encoder(sequence_in,properties_in)

        self.decoder = Decoder(self.vocab_size, d_model, n_layers, n_heads,
                               dropout, activation_func,ff_dim,self.pes)

        sequence_in = tf.constant([[14, 24, 36]])
        decoder_output, _, _ = self.decoder(sequence_in, encoder_output,properties_in)
        decoder_output.shape 
    

        self.crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False)
        
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
        for e in range(n_epochs):
            print(f'Epoch {e+1}/{n_epochs}') 
            start = time.time() 
            loss_epoch = [] 
            for batch, (source_seq,properties_set, target_seq_in, target_seq_out) in enumerate(data.take(-1)):
                loss_batch = self.train_step(source_seq,properties_set, target_seq_in,
                                  target_seq_out)
                loss_epoch.append(loss_batch)
                
                if batch == len(data.take(-1))-1: 
                    print(f'{batch+1}/{len(data.take(-1))} - {round(time.time() - start)}s - loss: {np.mean(loss_epoch):.4f}') 
            
            if (last_loss['value'] - np.mean(loss_epoch)) >= min_delta: 
                last_loss['value'] = np.mean(loss_epoch) 
                last_loss['epoch'] = e+1
                print('Saving model...') 
                self.encoder.save_weights(FLAGS.checkpoint_path + 'best_encoder.h5')  
                
            if ((e+1) - last_loss['epoch']) >= patience: 
                break 
        
        
       
    # @tf.function   
    def evaluate(self,processed_dataset_test):
        """Predicts resuts for the test dataset"""
        print('\nPredicting on test set...') 
        
        
        test_loss = []

        for num, (smiles_input,properties_inp) in enumerate(processed_dataset_test):
            encoder_output, _= self.encoder(tf.constant(np.array(smiles_input)),properties_inp,training=False) 
            
            decoder_output, _,_= self.decoder(tf.constant(np.array(smiles_input)),encoder_output,tf.constant(np.array(properties_inp)),training=False)
    
            loss_batch = self.loss_func(smiles_input, decoder_output)
 
            test_loss.append(loss_batch.numpy())

            
        print("Loss: ", np.mean(test_loss))
        
        return np.mean(test_loss)
    





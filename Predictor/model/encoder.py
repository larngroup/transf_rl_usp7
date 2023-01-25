# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:27:17 2022

@author: tiago
"""
# Internal
from model.multi_head_attention import MultiHeadAttention

# External
import tensorflow as tf

def create_mask(source_seq):
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
    
    return src_mask_encoder
                        

class Encoder(tf.keras.Model):
    """ Class for the Encoder

    Args:
        vocab_size: number of possible tokens
        model_size: d_model in the paper (depth size of the model)
        num_layers: number of layers (Multi-Head Attention + FNN)
        h: number of attention heads
        dropout: dropout_rate
        activation_func: fully conected nn activation function
        ff_dim: fully connected nn dimension
        pes: positional encoder
        embedding: Embedding layer
        embedding_dropout: Dropout layer for Embedding
        attention: array of Multi-Head Attention layers
        attention_dropout: array of Dropout layers for Multi-Head Attention
        attention_norm: array of LayerNorm layers for Multi-Head Attention
        dense_1: array of first Dense layers for FFN
        dense_2: array of second Dense layers for FFN
        ffn_dropout: array of Dropout layers for FFN
        ffn_norm: array of LayerNorm layers for FFN
    """
    def __init__(self, vocab_size, model_size, num_layers, h, dropout, activation_func,
                 ff_dim, pes): 
        super(Encoder, self).__init__()

        self.pes = pes
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)
        self.embedding_properties = tf.keras.layers.Dense(3*self.model_size,activation=None)
        self.embedding_dropout = tf.keras.layers.Dropout(dropout)
        self.attention = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_dropout = [tf.keras.layers.Dropout(dropout) for _ in range(num_layers)]

        self.attention_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.dense_1 = [tf.keras.layers.Dense(
            ff_dim, activation=activation_func) for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(
            model_size) for _ in range(num_layers)]
        self.ffn_dropout = [tf.keras.layers.Dropout(dropout) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]
        
        self.final_dense = tf.keras.layers.Dense(256)

    def call(self, sequence,properties=None, training=True, encoder_mask=None):
        """ Forward pass for the Encoder

        Args:
            sequence: source input sequences
            training: whether training or not (for Dropout)
            encoder_mask: padding mask for the Encoder's Multi-Head Attention
        
        Returns:
            The output of the Encoder (batch_size, length, model_size)
            The alignment (attention) vectors for all layers
        """
        
        mask_encoder  = create_mask(sequence)
        
        print(sequence.shape)
        print(properties.shape)
        embed_properties = self.embedding_properties(properties) 
        embed_properties = tf.reshape(embed_properties,[properties.shape[0],properties.shape[1],-1])
        
        embed_out = self.embedding(sequence)
     
        embed_out = tf.concat([embed_properties,embed_out], axis=1)
        
        embed_out *= tf.math.sqrt(tf.cast(self.model_size, tf.float32))
        
        embed_out += self.pes[:embed_out.shape[1],:]
        embed_out = self.embedding_dropout(embed_out, training=training)
        
        sub_in = embed_out
        alignments = []

        for i in range(self.num_layers):
            sub_out, alignment = self.attention[i](sub_in, sub_in, mask_encoder)

            sub_out = self.attention_dropout[i](sub_out, training=training)
            sub_out = sub_in + sub_out
            
            sub_out = self.attention_norm[i](sub_out)
            
            alignments.append(alignment)

            ffn_in = sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))

            ffn_out = self.ffn_dropout[i](ffn_out, training=training)
            ffn_out = ffn_in + ffn_out
            ffn_out = self.ffn_norm[i](ffn_out)
            
   
            sub_in = ffn_out

        return ffn_out, alignments
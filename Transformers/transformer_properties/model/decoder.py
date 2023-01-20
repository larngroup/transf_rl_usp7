# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:00:08 2022

@author: tiago
"""
import tensorflow as tf 
from model.multi_head_attention import MultiHeadAttention
class Decoder(tf.keras.Model):
    """ Class for the Decoder

    Args:
        model_size: d_model in the paper (depth size of the model)
        num_layers: number of layers (Multi-Head Attention + FNN)
        h: number of attention heads
        embedding: Embedding layer
        embedding_dropout: Dropout layer for Embedding
        attention_bot: array of bottom Multi-Head Attention layers (self attention)
        attention_bot_dropout: array of Dropout layers for bottom Multi-Head Attention
        attention_bot_norm: array of LayerNorm layers for bottom Multi-Head Attention
        attention_mid: array of middle Multi-Head Attention layers
        attention_mid_dropout: array of Dropout layers for middle Multi-Head Attention
        attention_mid_norm: array of LayerNorm layers for middle Multi-Head Attention
        dense_1: array of first Dense layers for FFN
        dense_2: array of second Dense layers for FFN
        ffn_dropout: array of Dropout layers for FFN
        ffn_norm: array of LayerNorm layers for FFN

        dense: Dense layer to compute final output
    """
    def __init__(self, vocab_size, model_size, num_layers, h, dropout, 
                 activation_func,ff_dim,pes): 
        
        super(Decoder, self).__init__()

        self.pes = pes
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.embedding_properties = tf.keras.layers.Dense(3*self.model_size,activation=None)
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)
        self.embedding_dropout = tf.keras.layers.Dropout(dropout)
        self.attention_bot = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_bot_dropout = [tf.keras.layers.Dropout(dropout) for _ in range(num_layers)]
        self.attention_bot_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]
        self.attention_mid = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_mid_dropout = [tf.keras.layers.Dropout(dropout) for _ in range(num_layers)]
        self.attention_mid_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.dense_1 = [tf.keras.layers.Dense(
           ff_dim, activation=activation_func) for _ in range(num_layers)] #model_size*4
        self.dense_2 = [tf.keras.layers.Dense(
            model_size) for _ in range(num_layers)]
        self.ffn_dropout = [tf.keras.layers.Dropout(dropout) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.dense = tf.keras.layers.Dense(vocab_size,activation=tf.keras.activations.softmax)
        
        # self.initial_dense = tf.keras.layers.Dense(units = 256)
        
    def call(self, sequence, encoder_output, properties=None, training=True, encoder_mask=None):
        """ Forward pass for the Decoder

        Args:
            sequence: source input sequences
            encoder_output: output of the Encoder (for computing middle attention)
            training: whether training or not (for Dropout)
            encoder_mask: padding mask for the Encoder's Multi-Head Attention
        
        Returns:
            The output of the Encoder (batch_size, length, model_size)
            The bottom alignment (attention) vectors for all layers
            The middle alignment (attention) vectors for all layers
        """
        # EMBEDDING AND POSITIONAL EMBEDDING
        embed_out = self.embedding(sequence)
        # embed_properties = self.embedding_properties(properties)
        # embed_properties = tf.reshape(embed_properties,[properties.shape[0],properties.shape[1],-1])
        
        # embed_out = tf.concat([embed_out, embed_properties], axis=1)
        
        embed_out *= tf.math.sqrt(tf.cast(self.model_size, tf.float32))
        embed_out += self.pes[:embed_out.shape[1], :]
        embed_out = self.embedding_dropout(embed_out)
     
        # encoder_output = self.initial_dense(encoder_output)
        embed_properties = self.embedding_properties(properties)
        embed_properties = tf.reshape(embed_properties,[properties.shape[0],properties.shape[1],-1])
        
        encoder_output = tf.concat([embed_properties,encoder_output], axis=1)
    
        

        bot_sub_in = embed_out
        bot_alignments = []
        mid_alignments = []

        for i in range(self.num_layers):
            # BOTTOM MULTIHEAD SUB LAYER
            seq_len = bot_sub_in.shape[1]

            if training:
                mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            else:
                mask = None
            bot_sub_out, bot_alignment = self.attention_bot[i](bot_sub_in, bot_sub_in, mask)
            bot_sub_out = self.attention_bot_dropout[i](bot_sub_out, training=training)
            bot_sub_out = bot_sub_in + bot_sub_out
            bot_sub_out = self.attention_bot_norm[i](bot_sub_out)
            
            bot_alignments.append(bot_alignment)

            # MIDDLE MULTIHEAD SUB LAYER
            mid_sub_in = bot_sub_out
            
            # print('\nsee dims')
            # print(mid_sub_in.shape)   
            # print(encoder_output.shape)   
            mid_sub_out, mid_alignment = self.attention_mid[i](
                mid_sub_in, encoder_output, encoder_mask)
            mid_sub_out = self.attention_mid_dropout[i](mid_sub_out, training=training)
            mid_sub_out = mid_sub_out + mid_sub_in
            mid_sub_out = self.attention_mid_norm[i](mid_sub_out)
            # print(mid_sub_out.shape)  
            mid_alignments.append(mid_alignment)

            # FFN
            ffn_in = mid_sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = self.ffn_dropout[i](ffn_out, training=training)
            ffn_out = ffn_out + ffn_in
            ffn_out = self.ffn_norm[i](ffn_out)

            bot_sub_in = ffn_out
        
        # print('vai alho')
        # print(ffn_out.shape)
        logits = self.dense(ffn_out)
        # print(logits.shape)
        return logits, bot_alignments, mid_alignments
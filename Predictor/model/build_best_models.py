# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:27:17 2022

@author: tiago
"""
# Internal
from model.attention import Attention

# External
import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU


class rnn_model(tf.keras.Model):
    """
    Implements the best Transformer standard-based Predictor
    """
    
    def __init__(self, FLAGS):
        super(rnn_model, self).__init__()
        
        # Implementation parameters
        self.FLAGS = FLAGS
        self.inp_dimension = self.FLAGS.max_str_len
        self.token_len = 47
        self.bidirectional_units = 512
        self.dropout_rate = 0.1
        self.rnn_units = 512
        
        self.bidirectional_layer = tf.keras.layers.Bidirectional(LSTM(self.bidirectional_units, dropout=self.dropout_rate, return_sequences=True)) 
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)
        self.rnn_layer = GRU(self.rnn_units, return_sequences=True)
        self.attention_layer = Attention()
        self.dense_layer = tf.keras.layers.Dense(1,activation='linear') 

    def call(self, sequenc_embed, training=True):

        bidirectional_out = self.bidirectional_layer(sequenc_embed)
        bidirectional_out = self.dropout_layer(bidirectional_out,training=training)
        rnn_out = self.rnn_layer(bidirectional_out)
        rnn_out = self.dropout_layer(rnn_out,training=training)
        attention_out = self.attention_layer(rnn_out)
        pred_out = self.dense_layer(attention_out)
        
        return pred_out
    
class fc_model(tf.keras.Model):
    """
    Implements the best Transformer mlm-based Predictor
    """
    
    def __init__(self, FLAGS):
        super(fc_model, self).__init__()

        self.FLAGS = FLAGS
        self.FLAGS.activation_fc = 'relu'
        self.dropout_rate = 0.1
        self.units_1 = 512
        self.units_2 = 256
        self.units_3 = 128
        
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense_1 = tf.keras.layers.Dense(self.units_1,activation=self.FLAGS.activation_fc) 
        self.dense_2 = tf.keras.layers.Dense(self.units_2,activation=self.FLAGS.activation_fc) 
        self.dense_3 = tf.keras.layers.Dense(self.units_3,activation=self.FLAGS.activation_fc) 
        self.final_dense = tf.keras.layers.Dense(1,activation='linear') 
 
        
    def call(self, inp, training=True):

        dense_1_out = self.dense_1(inp)
        dense_1_out = self.dropout_layer(dense_1_out,training=training)
        dense_2_out = self.dense_2(dense_1_out)
        dense_2_out = self.dropout_layer(dense_2_out,training=training)
        dense_3_out = self.dense_3(dense_2_out)
        dense_3_out = self.dropout_layer(dense_3_out,training=training)

        pred_out = self.final_dense(dense_3_out)
        
        return pred_out
    

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:46:29 2021

@author: tiago
"""
# Internal
from utils.utils import Utils
from model.attention import Attention

# External
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Embedding, Input, GRU, Bidirectional,Concatenate
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

class Build_models(object):
    """
    Unpack data, builds the models' architecture and trains it.
    
    """
    def __init__(self,data_i,FLAGS,split):

        self.token_table = Utils().table
        self.split = split
        
        # Unpack Data  
        self.train_mols = data_i[0]
        self.train_labels = data_i[1]
        self.validation_mols = data_i[2]
        self.validation_labels = data_i[3]
        self.test_mols = data_i[4]
        self.test_labels = data_i[5]

        # Implementation parameters
        self.FLAGS = FLAGS
        
        if self.FLAGS.optimizer_fn[0] == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=float(self.FLAGS.optimizer_fn[1]),
                                                beta_1=float(self.FLAGS.optimizer_fn[2]),
                                                beta_2=float(self.FLAGS.optimizer_fn[3]),
                                                epsilon=float(self.FLAGS.optimizer_fn[4]))
       # elif self.FLAGS.optimizer_fn[0] == 'radam':
       #     optimizer_fun = tfa.optimizers.RectifiedAdam(learning_rate=float(FLAGS.optimizer_fn[1]),
       #                                              beta_1=float(FLAGS.optimizer_fn[2]),
       #                                              beta_2=float(FLAGS.optimizer_fn[3]),
       #                                              epsilon=float(FLAGS.optimizer_fn[4]),
       #                                              weight_decay=float(FLAGS.optimizer_fn[5]))


       # elif FLAGS.optimizer_fn[0] == 'adamw':
       #     optimizer_fun = tfa.optimizers.AdamW(learning_rate=float(FLAGS.optimizer_fn[1]),
       #                                      beta_1=float(FLAGS.optimizer_fn[2]),
       #                                      beta_2=float(FLAGS.optimizer_fn[3]), epsilon=float(FLAGS.optimizer_fn[4]),
       #                                      weight_decay=float(FLAGS.optimizer_fn[5]))

        
        self.model = None
           
    def build_rnn(self):
        """
        Transformer-standard: two RNN layers
        """
        
        self.model = Sequential()
        self.model.add(Input(shape=(self.FLAGS.max_str_len,128)))#
        self.model.add(Embedding(len(self.token_table),256,input_length = self.config.paddSize))
        
        if self.config.rnn == 'lstm':
            self.model.add(LSTM(512, dropout=0.2,return_sequences=True))
            self.model.add(LSTM(512, dropout=0.2))
        elif self.config.rnn == 'gru':
            self.model.add(GRU(256, dropout=0.2, return_sequences=True))
            self.model.add(GRU(256, dropout=0.2))
            
        # self.model.add(Dense(256,activation='relu')) 
        self.model.add(Dense(1,activation='linear'))
        self.model.summary()
 
        self.model.compile(loss="mean_squared_error", optimizer = self.opt, metrics=[Utils().r_square,Utils().rmse,Utils().ccc])

        
    def build_rnn_am(self):
        """
        Transformer-standard: two RNN layers + attention mechanism
        """
        
        self.model = Sequential()
        self.model.add(Input(shape=(self.config.paddSize,)))
        self.model.add(Embedding(len(self.token_table),256,input_length = self.config.paddSize))
        
        if self.config.rnn == 'lstm':
            self.model.add(LSTM(256, dropout=0.2,return_sequences=True))
            self.model.add(LSTM(256,return_sequences=True,dropout=0.2))
        elif self.config.rnn == 'gru':
            self.model.add(GRU(256, dropout=0.2,return_sequences=True))
            self.model.add(GRU(256,return_sequences=True,dropout=0.2,recurrent_dropout=0.2))
            
        self.model.add(Attention())   
        # self.model.add(Dense(256,activation='relu'))
        self.model.add(Dense(1,activation='linear'))
        self.model.summary()
             
        self.model.compile(loss="mean_squared_error", optimizer = self.opt, metrics=[Utils().r_square,Utils().rmse,Utils().ccc])

        
    def build_bi_rnn(self):
        """
        Transformer-standard: one bidirectional RNN layer
        """

        input_data = Input(shape=(self.config.paddSize,), name = 'encoder_inputs')
  
        x = Embedding(len(self.token_table),256,input_length = self.config.paddSize) (input_data)

        layer_bi_1 = Bidirectional(GRU(256, dropout=0.2, return_sequences=False))

        x_combined = layer_bi_1(x)

        # x_combined = Dense(256,activation = 'relu')(x_combined)
        output = Dense(1,activation='linear') (x_combined)

        self.model = Model(input_data, output)  

        self.model.summary()
        self.model.compile(loss="mean_squared_error", optimizer = self.opt, metrics=[Utils().r_square,Utils().rmse,Utils().ccc])

    def build_bi_rnn_am(self):
        """
        Transformer-standard: one bidirectional RNN layer + attention mechanism
        """

        input_data = Input(shape=(75,128,), name = 'encoder_inputs')
  
        # x = Embedding(len(self.token_table),64,input_length = self.config.paddSize) (input_data)

        layer_bi_1 = Bidirectional(GRU(256, dropout=0.2, return_sequences=True))

        x_combined = layer_bi_1(input_data)
        
        x_att = Attention()(x_combined)

        # x_combined = Dense(256,activation = 'relu')(x_att)
        output = Dense(1,activation='linear') (x_att)

        self.model = Model(input_data, output)  

        self.model.summary()
        self.model.compile(loss="mean_squared_error", optimizer = self.opt, metrics=[Utils().r_square,Utils().rmse,Utils().ccc])

    def build_tri_rnn(self):
        """
        Transformer-standard: one bidirectional RNN + normal RNN
        """

        input_data = Input(shape=(self.config.paddSize,), name = 'encoder_inputs')
   
        x = Embedding(len(self.token_table),256,input_length = self.config.paddSize) (input_data)

        layer_bi_1 = Bidirectional(GRU(256, dropout=0.2, return_sequences=True))

        x_combined = layer_bi_1(x)
         
        layer_lstm_1 = GRU(256, dropout=0.2,return_sequences=False)
        
        x_lstm = layer_lstm_1(x_combined)
         
         # x_combined = Dense(256,activation = 'relu')(x_lstm)
         
        output = Dense(1,activation='linear') (x_lstm)

        self.model = Model(input_data, output)  

        self.model.summary()
        self.model.compile(loss="mean_squared_error", optimizer = self.opt, metrics=[Utils().r_square,Utils().rmse,Utils().ccc])

    def build_tri_rnn_am(self):
        """
        Transformer-standard: one bidirectional RNN + normal RNN + attention mechanism
        """
        
        # input_data = Input(shape=(self.config.paddSize,), name = 'encoder_inputs')
        input_data = Input(shape=(self.FLAGS.max_str_len+3,self.FLAGS.model_size_std,), name = 'encoder_inputs')
        # x = Embedding(len(self.token_table),256,input_length = self.config.paddSize) (input_data)
        
        if self.FLAGS.rnn_type == 'gru':
            
            if self.FLAGS.bi_rnn:
                layer_1 = Bidirectional(GRU(self.FLAGS.rnn_1,dropout=self.FLAGS.dropout_rnn, return_sequences=True))
            else:
                layer_1 = GRU(self.FLAGS.rnn_1,dropout=self.FLAGS.dropout_rnn, return_sequences=True)
                
            x_combined = layer_1(input_data)
            
            layer_2 = GRU(self.FLAGS.rnn_2, dropout=self.FLAGS.dropout_rnn,return_sequences=True)
            x_lstm = layer_2(x_combined)
        
        elif self.FLAGS.rnn_type == 'lstm':
            
            if self.FLAGS.bi_rnn:
                layer_1 = Bidirectional(LSTM(self.FLAGS.rnn_1,dropout=self.FLAGS.dropout_rnn, return_sequences=True))
            else:
                layer_1 = LSTM(self.FLAGS.rnn_1,dropout=self.FLAGS.dropout_rnn, return_sequences=True)

            x_combined = layer_1(input_data)
            
            layer_2 = LSTM(self.FLAGS.rnn_2, dropout=self.FLAGS.dropout_rnn,return_sequences=True)
            x_lstm = layer_2(x_combined)
        
        x_att = Attention()(x_lstm)

        # x_combined = Dense(256,activation = 'relu')(x_att)
        output = Dense(1,activation='linear') (x_att)

        self.model = Model(input_data, output)  

        self.model.summary()
        self.model.compile(loss="mean_squared_error", optimizer = self.optimizer, metrics=[Utils().r_square,Utils().rmse,Utils().ccc])
        
    def build_tri_rnn_am_mlm(self):
        """
        Transformer-standard: n dense layers
        """
                
        input_data = Input(shape=(self.FLAGS.model_size_mlm,), name = 'encoder_inputs')
        
        f = 1
        
        x_combined = Dense(self.FLAGS.units/f,activation = self.FLAGS.activation_fc)(input_data)
        x_combined = Dropout(self.FLAGS.dropout_fc)(x_combined)
        
        for l in range(0,self.FLAGS.n_layers-1):
            f=f*2
            print(l,self.FLAGS.units/f)
            x_combined = Dense(self.FLAGS.units/f,activation = self.FLAGS.activation_fc)(x_combined)
            x_combined = Dropout(self.FLAGS.dropout_fc)(x_combined)
            
        
        output = Dense(1,activation = 'linear')(x_combined)

        self.model = Model(input_data, output)  
        self.model.summary()
        self.model.compile(loss="mean_squared_error", optimizer = self.optimizer, metrics=[Utils().r_square,Utils().rmse,Utils().ccc])

    def build_separated_bi_rnn_am(self):
        """
        Transformer-standard: one bidirectional RNN + attention forward + 
                                attention backward
        """
        
        input_data = Input(shape=(self.config.paddSize,), name = 'encoder_inputs')
  
        x = Embedding(len(self.token_table),64,input_length = self.config.paddSize) (input_data)

        layer_bi_1 = Bidirectional(LSTM(64, dropout=0.2, return_sequences=True),merge_mode = None)
        x_forward,x_backward = layer_bi_1(x)
        
        x_att_for = Attention()(x_forward)
        x_att_backward = Attention()(x_backward)
        
        x_concat = Concatenate(axis=1)([x_att_for,x_att_backward])
    
        output = Dense(1,activation='linear') (x_concat)

        self.model = Model(input_data, output)  

        self.model.summary()
        self.model.compile(loss="mean_squared_error", optimizer = self.opt, metrics=[Utils().r_square,Utils().rmse,Utils().ccc])

      
    def train_dl_model(self):
        """
        Training process applying early stopping and modelcheckpoint
        """
               
       	#Reduces the learning rate by a factor of 0.8 when no improvement has been see in the validation set for 4 epochs
        reduce_lr = ReduceLROnPlateau(monitor = "val_loss", factor = float(self.FLAGS.reduction_lr[0]), patience=int(self.FLAGS.reduction_lr[1]), min_lr = float(self.FLAGS.reduction_lr[2]))
       
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.FLAGS.patience, restore_best_weights=True)
        mc = ModelCheckpoint(self.FLAGS.checkpoint_path + 'best_model_' + str(self.split)+'.hdf5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

        history = self.model.fit(self.train_mols,self.train_labels,epochs=self.FLAGS.n_iterations,
                                    batch_size=self.FLAGS.batch_size,validation_data=(self.validation_mols, self.validation_labels),callbacks=[es,mc,reduce_lr])
        
        self.model.save(self.FLAGS.checkpoint_path + 'final_model_' + str(self.split)+'.hdf5') 

        if self.FLAGS.finetuning:
            
            self.model_ft = self.model
            
            print("\nFinetuning training process\n")
            self.model_ft.fit(self.train_mols_ft,self.train_labels_ft,epochs=self.FLAGS.n_iterations_ft,
                                        batch_size=self.FLAGS.batch_size)
        
        if self.FLAGS.print_result:
            
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'],color='red')
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.ylim([0, 2])
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
        

# -*- coding: utf-8 -*-

# External 
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Embedding
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from math import log
import matplotlib.pyplot as plt

class Generator:
    def __init__(self,FLAGS,model_type):
        self.tokens = ['H','Se','se','As','Si','Cl', 'Br','B', 'C', 'N', 'O', 'P', 
                  'S', 'F', 'I', '(', ')', '[', ']', '=', '#', '@', '*', '%', 
                  '0', '1', '2','3', '4', '5', '6', '7', '8', '9', '.', '/',
                  '\\', '+', '-', 'c', 'n', 'o', 's','p','<Start>','<End>','<Padd>']
        
        self.model_type = model_type
        self.model = None

        self.vocab_size = len(self.tokens)
        self.emb_dim = 256
        self.max_len = 100
        self.n_layers = 2
        self.units = 512
        self.dropout_rate = 0.2
        self.activation = "softmax"
        self.epochs = 50
        self.batch_size = 16
        
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False, name='RMSprop')
            
        self.build()
        
    def build(self):        
        """
        Initialization of the Generator architecture
        """
        self.model = Sequential()

        self.model.add(Embedding(self.vocab_size, self.emb_dim, input_length = self.max_len))
        
        for i in range(self.n_layers): 
            self.model.add(LSTM(self.units, return_sequences=True))            
            if self.dropout_rate != 0:
                self.model.add(Dropout(self.dropout_rate))

        
        self.model.add(Dense(units = self.vocab_size, activation = self.activation))
        
        print(self.model.summary())
        
        if self.model_type == True:
            self.model.compile(optimizer = self.optimizer, loss = 'sparse_categorical_crossentropy') #'mse' emb
        

    def load_model(self, path):
        """
        Loading of the weights of the pre-trained Generator 
        """
        self.model.load_weights(path)

    def fit_model(self, dataX, dataY):
        """
        Generator training process 
        """
        filename="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"       
        early_stop = EarlyStopping(monitor = "loss", patience=5)    
        path = self.path+F"{filename}"
        checkpoint = ModelCheckpoint(path, monitor = 'loss', verbose = 1, mode = 'min') 
        callbacks_list = [checkpoint, early_stop]#, PlotLossesKerasTF()]
        results = self.model.fit(dataX, dataY, verbose = 1, epochs = self.epochs, batch_size = self.batch_size, shuffle = True, callbacks = callbacks_list)
        
        #plot
        fig, ax = plt.subplots()
        ax.plot(results.history['loss'])
        ax.set(xlabel='epochs', ylabel = 'loss')
        figure_path = self.path + "Loss_plot.png"
        fig.savefig(figure_path)
        #plt.show()
        last_epoch = early_stop.stopped_epoch
        
        return results, last_epoch
    
    
    def sample_with_temp(self, preds):
        """
        Sampling of the next token considering the probability array 'preds'
        preds: probabilities of choosing a character
        """
       # self.config.sampling_temp
        preds_ = np.log(preds).astype('float64')/0.8
        probs= np.exp(preds_)/np.sum(np.exp(preds_))
        #out = np.random.choice(len(preds), p = probs)
        
        # out_normal=np.argmax(np.random.multinomial(1,probs, 1))
        out_old = np.random.choice(len(probs), p=probs)

        return out_old
    
    
    def softmax(self, preds):

        preds_ = np.log(preds).astype('float64')/1.0
        probs= np.exp(preds_)/np.sum(np.exp(preds_))
       
        return probs
        
    
    def generate(self, numb):
        """
        Generatio of 'numb' new SMILES strings
        """

        list_seq = []
        list_ints = []
        

        start_idx = self.tokens.index('<Start>')       
        end_idx = self.tokens.index('<End>')
        
        for j in range(numb):
            list_ints = [start_idx]
            smi = '<Start>'
            #x = np.reshape(seq, (1, len(seq),1))
            
            for i in range(self.max_len-1):
                x = np.reshape(list_ints, (1, len(list_ints),1))
                preds = self.predict(x)
                
                #sample
                #index = np.argmax(preds[0][-1])
                #sample with T
                index = self.sample_with_temp(preds[0][-1])
                list_ints.append(index)
                smi +=self.tokens[index]
                if (index) == end_idx:
                    break
            list_seq.append(smi)

        return list_ints,list_seq
        
    def predict(self, input_x):
        preds = self.model.predict(input_x)
        return preds
    
    def beam_search(self,k=3):
           
        counter = 0 
        sequences = [[list(), 0.0]]
        end_idx = self.tokens.index('<End>') 
        sequences_final = []
        
        # walk over each step in sequence 
        while counter < 75 and k>0:
            
            all_candidates = list() 
            # expand each current candidate 
            for i in range(len(sequences)): 
                seq, score = sequences[i] 
                if counter == 0:
                    start_idx = self.tokens.index('<Start>') 
                    list_ints = [start_idx]
                    x_init = np.reshape(list_ints, (1, len(list_ints),1))
                    preds = self.predict(x_init)
                    preds_out_soft = self.softmax(preds)
                else:    
                    
                    inputs_ints = seq.copy()
                    
                    inputs_ints.insert(0, 44)

                    x = np.reshape(inputs_ints, (1, len(inputs_ints),1))
         
        
                    preds = self.predict(x)
                    preds_out = preds[0,-1,:]
                    preds_out_soft = self.softmax(preds_out)
                    
                
                for j in range(47): 
                    try:
                        if counter == 0 :
                            candidate = [seq + [j], score - log(preds_out_soft[0,0,j]+1e-12)]
                            
                        else: 
                            candidate = [seq + [j], score - log(preds_out_soft[j]+1e-12)]
                    except:
                        print('fio')
                    all_candidates.append(candidate) 
            # order all candidates by score 
            ordered = sorted(all_candidates, key=lambda tup:tup[1]) 
            # select k best 
            sequences = ordered[:k] 

            counter+=1
        
        sequences_final = [s[0][:s[0].index(45)] for idx,s in enumerate(sequences)]
        print('\nResult beam search: ')
        print(sequences_final)
            
        return sequences_final
  
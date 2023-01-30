# -*- coding: utf-8 -*-
# Internal 
from utils.utils import Utils
from model.model_predictor import Rnn_predictor,Fc_predictor

# External 
import tensorflow as tf
import numpy as np
import joblib

class BaseModel(object):
    def __init__(self,FLAGS):
        """
        This class builds the Predictor model and loads the pre-trained model.
        """
        
        self.tokens = ['H','Se','se','As','Si','Cl', 'Br','B', 'C', 'N', 'O', 'P', 
          'S', 'F', 'I', '(', ')', '[', ']', '=', '#', '@', '*', '%', 
          '0', '1', '2','3', '4', '5', '6', '7', '8', '9', '.', '/',
          '\\', '+', '-', 'c', 'n', 'o', 's','p','G','E','A']
        
        self.labels = Utils.read_csv(FLAGS)
       
class DnnQSAR_model(BaseModel):
    
    def __init__(self,FLAGS):
        super(DnnQSAR_model, self).__init__(FLAGS)
        self.FLAGS = FLAGS
        
        
        self.scaler = joblib.load('data//scaler_predictor.save') 
        
        if FLAGS.option == 'mlm' or FLAGS.option == 'mlm_exp1' or FLAGS.option == 'mlm_exp2':
            self.predictor = Fc_predictor(FLAGS)
            sequence_in = tf.constant([list(np.ones((1,256)))])
            prediction_test = self.predictor(sequence_in)
            self.predictor.load_weights(FLAGS.models_path['predictor_mlm'])
        

        elif FLAGS.option == 'standard' or FLAGS.option == 'standard_exp1' or FLAGS.option == 'standard_exp2':
            self.predictor = Rnn_predictor(FLAGS)
            sequence_in = tf.constant([list(np.ones((1,256)))])
            prediction_test = self.predictor(sequence_in)
            self.predictor.load_weights(FLAGS.models_path['predictor_standard'])

        
             
    def predict(self, smiles_original):
        """
        This function performs the prediction of the USP7 pIC50 for the input 
        molecules
        Parameters
        ----------
        smiles_original: List of SMILES strings to perform the prediction  
        
        Returns
        -------
        This function performs the denormalization step and returns the model's
        prediction
        """
        prediction = self.predictor.predict(smiles_original)            
        prediction = Utils.denormalization(prediction,self.labels,self.scaler)
                
        return prediction
        

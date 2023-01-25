# -*- coding: utf-8 -*-

# External
import numpy as np
from tensorflow.keras.optimizers import Adam

# Internal
from utils.utils import Utils

class Predictor(object):
    def __init__(self,obj_build_models,tokens,FLAGS):
        """
        This class loads the previously trained models, whether they are the 
        DNN-based or the standard QSAR models, to evaluate them with the test set
        ----------        
        obj_build_models: Object of the previously trained model
        tokens: List with all possible tokens forming a SMILES string
        FLAGS: Implementation parameters
        Returns
        -------
        This function loads the trained parameters and evaluates the model 
        by predicting the true output of the test set.
        """
        super(Predictor, self).__init__()
        self.obj_build_models = obj_build_models
        self.tokens = tokens
        self.FLAGS = FLAGS
   
        loaded_models_finetuned = []
        loaded_models = []
        
        for i in range(5):
                
            loaded_model = self.obj_build_models.model
            
            # load weights into new model
            loaded_model.load_weights(self.FLAGS.checkpoint_path + 'final_model_' + str(i+1)+'.hdf5')

            print("Model " + str(i) + " loaded from disk!")
            print(loaded_model)
            loaded_models.append(loaded_model)
        
        self.loaded_models = loaded_models
        self.loaded_models_finetuned = loaded_models_finetuned
        
            
    def evaluator_cv(self,data,scaler,finetuning = False):
        """
        This function evaluates the trained Predictor
        ----------
        data: List with testing SMILES and testing labels
        scaler: robustscaler object 
        Returns
        -------
        The evaluated metrics for the test and validation sets (MSE,RMSE,R2,CCC)
        """
        
        print("\n------- Evaluation with test set -------")
        
        if self.FLAGS.optimizer_fn[0] == 'adam':
           opt =Adam(learning_rate=float(self.FLAGS.optimizer_fn[1]),
                                                beta_1=float(self.FLAGS.optimizer_fn[2]),
                                                beta_2=float(self.FLAGS.optimizer_fn[3]),
                                                epsilon=float(self.FLAGS.optimizer_fn[4]))
        
        smiles_test = data[4]
        label_test = data[5]
        metrics_test = []
        prediction_test = []
        
        smiles_val = data[2]
        label_val = data[3]
        metrics_val = []
        prediction_val = []

        for m in range(len(self.loaded_models)):
            self.loaded_models[m].compile(loss="mean_squared_error", optimizer = opt, metrics=[Utils().mse,Utils().rmse,Utils().r_square,Utils().ccc])
            self.loaded_models[m].load_weights(self.FLAGS.checkpoint_path + 'best_model_' + str(m+1)+'.hdf5')
            metrics_test.append(self.loaded_models[m].evaluate(smiles_test,label_test))
            prediction_test.append(self.loaded_models[m].predict(smiles_test))
            
            metrics_val.append(self.loaded_models[m].evaluate(smiles_val,label_val))
            prediction_val.append(self.loaded_models[m].predict(smiles_val))
                
        print(metrics_test)
        print(metrics_val)

        metrics_test = np.array(metrics_test).reshape(len(self.loaded_models), -1)
        metrics_test = metrics_test[:,1:5]
        metrics_test = np.mean(metrics_test, axis = 0)
        

        metrics_val = np.array(metrics_val).reshape(len(self.loaded_models), -1)
        metrics_val = metrics_val[:,1:5]
        metrics_val = np.mean(metrics_val, axis = 0)
        
        print(metrics_test)
        print(metrics_val)
      
        return metrics_test,metrics_val

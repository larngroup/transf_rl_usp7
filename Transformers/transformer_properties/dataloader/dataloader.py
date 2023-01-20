# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:48:26 2021

@author: tiago
"""

from rdkit import Chem
from sklearn.model_selection import train_test_split
from utils.utils import Utils
import tensorflow as tf

class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_smiles(FLAGS):
        """ Loads the molecular dataset and filters the compounds considered
             syntactically invalid by RDKit.
        Args:
            FLAGS (argparse): Implementation parameters

        Returns:
            dataset (list): The list with the training and testing SMILES 
        """
       
        mols_chembl = [] 
            

        file = open(FLAGS.paths['chembl_data'],"r")
        lines = file.readlines()        
        for line in lines:
            x =line.split()
            try:
                mol = Chem.MolFromSmiles(x[0].strip())
                smi = Chem.MolToSmiles(mol)
                mols_chembl.append(smi)
            except:
                print("Invalid molecule")
        
        
        
        mols_drugs = [] 
            

        file = open(FLAGS.paths['usp7_data'],"r")
        lines = file.readlines()        
        for line in lines:
            x =line.split(',')
            try:
                mol = Chem.MolFromSmiles(x[-1].strip())
                smi = Chem.MolToSmiles(mol)
                mols_drugs.append(smi)
            except:
                print("Invalid molecule",x[-1])
        
        all_mols = mols_chembl +list(set(mols_drugs))
        return all_mols
    
    
    def pre_process_data(dataset,transformer_model,FLAGS):
        """ Pre-processes the dataset of molecules including padding, 
            tokenization and transformation of tokens into integers.
        
        Args    
            dataset (list): List with all loaded molecules
            transformer_model (class): Transformer object
            FLAGS (argparse): Implementation parameters
        
        Returns
        -------
            data_train (list): Set of training molecules
            data_test (list): Set of testing molecules
        """     
        print("\nPre-processing data...")
        # split the data in training and testing sets
        train_smiles, test_smiles = train_test_split(dataset, test_size=FLAGS.test_rate, random_state=55)
                  
        pre_processed_dataset = []
        for idx in range(0,4):
            # Padd each SMILES string with the padding character until 
            # reaching the size of the largest molecule
        
            if idx == 0:
                data_type = 'encoder_in'
                mols_set = train_smiles
                filtered_smiles = [item for item in mols_set if len(item)<=FLAGS.max_strlen-1]
                properties_set_train = Utils.compute_normalize_properties(FLAGS,filtered_smiles)
            elif idx == 1:
                data_type = 'decoder_in'
                mols_set = train_smiles
            elif idx == 2:
                data_type = 'decoder_out'
                mols_set = train_smiles
            elif idx == 3:
                data_type = 'test_data'
                mols_set =test_smiles
                filtered_smiles = [item for item in mols_set if len(item)<=FLAGS.max_strlen-1]
                properties_set_test = Utils.compute_normalize_properties(FLAGS,filtered_smiles)

            tokens,_ = Utils.tokenize_and_padd(FLAGS,mols_set,data_type,transformer_model.token_table)
        
            # Maps each token to the respective integer, according to 
            # the previously computed dictionary
            pre_processed_dataset.append(Utils.smiles2idx(tokens,transformer_model.tokenDict))
                
        encoder_in_train = pre_processed_dataset[0]
        decoder_in_train = pre_processed_dataset[1]
        decoder_out_train = pre_processed_dataset[2]
        data_test = pre_processed_dataset[3]
 
        # Create the dataset object
        data_train = tf.data.Dataset.from_tensor_slices(
            (encoder_in_train, properties_set_train, decoder_in_train, decoder_out_train))
        data_train = data_train.shuffle(len(encoder_in_train)).batch(FLAGS.batchsize)
        
        # Create the testing dataset object
        data_test = tf.data.Dataset.from_tensor_slices((data_test,properties_set_test))
        data_test = data_test.shuffle(len(data_test)).batch(FLAGS.batchsize)        
        
        return data_train,data_test            
        

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
        """ Loads the molecular dataset, filtering the compounds considered
             syntactically invalid by RDKit.
        Args:
           FLAGS (argparse): Implementation parameters

        Returns:
            dataset (list): The list with the training and testing SMILES 
        """
       
        all_mols = [] 
            

        file = open(FLAGS.data_path,"r")
        lines = file.readlines()        
        for line in lines:
            x =line.split()
            try:
                mol = Chem.MolFromSmiles(x[0].strip())
                smi = Chem.MolToSmiles(mol)
                all_mols.append(x[0].strip())
            except:
                print("Invalid molecule")
        
        return all_mols
    
    
    def pre_process_data(dataset,transf_obj,FLAGS):
        """ Pre-processes the dataset of molecules including padding, 
            tokenization and transformation of tokens into integers.
            
        Args: 
            dataset (list): Set of molecules
            transf_obj (class): Transformer object 
            FLAGS (argparse): Implementation parameters
    
        Returns
        -------
            data_train (list): List with pre-processed training set
            data_test (list): List with pre-processed testing set
        """     
        
        # split the data in training and testing sets
        train_smiles, test_smiles = train_test_split(dataset, test_size=FLAGS.test_rate, random_state=55)
                   
       	# Tokenize - transform the SMILES strings into lists of tokens 
        tokens,smiles_filtered = Utils().tokenize_and_pad(FLAGS,dataset,transf_obj.token_table)   
   
       # Identify the functional groups of each molecule
        fgs = Utils().identify_fg(FLAGS,smiles_filtered)
                
       # Transforms each token to the respective integer, according to 
       # the previously computed dictionary
        input_raw = Utils.smiles2idx(tokens,transf_obj.tokenDict)
                    
        # 'Minimumm number of functional groups (FGs) where masking considers FGs'
        threshold_min = FLAGS.threshold_min
        
        # masking mlm
        input_masked, masked_positions = transf_obj.masking(input_raw, fgs, threshold_min)   
        
        print(input_masked[0].shape)
        print(input_raw[0].shape)
        # Create tf.dataset object
        inp = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(input_masked), tf.convert_to_tensor(masked_positions))).batch(FLAGS.batchsize, drop_remainder = False)
        label = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(input_raw)).batch(FLAGS.batchsize, drop_remainder = False) 
        data_train = tf.data.Dataset.zip((inp, label))
        


    	# Tokenize - transform the SMILES strings into lists of tokens 
        tokens_test,smiles_filtered_test = Utils().tokenize_and_pad(FLAGS,test_smiles,transf_obj.token_table)   

        # Identify the functional groups of each molecule
        fgs_test = Utils().identify_fg(FLAGS,smiles_filtered_test)
             
        # Transforms each token to the respective integer, according to 
        # the previously computed dictionary
        input_test = Utils().smiles2idx(tokens_test,transf_obj.tokenDict)
        
        inp_smiles_masked_test, masked_positions_test = transf_obj.masking(input_test, fgs_test, FLAGS.threshold_min) 
        
        data_test = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(input_test), tf.convert_to_tensor(inp_smiles_masked_test),tf.convert_to_tensor(masked_positions_test))).batch(FLAGS.batchsize, drop_remainder = False)
        
        return data_train,data_test
            
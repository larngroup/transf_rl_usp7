# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:48:26 2021

@author: tiago
"""
# Internal
from model.generator import Generator  
from model.dnnQSAR import DnnQSAR_model


# External
from keras.models import Sequential
import csv
from rdkit import Chem
import openpyxl

class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_generator(FLAGS,generator_type):
        """ Initializes and loads the weights of the trained generator model, 
            whether the unbiased or biased model.

        Args
        ----------
            FLAGS (argparse): Implementation parameters
            generator_type (str): Indication if its to load the unbiased or the
                                  biased model 

        Returns
        -------
            generator_model (sequential): model with the trained weights
        """
                
        generator_model = Sequential()
        generator_model=Generator(FLAGS,True)
        
        path = ''
        if generator_type == 'biased':
            path = FLAGS.checkpoint_path+'biased_generator.hdf5'
        elif generator_type == 'unbiased':
            path =FLAGS.models_path['generator_unbiased_path'] 
        
        generator_model.model.load_weights(path)
        
        return generator_model
    
    @staticmethod
    def load_predictor(FLAGS):
        """ Initializes and loads biological affinity Predictor

        Args
        ----------
            FLAGS (argparse): Implementation parameters

        Returns
        -------
            predictor_obj (object): Predictor object that enables application 
                                of the model to perform predictions 
        """
        
        predictor_obj = DnnQSAR_model(FLAGS)
        
        return predictor_obj
    
    @staticmethod
    def load_promising_mols(FLAGS):
        """ Loads a set of pre-generated molecules

        Args
        ----------
            FLAGS (argparse): Implementation parameters

        Returns
        -------
            df (pandas DataFrame): object that contains the smiles and the 
                                   corresponding biological affinity values
        """
        
        smiles = []
        pic50_values = []
            
        with open(FLAGS.path_promising_hits, 'r') as csvFile:
            reader = csv.reader(csvFile)
            
            it = iter(reader)

            for idx,row in enumerate(it):
                
                if idx > 3:
                    smiles.append(row[0])
                    pic50_values.append(float(row[1]))
       
        return smiles
    
    @staticmethod
    def load_usp7_mols(FLAGS):
        """ Loads a set of pre-generated molecules

        Args
        ----------
            FLAGS (argparse): Implementation parameters

        Returns
        -------
            smiles_all (pandas DataFrame): object that contains the smiles and the 
                                   corresponding biological affinity values
        """
        
        smiles_all = []
                
        file_names = [FLAGS.models_path['usp7_path_1'],FLAGS.models_path['usp7_path_2']]        
        for xlsx_file in file_names:

            wb_obj = openpyxl.load_workbook(xlsx_file) 
    
            # Read the active sheet:
            sheet = wb_obj.active
            
            for idx,row in enumerate(sheet.iter_rows()):

                if idx > 0 and isinstance(row[3].value, str):
                    try:
                        mol = Chem.MolFromSmiles(row[3].value, sanitize=True)
                        s = Chem.MolToSmiles(mol)
                        smiles_all.append(s)
                    except:
                        print('Invalid mol')
        return smiles_all

    @staticmethod
    def load_generator_smiles(data_path):
        """ Loads the molecular dataset and filters the compounds considered
             syntactically invalid by RDKit.
        Args:
            data_path (str): The path of all SMILES set

        Returns:
            all_mols (list): The list with the training and testing SMILES of
                               the Transformer model
        """
       
        all_mols = [] 
        file = open(data_path,"r")
        lines = file.readlines()        
        for line in lines:
            x = line.split()
            try:
                mol = Chem.MolFromSmiles(x[0].strip())
                smi = Chem.MolToSmiles(mol)
                all_mols.append(smi)
            except:
                print("Invalid molecule")
        
        return all_mols
    
    
        
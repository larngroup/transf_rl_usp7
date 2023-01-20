# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:49:12 2021

@author: tiago
"""
# External
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors,QED
from sklearn.preprocessing import RobustScaler
import joblib

class Utils:
    """Data Loader class"""
    
    def __init__(self):
        """ Definition of the SMILES vocabulary """
    
        
        self.voc_table  = ['[Padd]','[o+]','[NH+]','[NH-]','[S+]','[O+]','[SH+]','[n-]','[2H]',
                       '[N+]','[SH]','[N-]','[n+]','[S-]','[NH3+]','[s+]','[S@]','[Si]','[Au]',
                       '[O]','[CH]','[O-]','[N]','[NH2+]','[nH]','[nH+]','[C@H]','[P@@]','[Mg]',
                       '[C@@H]','[C@@]','[C@]','Cl','Br','C','N','O','S','F','B','I','P','.',
                       '(', ')','=','#','%','0','1','2','3','4','5','6','7','8','9','\\','-','/',
                       '[S@@]','[Pt]','[N@]','[As]','[Co]','[se]','[Hg]','[Ca]','[Co+]',
                       '[Cr]','[N@@+]','[I+]','[N@@]','[C-]','c', 'n', 'o', 's','[Start]','[End]']
        
    @staticmethod
    def smilesDict(tokens):
        """ Computes the dictionary that makes the correspondence between 
        each token and an given integer.

        Args
        ----------
            tokens (list): List of each possible symbol in the SMILES

        Returns
        -------
            tokenDict (dict): Dictionary mapping characters into integers
        """

        tokenDict = dict((token, i) for i, token in enumerate(tokens))
        return tokenDict
    

    @staticmethod            
    def smiles2idx(smiles,tokenDict):
        """ Transforms each token in the SMILES into the respective integer.

        Args
        ----------
            smiles (list): SMILES strings with different sizes
            tokenDict (dict): Dictionary mapping characters to integers 

        Returns
        -------
            newSmiles (list): List of transformed smiles, with the characters 
                              replaced by the numbers. 
        """   
        
        newSmiles =  np.zeros((len(smiles), len(smiles[0])))
        for i in range(0,len(smiles)):
            # print(i, ": ", smiles[i])
            for j in range(0,len(smiles[i])):
                
                try:
                    newSmiles[i,j] = tokenDict[smiles[i][j]]
                except:
                    value = tokenDict[smiles[i][j]]
        return newSmiles
        
    @staticmethod             
    def idx2smi(model_output,tokenDict):
        """ Transforms model's predictions into SMILES

        Args
        ----------
            model_output (array): List with the autoencoder's predictions 
            tokenDict (dict): Dictionary mapping characters into integers

        Returns
        -------
            reconstructed_smiles (array): List with the reconstructed SMILES 
                                          obtained by transforming indexes into
                                          tokens. 
        """           

        key_list = list(tokenDict.keys())
        val_list = list(tokenDict.values())

        reconstructed_smiles =  []
        for i in range(0,len(model_output)):
            smi = []
            for j in range(0,len(model_output[i])):
                
                smi.append(key_list[val_list.index(model_output[i][j])])
                
            reconstructed_smiles.append(smi)
                
        return reconstructed_smiles
    
              

    def positional_encoding(pos, model_size):
        """ Compute positional encoding for the Transfomer's input
    
        Args:
            pos: position of a token in the sequence
            model_size: depth size of the model
        
        Returns:
            The positional encoding for the given token
        """
        PE = np.zeros((1, model_size))
        for i in range(model_size):
            if i % 2 == 0:
                PE[:, i] = np.sin(pos / 10000 ** (i / model_size))
            else:
                PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
        return PE
        
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
               
    def compute_normalize_properties(FLAGS,mols_list):
        """ Computes and normalize the drug-like properties: logP, TPSA, and QED
    
        Args
        ----------
            FLAGS (argparse): Implementation parameters        
            mols_lsit (list): Set of SMILES 

        Returns
        -------
            scaled_prop (array): Array containing the three mentioned properties
                                normalized using statistics that are robust to
                                outliers. 
        """      
        properties_set = np.zeros((len(mols_list),3))
        for idx,smiles in enumerate(mols_list):
            try:
                m = Chem.MolFromSmiles(smiles)
            except:
                print('Invalid smiles')
                
            properties_set[idx,0] = Descriptors.MolLogP(m)
            properties_set[idx,1] = Descriptors.TPSA(m)
            properties_set[idx,2] = QED.qed(m)
       

        
        robustScaler = RobustScaler()
        robustScaler.fit(properties_set)
        scaled_prop = robustScaler.transform(properties_set)
        joblib.dump(robustScaler,FLAGS.checkpoint_path+'scaler.save')
            
        return scaled_prop
    
    @staticmethod             
    def tokenize_and_padd(FLAGS,smiles,data_type,token_table,padd=True):
        """ Filters the molecules by their size, transforms SMILES strings into
            lists of tokens and padds the sequences until all sequences have 
            the same pre-defined size.
    
        Args
        ----------
            FLAGS (argparse): Implementation parameters        
            smiles (list): SMILES strings with different sizes
            data_type (string): Indicates the type of input data
            token_table (list): List of each possible symbol in the SMILES
            padd (bol): Indicates if sequences should be padded 
    
        Returns
        -------
            tokenized (list): List of SMILES with individualized tokens. The 
                              compounds are filtered by length, i.e., if it is
                              higher than the defined threshold, the compound
                              is discarded.
        """           
    
        filtered_smiles = [item for item in smiles if len(item)<=FLAGS.max_strlen-1]
        
        tokenized = []
        
        for idx,smile in enumerate(filtered_smiles):
            # smile = token_table.index('<CLS>') + smile 
            if data_type == 'decoder_in':
                smile = '[Start]' +  smile  
            elif data_type == 'decoder_out':
                smile =  smile + '[End]'

            N = len(smile)
            i = 0
            j= 0
            tokens = []
            # print(idx,smile)
            while (i < N):
                for j in range(len(token_table)):
                    symbol = token_table[j]
                    if symbol == smile[i:i + len(symbol)]:
                        tokens.append(symbol)
                        i += len(symbol)
                        break
            tokenized.append(tokens)
            if padd == True:
                while len(tokens) < FLAGS.max_strlen:
                    tokens.append(token_table[0])
                    
        return tokenized,filtered_smiles
                
            


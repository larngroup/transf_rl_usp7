# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:49:12 2021

@author: tiago
"""

# External
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.IFG import ifg
# import pylev

class Utils:
    """Data Loader class"""
    
    def __init__(self):
        """ Definition of the SMILES vocabulary """
        
        self.voc_table  = ['<PAD>','[o+]','[NH+]','[NH-]','[S+]','[O+]','[SH+]','[n-]',
                   '[N+]','[SH]','[N-]','[n+]','[S-]','[NH3+]','[s+]',
                   '[O]','[CH]','[O-]','[N]','[NH2+]','[nH]','[nH+]',
                   '@','[C@@H]','[C@H]','[C@]','[C@@]','.','I',
                   '[Cl+3]','P','H','Cl','Br','B','C','N','O','S','F','(', 
                   ')','=','#','%','0','1','2','3','4','5','6','7','8',
                   '9','+','-','c', 'n', 'o', 's','<CLS>','<SEP>','<MASK>']
        
        self.tokens_individual = ['H','Se','se','As','Si','Cl', 'Br','B', 'C', 'N', 'O', 'P', 
                  'S', 'F', 'I', '(', ')', '[', ']', '=', '#', '@', '*', '%', 
                  '0', '1', '2','3', '4', '5', '6', '7', '8', '9', '.', '/',
                  '\\', '+', '-', 'c', 'n', 'o', 's','p','<Start>','<End>','<Padd>']
# 
        
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
        
                 
    def tokenize_and_pad(self,FLAGS,smiles,token_table,padd=True,initial_token = True):
        """ Filters the molecules by their size, transforms SMILES strings into
            lists of tokens and padds the sequences until all sequences have 
            the same size.

        Args
        ----------
            FLAGS (argparse): Implementation parameters
            smiles (list): SMILES strings with different sizes
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
            if initial_token == True:
                smile = '<CLS>' + smile 
            N = len(smile)
            i = 0
            j= 0
            tokens = []
            # print(idx)
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
                
                
    @staticmethod 
    def external_diversity(set_A,set_B):
        """ Computes the Tanimoto external diversity between two sets
        of molecules

        Args
        ----------
            set_A (list): Set of molecules in the form of SMILES notation
            set_B (list): Set of molecules in the form of SMILES notation


        Returns
        -------
            td (float): Outputs a number between 0 and 1 indicating the Tanimoto
                        distance.
        """

        td = 0
        set_A = [set_A]
        fps_A = []
        for i, row in enumerate(set_A):
            try:
                mol = Chem.MolFromSmiles(row)
                fps_A.append(AllChem.GetMorganFingerprint(mol, 3))
            except:
                print('ERROR: Invalid SMILES!')
        if set_B == None:
            for ii in range(len(fps_A)):
                for xx in range(len(fps_A)):
                    ts = 1 - DataStructs.TanimotoSimilarity(fps_A[ii], fps_A[xx])
                    td += ts          
          
            td = td/len(fps_A)**2
        else:
            fps_B = []
            for j, row in enumerate(set_B):
                try:
                    mol = Chem.MolFromSmiles(row)
                    fps_B.append(AllChem.GetMorganFingerprint(mol, 3))
                except:
                    print('ERROR: Invalid SMILES!') 
            
            
            for jj in range(len(fps_A)):
                for xx in range(len(fps_B)):
                    ts = 1 - DataStructs.TanimotoSimilarity(fps_A[jj], fps_B[xx]) 
                    td += ts
            
            td = td / (len(fps_A)*len(fps_B))
        print("Tanimoto distance: " + str(td))  
        return td               

    def positional_encoding(pos, model_size):
        """ Compute positional encoding for a particular position
    
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

    def compute_dict_atom_token(self,smi):
         """ This function computes the dictionary that maps atom indexes of a
             molecule to its token indexes 
         -------
         Args:
         - smi (str): iput molecule
     
         -------
         Returns:
         - d (dictionary): Mapping between atom indexes and sequence indexes
     
         """
         d = {}
         aux_tokens = ['<Start>','<Padd>','(', ')', '[', ']', '=', '#', '@', '*', '%','0', '1',
                           '2','3', '4', '5', '6', '7', '8', '9', '.', '/','\\',
                           '+', '-']

         gap = 0 
         for i in range(0,len(smi[0])):
             symbol = smi[0][i]
             if symbol not in aux_tokens: # if it is an atom
                 d[i - gap] = i
             else:
                 gap +=1 
         return d
     
        
    def identify_fg(self,FLAGS,training_mols):
        """ Identifyes the functional groups within the training set mols
    
        Args:
            FLAGS (argparse): Implementation parameters
            training_mols (list): List of training mols
        
        Returns:
            fgs_all (list): List with indexes of atoms that belong to FG's'
        """
        fgs_all = []
        for smi in training_mols:
        
            # Transform smiles to mol
            m =  Chem.MolFromSmiles(smi) 
            
            # Split the sequence into its tokens
            processed_smi,_ = self.tokenize_and_pad(FLAGS,[smi],self.tokens_individual,padd=False,initial_token=False)
            
            # Mapping between atom number and sequence number (considering all tokens)
            dict_atom_token = self.compute_dict_atom_token(processed_smi)
            fgs = ifg.identify_functional_groups(m)
            fgs_list_tokens = [dict_atom_token[idx] for fg in fgs for idx in fg.atomIds] # allmolecule indexes
            fgs_all.append(fgs_list_tokens)
            
            
        return fgs_all
    
    
   
    

    
        
  

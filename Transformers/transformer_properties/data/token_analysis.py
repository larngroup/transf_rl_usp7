# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:01:33 2022

@author: tiago
"""
from rdkit import Chem
import matplotlib.pyplot as plt

def load_smiles_stereo(file_name):
    """ Loads the molecular dataset and filters the compounds considered
         syntactically invalid by RDKit.
    Args:
        config (json): The path of the configuration file

    Returns:
        dataset (list): The list with the training and testing SMILES 
    """
   
    all_mols = [] 
        

    file = open(file_name,"r")
    lines = file.readlines()        
    for line in lines:
        x =line.split(',')
        try:
            mol = Chem.MolFromSmiles(x[-1].strip())
            smi = Chem.MolToSmiles(mol)
            all_mols.append(smi)
        except:
            print("Invalid molecule",x)
    
    return list(set(all_mols))

def load_smiles(file_name):
    """ Loads the molecular dataset and filters the compounds considered
         syntactically invalid by RDKit.
    Args:
        opts (argparse obj): The path of the configuration file

    Returns:
        dataset (list): The list with the training and testing SMILES 
    """
   
    all_mols = [] 
        

    file = open(file_name,"r")
    lines = file.readlines()        
    for line in lines:
        x =line.split()
        try:
            mol = Chem.MolFromSmiles(x[0].strip())
            smi = Chem.MolToSmiles(mol)
            all_mols.append(smi)
        except:
            print("Invalid molecule")
    
    return all_mols

if __name__ == '__main__':
    
    voc_table = ['[Padd]','[o+]','[NH+]','[NH-]','[S+]','[O+]','[SH+]','[n-]','[2H]',
               '[N+]','[SH]','[N-]','[n+]','[S-]','[NH3+]','[s+]','[S@]','[Si]','[Au]',
               '[O]','[CH]','[O-]','[N]','[NH2+]','[nH]','[nH+]','[C@H]','[P@@]','[Mg]',
               '[C@@H]','[C@@]','[C@]','Cl','Br','C','N','O','S','F','B','I','P','.',
               '(', ')','=','#','%','0','1','2','3','4','5','6','7','8','9','\\','-','/',
               '[S@@]','[Pt]','[N@]','[As]','[Co]','[se]','[Hg]','[Ca]','[Co+]',
               '[Cr]','[N@@+]','[I+]','[N@@]','[C-]','c', 'n', 'o', 's','[Start]','[End]']
    
    
    file_name = 'smiles_dataset_3.csv'
    loaded_smiles_stereo = load_smiles_stereo(file_name)
    
    file_name_1 = 'ChEMBL_filtered.txt'
    loaded_smiles_normal = load_smiles(file_name_1)
    
    loaded_smiles = loaded_smiles_stereo + loaded_smiles_normal

    brackets_elements = []
    possible_tokens = []
    idx_start = 0 
    idx_end =  0
    d_all = {}
    
    for idx,smile in enumerate(loaded_smiles):
        N = len(smile)
        i = 0
        j= 0
        tokens = []
        print(idx,smile)
        while (i < N):
            for j in range(len(voc_table)):
                symbol = voc_table[j]
                if symbol == smile[i:i + len(symbol)]:
                    if symbol in d_all.keys():
                        d_all[symbol] = d_all[symbol] + 1
                    else:
                        d_all[symbol]=1
                    i += len(symbol)
                    break
      
            
    d_all
    
    # plt.bar(*zip(*d_all.items()))
    # plt.show()
        
        
        
    voc_table = ['H','+','[AU]']
         

       
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:48:26 2021

@author: tiago
"""

class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_dataset(FLAGS):
        """ Loads the molecular dataset and filters the compounds by the 
            sequence length
        Args:
            FLAGS (argparse): Implementation parameters

        Returns:
            smiles_a2d (list):  Raw smiles
            labels_a2d (list): Labels (pIC50)
        """
    

        smiles_a2d = []
        labels_a2d = []
        with open(FLAGS.paths['predictor_data_path']) as f:
            lines = f.readlines()
            
            for idx,line in enumerate(lines):
                
                l = line.strip().split(';')
                if idx < 1:
                    idx_smiles = l.index('Smiles')
                    idx_pic50 = l.index('pChEMBL Value')
                    idx_target_name = l.index('Target Name')
                    
                else:
                    target_name = l[idx_target_name]
                    
                    if '/' not in l[idx_smiles] and len(l[idx_smiles] ) <= FLAGS.max_str_len-2:
                    
                        try:                            
                            labels_a2d.append(float(l[idx_pic50]))
                            smiles_a2d.append(l[idx_smiles])
   
                        except:
                            pass
             
        return smiles_a2d, labels_a2d
        

    
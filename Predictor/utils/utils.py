# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:49:12 2021

@author: tiago
"""

# external
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem,Descriptors,QED
from rdkit import DataStructs
from rdkit.Chem.IFG import ifg
import tensorflow as tf

np.random.seed(52)

class Utils(object):
    """Data Loader class"""
    
    def __init__(self):
        super(Utils, self).__init__()
        
        """ Definition of the SMILES vocabulary """

        self.table = ['H','Se','se','As','Si','Cl', 'Br','B', 'C', 'N', 'O', 'P', 
          'S', 'F', 'I', '(', ')', '[', ']', '=', '#', '@', '*', '%', 
          '0', '1', '2','3', '4', '5', '6', '7', '8', '9', '.', '/',
          '\\', '+', '-', 'c', 'n', 'o', 's','p','G','E','A']
        
     
        self.voc_table  = ['[Padd]','[o+]','[NH+]','[NH-]','[S+]','[O+]','[SH+]','[n-]','[2H]',
                       '[N+]','[SH]','[N-]','[n+]','[S-]','[NH3+]','[s+]','[S@]','[Si]','[Au]',
                       '[O]','[CH]','[O-]','[N]','[NH2+]','[nH]','[nH+]','[C@H]','[P@@]','[Mg]',
                       '[C@@H]','[C@@]','[C@]','Cl','Br','C','N','O','S','F','B','I','P','.',
                       '(', ')','=','#','%','0','1','2','3','4','5','6','7','8','9','\\','-','/',
                       '[S@@]','[Pt]','[N@]','[As]','[Co]','[se]','[Hg]','[Ca]','[Co+]',
                       '[Cr]','[N@@+]','[I+]','[N@@]','[C-]','c', 'n', 'o', 's','[Start]','[End]']
        
        self.voc_table_mlm  = ['<PAD>','[o+]','[NH+]','[NH-]','[S+]','[O+]','[SH+]','[n-]',
                   '[N+]','[SH]','[N-]','[n+]','[S-]','[NH3+]','[s+]',
                   '[O]','[CH]','[O-]','[N]','[NH2+]','[nH]','[nH+]',
                   '@','[C@@H]','[C@H]','[C@]','[C@@]','.','I',
                   '[Cl+3]','P','H','Cl','Br','B','C','N','O','S','F','(', 
                   ')','=','#','%','0','1','2','3','4','5','6','7','8',
                   '9','+','-','c', 'n', 'o', 's','<CLS>','<SEP>','<MASK>']


    def smilesDict(self,token_table):
        """ Computes the dictionary that makes the correspondence between 
        each token and an given integer.

        Args
        ----------
            token_table (list): List of each possible symbol in the SMILES

        Returns
        -------
            tokenDict (dict): Dictionary mapping characters into integers
        """

        tokenDict = dict((token, i) for i, token in enumerate(token_table))
        return tokenDict
    
    
    def pad_seq(self,smiles,tokens,maxLength):
        """
        This function performs the padding of each SMILE.
        ----------
        smiles: Set of SMILES strings with different sizes;
        tokens: Set of characters;
        maxLength: Integer that specifies the maximum size of the padding    
        
        Returns
        -------
        newSmiles: Returns the padded smiles, all with the same size.
        maxLength: Integer that specifies the maximum size of the padding    
        
        """
        
        
        for i in range(0,len(smiles)):
            if len(smiles[i]) < maxLength:
                # smiles[i] = tokens[-1]*(maxLength - len(smiles[i])) + smiles[i] 
                smiles[i] =  smiles[i] + tokens[-1]*(maxLength - len(smiles[i]))  
        
        return smiles,maxLength
             
    def smiles2idx(self,FLAGS,smiles,tokenDict):
        """
        This function transforms each SMILES token to the correspondent integer,
        according the token-integer dictionary previously computed.
        ----------
        FLAGS: Implementation parameters
        smiles: Set of SMILES strings with different sizes;
        tokenDict: Dictionary that maps the characters to integers;    
        
        Returns
        -------
        newSmiles: Returns the transformed smiles, with the characters replaced by 
        the numbers. 
        """           
        newSmiles =  np.zeros((len(smiles), FLAGS.max_str_len))
        for i in range(0,len(smiles)):
            for j in range(0,len(smiles[i])):
                try:
                    newSmiles[i,j] = tokenDict[smiles[i][j]]
                except:
                    print('error in input dimensions')
        return newSmiles


    def rmse(self,y_true, y_pred):
        """
        This function implements the root mean squared error measure
        ----------
        y_true: True label   
        y_pred: Model predictions 
        Returns
        -------
        Returns the rmse metric to evaluate regressions
        """

        return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))
    
    def mse(self,y_true, y_pred):
        """
        This function implements the mean squared error measure
        ----------
        y_true: True label   
        y_pred: Model predictions 
        Returns
        -------
        Returns the mse metric to evaluate regressions
        """
        return tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1)
    
    def r_square(self,y_true, y_pred):
        """
        This function implements the coefficient of determination (R^2) measure
        ----------
        y_true: True label   
        y_pred: Model predictions 
        Returns
        -------
        Returns the R^2 metric to evaluate regressions
        """

        SS_res =  tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred)) 
        SS_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true))) 
        return (1 - SS_res/(SS_tot + tf.keras.backend.epsilon()))
    

    def ccc(self,y_true,y_pred):
        """
        This function implements the concordance correlation coefficient (ccc)
        ----------
        y_true: True label   
        y_pred: Model predictions 
        Returns
        -------
        Returns the ccc measure that is more suitable to evaluate regressions.
        """
        num = 2*tf.keras.backend.sum((y_true-tf.keras.backend.mean(y_true))*(y_pred-tf.keras.backend.mean(y_pred)))
        den = tf.keras.backend.sum(tf.keras.backend.square(y_true-tf.keras.backend.mean(y_true))) + tf.keras.backend.sum(tf.keras.backend.square(y_pred-tf.keras.backend.mean(y_pred))) + tf.keras.backend.int_shape(y_pred)[-1]*tf.keras.backend.square(tf.keras.backend.mean(y_true)-tf.keras.backend.mean(y_pred))
        return num/den
        
    def regression_plot(self,y_true,y_pred):
        """
        Function that graphs a scatter plot and the respective regression line to 
        evaluate the QSAR models.
        Parameters
        ----------
        y_true: True values from the label
        y_pred: Predictions obtained from the model
        Returns
        -------
        This function returns a scatter plot.
        """
        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred,edgecolors = 'black', alpha = 0.85)
        ax.plot([np.min(y_true), np.max(y_true)], [np.min(y_true), np.max(y_true)], 'k--', lw=4)
        ax.set_xlabel('True values')
        ax.set_ylabel('Predictions')
        plt.show()
    
                  
    def idx2smi(self,model_output,tokenDict):
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
    def tokenize_transformer(FLAGS,smiles,token_table,transformer=False):
        """ Transforms the SMILES string into a list of tokens.
    
        Args
        ----------
            FLAGS (argparse): Implementation parameters
            smiles (str): Sampled SMILES string
            token_table (list): List of each possible symbol in the SMILES
            transformer (bool): Indicates if it is necessary to padd the sequence
    
        Returns
        -------
            tokenized (str):  SMILES string with individualized tokens.
        """           
    
        tokenized = []
        
        for idx,smile in enumerate(smiles):
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
            while (len(tokens) < FLAGS.max_str_len) and transformer == False:
                tokens.append(token_table[-1])
                
                
            tokenized.append(tokens)
    
        return tokenized


    def normalize(self,FLAGS,data,train_val_labels,scaler,raw_data=False):
        # self.FLAGS,y_data_raw,None,scaler,raw_data = True
        """
        This function implements the normalization step (to avoid the 
        interference of outliers) (percentile, min-max or robust)
        ----------
        FLAGS: Implementation parameters
        data: List of label lists. It contains the y_train, y_test, and y_val (validation)
        train_val_labels:  list of labels for the training_validation set
        scaler: robustScaler object
        raw_data: bool to indicate if the normalization is for all data
        Returns
        -------
        Returns z_train, z_test, z_val (normalized targets) and data (values to 
        perform the denormalization step). 
        """
        
        if FLAGS.normalization_strategy == 'percentile':
            if raw_data == True:
               q1_train = np.percentile(data, 5)
               q3_train = np.percentile(data, 90)  
               data_normalized = (data - q1_train) / (q3_train - q1_train)
               return np.float32(data_normalized)
            else:      
               y_train = data[1]
               y_val = data[3]
               y_test = data[5]
               
               q1_train = np.percentile(train_val_labels, 5)
               q3_train = np.percentile(train_val_labels, 90)
          
               data[1] = (y_train - q1_train) / (q3_train - q1_train)
               data[3]  = (y_val - q1_train) / (q3_train - q1_train)
               data[5]  = (y_test - q1_train) / (q3_train- q1_train)
               
        elif FLAGS.normalization_strategy == 'min_max' or FLAGS.normalization_strategy == 'robust':
            if raw_data == True:
                data_normalized = scaler.transform(np.array(data).reshape(-1, 1))
                return np.float32(data_normalized)
            else:      
               y_train = data[1]
               y_val = data[3]
               y_test = data[5]
               
               data[1] = scaler.transform(np.array(y_train).reshape(-1, 1))
               data[3]  = scaler.transform(np.array(y_val).reshape(-1, 1))
               data[5]  = scaler.transform(np.array(y_test).reshape(-1, 1))

        return data

    def denormalization(FLAGS,predictions,train_val_labels,scaler):
        """
        This function implements the denormalization step.
        ----------
        FLAGS: Implementation parameters
        predictions: Output from the model
        train_val_labels: train_validation labels
        scaler: RobustScaler object
        Returns
        -------
        Returns the denormalized predictions.
        """

        if FLAGS.normalization_strategy == 'percentile':
            q1_train = np.percentile(train_val_labels, 5)
            q3_train = np.percentile(train_val_labels, 90) 
            # label = (q3_train - q1_train) * label + q1_train
            for l in range(len(predictions)):
                
                for c in range(len(predictions[0])):
                    predictions[l,c] = (q3_train - q1_train) * predictions[l,c] + q1_train
      
        return predictions
           
       
    def SMILES_2_ECFP(self,smiles, descriptor, radius=3, bit_len=4096):
        """
        This function transforms a list of SMILES strings into a list of ECFP with 
        radius 3.
        ----------
        smiles: List of SMILES strings to transform
        descriptor: string indicating the ECFP (ECFP6 or ECFP4)
        radius: integer indicating the radius to analyze for each atom 
        bit_len: integer indicating the size of the descriptor vector
        Returns
        -------
        This function return the SMILES strings transformed into a ECFP vector
        """
        
        if descriptor[-1] == '6':
            radius = 3
        elif descriptor[-1] == '4':
            radius = 2
        
        fps = np.zeros((len(smiles), bit_len))
        for i, smile in enumerate(smiles):
            mol = Chem.MolFromSmiles(smile)
            arr = np.zeros((1,))
            try:
        
                mol = MurckoScaffold.GetScaffoldForMol(mol)
         
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bit_len)
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps[i, :] = arr
            except:
                print(smile)
                fps[i, :] = [0] * bit_len
        return fps#pd.DataFrame(fps, index=(smiles if index is None else index))
               
    
    def smiles_2_rdkitFP(self,smiles):
        """
        This function transforms a list of SMILES strings into a list of 
        RDKIT fingerprint
        ----------
        smiles: List of SMILES strings to transform
        Returns
        -------
        This function return the SMILES strings transformed into RDKIT FP vectors
        """
    
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        
        rdk_fp = [Chem.RDKFingerprint(mol) for mol in mols]
        

        return np.array(rdk_fp) 
    
       
    def data_division(self,mols,labels):
        """
        This function divides data in two sets. We divide between 
        train/validation and test sets because the train/validation set will be then
        divided during CV.
        ----------
        mols: List with SMILES strings set;
        labels: List with label property set;
    
        Returns
        -------
        mols_train_val: train_validation smiles set
        mols_test: test smiles set
        labels_train_val: train_validation labels set
        labels_test: test smiles set        
        """ 
        mols_idxs = [i for i in range(0,len(labels))]
        
        msk = np.random.rand(len(labels)) < 0.9 # 10% for hold-out set
        
        idxs_train_validate = np.array(mols_idxs)[msk]
        idxs_test = np.array(mols_idxs)[~msk]
        
        mols_train_val = [m for idx,m in enumerate(mols) if idx in idxs_train_validate]
        mols_test = [m for idx,m in enumerate(mols) if idx in idxs_test]
        
        labels_train_val = np.array(labels)[idxs_train_validate]
        labels_test = np.array(labels)[idxs_test]
        
        return mols_train_val,mols_test,labels_train_val,labels_test

    def cv_split(self,mols_train_val,labels_train_val,FLAGS):
        """
        This function performs the data spliting into 5 consecutive folds. Each 
        fold is then used once as a test set while the 4 remaining folds 
        form the training set.
        ----------
        mols_train_val: list of train_validation smiles
        labels_train_val: list of train_validation labels
        FLAGS: Implementation parameters
        Returns
        -------
        data_cv: object that contains the indexes for training and testing for the 5 
              folds
        """
        cross_validation_split = KFold(n_splits=FLAGS.n_splits, shuffle=True,random_state=54)
        data_cv = list(cross_validation_split.split(mols_train_val, labels_train_val))
        return data_cv
    
        
    @staticmethod
    def pad_seq_transformer(FLAGS,smiles,data_type,tokens):
       """ Performs the padding for each SMILE. To speed up the process, the
           molecules are previously filtered by their size.

       Args
       ----------
           FLAGS (argparse): Implementation parameters
           smiles (list): SMILES strings with different sizes
           data_type (string): indicates the type of input data
           tokens (list): List possible symbols in the SMILES

       Returns
       -------
           filtered_smiles (list): List of padded smiles (with the same size)
       """        
       
       filtered_smiles = [item for item in smiles if len(item)<100]
       
       maxSmile= len(max(filtered_smiles, key=len))
       
       
       for i in range(0,len(filtered_smiles)):
           
           if data_type == 'encoder_in' or data_type == 'test_data':
               filtered_smiles[i] = filtered_smiles[i]
           elif data_type == 'decoder_in':
               filtered_smiles[i] = '<Start>' + filtered_smiles[i] 
           elif data_type == 'decoder_out':
               filtered_smiles[i] = filtered_smiles[i] + '<End>'
               
               
           if len(filtered_smiles[i]) < maxSmile:
                 filtered_smiles[i] = filtered_smiles[i] + tokens[0]*(FLAGS.max_str_len- len(filtered_smiles[i]))
      
       return filtered_smiles
   
    @staticmethod             
    def tokenize_and_pad(FLAGS,smiles,token_table,padd=True):
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

        filtered_smiles = [item for item in smiles if len(item)<=FLAGS.max_str_len-1]
        
        tokenized = []
        
        for idx,smile in enumerate(filtered_smiles):

            smile = '<CLS>' + smile 
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
                while len(tokens) < FLAGS.max_str_len:
                    tokens.append(token_table[0])
                    
        return tokenized,filtered_smiles
    
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

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    
    def identify_fg(training_mols):
        """ Identifyes the functional groups of input set of molecules
    
        Args:
            training_mols (list): List of training mols
        
        Returns:
            fgs_all (list): List with indexes of atoms that belong to FG's'
        """
        fgs_all = []
        for smi in training_mols:
            # print(smi)
            m =  Chem.MolFromSmiles(smi)
            fgs = ifg.identify_functional_groups(m)
            fgs_list = [idx for fg in fgs for idx in fg.atomIds]
            fgs_all.append(fgs_list)
            
            
        return fgs_all
    
    
    def compute_properties(mols_list,scaler):
        """ Computes and normalize the drug-like properties: logP, TPSA, and QED
        Args
        ----------
            mols_list (list): Set of SMILES
            scaler: robustscaler object
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
            
        scaled_prop = scaler.transform(properties_set)
        return scaled_prop








       
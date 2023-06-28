# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:49:12 2021

@author: tiago
"""
# external
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from rdkit import Chem
from rdkit.Chem.IFG import ifg
from rdkit.Chem import AllChem,Descriptors,QED
from rdkit import DataStructs
from utils.sascorer_calculator import SAscore

class Utils:
    """Data Loader class"""
    
    def __init__(self):
        """ Definition of the SMILES vocabulary """
        
        self.tokens = ['H','Se','se','As','Si','Cl', 'Br','B', 'C', 'N', 'O', 'P', 
                  'S', 'F', 'I', '(', ')', '[', ']', '=', '#', '@', '*', '%', 
                  '0', '1', '2','3', '4', '5', '6', '7', '8', '9', '.', '/',
                  '\\', '+', '-', 'c', 'n', 'o', 's','p','<Start>','<End>','<Padd>']
        
        
        self.voc_table = ['[Padd]','[o+]','[NH+]','[NH-]','[S+]','[O+]','[SH+]','[n-]','[2H]',
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
    
    @staticmethod
    def smilesDict(token_table):
        """ Computes the dictionary that makes the correspondence between 
        each token and the respective integer.

        Args
        ----------
            token_table (list): List of each possible SMILES tokens

        Returns
        -------
            tokenDict (dict): Dictionary mapping characters into integers
        """

        tokenDict = dict((token, i) for i, token in enumerate(token_table))
        return tokenDict
    
    @staticmethod
    def pad_seq(smiles,tokens,FLAGS):
        """ Performs the padding for each sampled SMILES.

        Args
        ----------
            smiles (str): SMILES string
            tokens (list): List of each possible symbol in the SMILES
            FLAGS (argparse): Implementation parameters

        Returns
        -------
            smiles (str): Padded SMILES string
        """
        
        if isinstance(smiles, str) == True:
            smiles = [smiles]
            
        maxLength = FLAGS.max_str_len
    
        for i in range(0,len(smiles)):
            smiles[i] = '<Start>' + smiles[i] + '<End>'
            if len(smiles[i]) < maxLength:
                smiles[i] = smiles[i] + tokens[-1]*(maxLength - len(smiles[i]))

        return smiles
    

    @staticmethod            
    def smiles2idx(smiles,tokenDict):
        """ Transforms each token into the respective integer.

        Args
        ----------
            smiles (str): Sampled SMILES string 
            tokenDict (dict): Dictionary mapping characters to integers 

        Returns
        -------
            newSmiles (str): Transformed smiles, with the characters 
                              replaced by the respective integers. 
        """   
        max_len = max([len(i) for i in smiles])
        newSmiles =  np.zeros((len(smiles), max_len))
        for i in range(0,len(smiles)):
            for j in range(0,len(smiles[i])):
                try:
                    newSmiles[i,j] = tokenDict[smiles[i][j]]
                except: 
                    print('ERROR: wrong input dimensions')
        return newSmiles
        
    @staticmethod             
    def tokenize(FLAGS,smiles,token_table,transformer=False):
        """ Transforms the SMILES string into a list of tokens.

        Args
        ----------
            FLAGS (argparse): Implementation parameters
            smiles (str): Sampled SMILES string
            token_table (list): List of each possible symbol in the SMILES
            transformer (bool): Indicates if it is necessary to padd the sequence

        Returns
        -------
            tokenized (list):  SMILES string with individualized tokens.
        """           
        tokenized = []
        
        for idx,smile in enumerate(smiles):
            N = len(smile)
            i = 0
            j= 0
            tokens = []
            while (i < N):
                for j in range(len(token_table)):
                    symbol = token_table[j]
                    if symbol == smile[i:i + len(symbol)]:
                        tokens.append(symbol)
                        i += len(symbol)
                        break
            while (len(tokens) < FLAGS.max_str_len) and transformer == False:
                tokens.append(token_table[0])
                
            tokenized.append(tokens)
    
        return tokenized

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
            seq,score = model_output[i]
            smi = []
            for j in range(0,len(seq)):
                token = key_list[val_list.index(seq[j])]
                
                if token == '<End>' or token == '<Padd>':
                    break 
                
                smi.append(token)
            
            print(''.join(smi))
            reconstructed_smiles.append(smi)
                
        return reconstructed_smiles
    
    @staticmethod
    def remove_padding(trajectory):  
        """ Function that removes the padding characters from the sampled 
            molecule

        Args
        ----------
            trajectory (str): Padded generated molecule

        Returns
        -------
            trajectory (str): SMILES string without the padding characters
        """     
        
        if '<Padd>' in trajectory:
        
            firstA = trajectory.find('<Padd>')
            trajectory = trajectory[0:firstA]
        return trajectory
    
    @staticmethod
    def read_csv(FLAGS):
        """ This function loads the labels of the biological affinity dataset
        Args
        ----------
            FLAGS (argparse): Implementation parameters
        
        Returns
        -------
            raw_labels (list): Returns the pIC50 values 
        """
        
        labels_raw = []
        with open(FLAGS.models_path['predictor_data_path']) as f:
            lines = f.readlines()
        
            for idx,line in enumerate(lines):
                
                l = line.strip().split(';')
            
                if idx < 1:
                    idx_smiles = l.index('Smiles')
                    idx_pic50 = l.index('pChEMBL Value')
                    idx_target_name = l.index('Target Name')

                else:
          
                    if '/' not in l[idx_smiles] and len(l[idx_smiles] ) <= FLAGS.max_str_len-2:
                    
                        try:                            
                            labels_raw.append(float(l[idx_pic50]))

   
                        except:
                            pass   
                                                       
        return labels_raw
    
    
    @staticmethod 
    def get_reward_MO(predictor_obj,smile,embeddings,memory_smiles,FLAGS):
        """ This function uses the predictor and the sampled SMILES string to 
        predict the numerical rewards regarding the evaluated properties.

        Args
        ----------
            predictor_obj (object): Predictive model that accepts a trajectory
                                     and returns the respective pIC50 
            smile (str): SMILES string of the sampled molecule
            embeddings (array): contextual embeddings computed by the 
                                Transformer encoder
            memory_smiles (list): List of the last 30 generated molecules
            FLAGS (argparse): Implementation parameters

        Returns
        -------
            rewards (list): Outputs the list of reward values for the evaluated 
                            properties
            sas (float): Synthetic accessibility score
            pred (float): Predicted pIC50 for the USP7 target
        """
    
        # pIC50 for USP7
        rewards = []        

        pred = predictor_obj.predict(embeddings)[0][0]
        
        if 'mlm' in FLAGS.option:
            reward_affinity = np.exp(pred/3-2.4) 
        elif 'standard' in FLAGS.option: 
            reward_affinity = np.exp(pred/3-1) #standard
        
        if pred < 4:
            rewards.append(0)
        else:
            rewards.append(reward_affinity)
        
        # SA score
        list_mol_sas = []
        list_mol_sas.append(Chem.MolFromSmiles(smile))
        sas_list = SAscore(list_mol_sas)
        sas = sas_list[0]
        rew_sas = np.exp(-sas/4 + 1.6)

        rewards.append(rew_sas)
       
        diversity = 1
        if len(memory_smiles)> 30:
            diversity = Utils.external_diversity(smile,memory_smiles)
            
        if diversity < 0.75:
            rew_div = 0.95
            print("\Alert: Similar compounds")
        else:
            rew_div = 1
            
        rewards.append(rew_div)
        return rewards,sas,pred

    @staticmethod
    def scalarization(rewards,weights,pred_range_pic50,pred_range_sas,scalarMode='linear'):
        """ Transforms the vector of two rewards into a unique reward value.
        
        Args
        ----------
            rewards (list): List of rewards of each property;
            weights (list): List containing the weights indicating the importance 
                            of the each property;
            pred_range_pic50 (list): List with the max and min prediction values
                                of the reward to for the pic50 to normalize the
                                obtained reward (between 0 and 1).
            pred_range_sas (list): List with the max and min prediction values
                                of the reward to for the SAS to normalize the
                                obtained reward (between 0 and 1).
            scalarMode (str): String indicating the scalarization type;

        Returns
        -------
            rescaled_reward (float): Scalarized reward
        """
        

        w_affinity = weights[0]
        w_sas = weights[1]

        rew_affinity = rewards[0]
        rew_sas = rewards[1]
        rew_div = rewards[2]
        
        max_affinity = pred_range_pic50[1]
        min_affinity = pred_range_pic50[0]
    
        max_sas = pred_range_sas[1]
        min_sas = pred_range_sas[0]
    
        rescaled_rew_sas = (rew_sas - min_sas )/(max_sas - min_sas)
        
        if rescaled_rew_sas < 0:
            rescaled_rew_sas = 0
        elif rescaled_rew_sas > 1:
            rescaled_rew_sas = 1
    
        rescaled_rew_affinity = (rew_affinity  - min_affinity)/(max_affinity -min_affinity)
        
        if rescaled_rew_affinity < 0:
            rescaled_rew_affinity = 0
        elif rescaled_rew_affinity > 1:
            rescaled_rew_affinity = 1
        
        if scalarMode == 'linear':
            lws_reward = (w_affinity*rescaled_rew_affinity + w_sas*rescaled_rew_sas)*2.0*rew_div

            return lws_reward,rescaled_rew_sas,rescaled_rew_affinity
    
        elif scalarMode == 'chebyshev':

            dist_affinity = abs(rescaled_rew_affinity-1)*w_affinity
            dist_sas = abs(rescaled_rew_sas-1)*w_sas
            print("distance a2d: " + str(dist_affinity))
            print("distance sas: " + str(dist_sas))
            
            if dist_affinity > dist_sas:
                return rescaled_rew_affinity*3
            else:
                return rescaled_rew_sas*3
    
    @staticmethod 
    def external_diversity(set_A,set_B=None):
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
        if isinstance(set_A, list) == False:
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
        return td
    
    def smiles2mol(self,smiles_list):
        """
        Function that converts a list of SMILES strings to a list of RDKit molecules 
        Parameters
        
        Args
        ----------
        smiles_list (list): Input SMILES strings
        
        Returns
        -------
        mol_list (list): Molecular instances 
        """
        
        mol_list = []
        if isinstance(smiles_list,str):
            mol = Chem.MolFromSmiles(smiles_list, sanitize=True)
            mol_list.append(mol)
        else:
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi, sanitize=True)
                mol_list.append(mol)
        return mol_list
                
    def denormalization(predictions,data,scaler):
        """
        This function implements the denormalization step.
        
        Args
        ----------
        predictions (array): Output from the Predictor model
        data (list): q3 and q1 values to perform the denormalization
        scaler (robustscaler): RobustScaler object
        
        Returns
        -------
        predictions (array): denormalized predictions.
        """
        
        predictions_new = scaler.inverse_transform(predictions)
        
        # q1_train = np.percentile(data, 5)
        # q3_train = np.percentile(data, 90)        
        # for l in range(len(predictions)):
        #     for c in range(len(predictions[0])):
        #         predictions[l,c] = (q3_train - q1_train) * predictions[l,c] + q1_train
        
        return predictions_new


    @staticmethod 
    def padding_one_hot(smiles,tokens): 
        """ Performs the padding of the sampled molecule represented in OHE
        Args
        ----------
            smiles (str): Sampled molecule in the form of OHE;
            tokens (list): List of tokens that can constitute the molecules   

        Returns
        -------
            smiles (str): Padded sequence
        """

        smiles = smiles[0,:,:]
        maxlen = 65
        idx = tokens.index('A')
        padding_vector = np.zeros((1,43))
        padding_vector[0,idx] = 1
    
        while len(smiles) < maxlen:
            smiles = np.vstack([smiles,padding_vector])
                
        return smiles
    
   
    
    def plot_training_progress(training_rewards,losses_generator,training_pic50,training_sas, rewards_pic50, rewards_sas,scaled_rewards_pic50,scaled_rewards_sas):
        """ Plots the evolution of the rewards and loss throughout the 
        training process.
        Args
        ----------
            training_rewards (list): List of the combined rewards for each 
                                     sampled batch of molecules;
            losses_generator (list): List of the computed losses throughout the 
                                     training process;
            training_pic50 (list): List of the pIC50 values for each sampled 
                                     batch of molecules;
            training_sas (list): List of the SAS values for each sampled 
                                     batch of molecules;
            rewards_pic50 (list): List of the rewards for the pIC50 property
                                  for each sampled batch of molecules;
            rewards_sas (list): List of the rewards for the SAS property
                                  for each sampled batch of molecules;
            scaled_rewards_pic50 (list): List of scaled rewards for the pIC50
            scaled_rewards_sas (list): List of scaled rewards for SAS

        Returns
        -------
            Plot
        """

        plt.plot(training_rewards)
        plt.xlabel('Training iterations')
        plt.ylabel('Average rewards')
        plt.show(block=False)
        plt.show()
        
        plt.plot(training_pic50)
        plt.xlabel('Training iterations')
        plt.ylabel('Average pIC50')
        plt.show(block=False)
        # plt.show()

        plt.plot(training_sas)
        plt.xlabel('Training iterations')
        plt.ylabel('Average SA score')
        plt.show()
        
        plt.plot(losses_generator)
        plt.xlabel('Training iterations')
        plt.ylabel('Average losses PGA')
        plt.show()
        
        plt.plot(rewards_pic50)
        plt.xlabel('Training iterations')
        plt.ylabel('Average rewards pic50')
        plt.show()
        
        plt.plot(rewards_sas)
        plt.xlabel('Training iterations')
        plt.ylabel('Average rewards sas')
        plt.show()
        
        plt.plot(scaled_rewards_pic50)
        plt.xlabel('Training iterations')
        plt.ylabel('Scaled rewards pic50')
        plt.show()
        
        plt.plot(scaled_rewards_sas)
        plt.xlabel('Training iterations')
        plt.ylabel('Scaled rewards sas')
        plt.show()

                
    def moving_average(previous_values, new_value, ma_window_size=10): 
        """
        This function performs a simple moving average between the previous 9 and the
        last one reward value obtained.
        
        Args
        ----------
        previous_values (list): list with previous values 
        new_value (float): new value to append, to compute the average with the 
                          last ten elements
        
        Returns
        -------
        values_ma (float): Outputs the average of the last 10 elements 
        """
        value_ma = np.sum(previous_values[-(ma_window_size-1):]) + new_value
        value_ma = value_ma/(len(previous_values[-(ma_window_size-1):]) + 1)
        return value_ma
                
        
    def compute_thresh(rewards,thresh_set):
        """
        Function that computes the thresholds to choose which Generator will be
        used during the generation step, based on the evolution of the reward values.
       
        Args
        ----------
        rewards (list): Last 3 reward values obtained from the RL method
        thresh_set (int): Integer that indicates the threshold set to be used
        
        Returns
        -------
        threshold(float): This function returns a threshold depending on the 
                          recent evolution of the reward. If the reward is 
                          increasing the threshold will be lower and vice versa.
        """
        reward_t_2 = rewards[0]
        reward_t_1 = rewards[1]
        reward_t = rewards[2]
        q_t_1 = reward_t_2/reward_t_1
        q_t = reward_t_1/reward_t
        
        if thresh_set == 1:
            thresholds_set = [0.15,0.3,0.2]
        elif thresh_set == 2:
            thresholds_set = [0.05,0.2,0.1] 
        #        thresholds_set = [0,0,0] 
        
        threshold = 0
        if q_t_1 < 1 and q_t < 1:
            threshold = thresholds_set[0]
        elif q_t_1 > 1 and q_t > 1:
            threshold = thresholds_set[1]
        else:
            threshold = thresholds_set[2]
        
        return threshold

    def serialize_model(generator_biased,FLAGS):
        """
        Serializes trained model to disk

        Args
        ----------
        generator_biased (Generator): Object that contains the Generator model
        FLAGS (argparse): Implementation parameters

        Returns
        -------
        None.

        """
        generator_biased.model.save(FLAGS.checkpoint_path+'biased_generator.hdf5')
        

        
    def canonical_smiles(self,smiles,sanitize=True, throw_warning=False):
        """
        Takes list of generated SMILES strings and returns the list of valid SMILES.
        
        Args
        ----------
        smiles: List of SMILES strings to validate
        sanitize: bool (default True)
            parameter specifying whether to sanitize SMILES or not.
                For definition of sanitized SMILES check
                http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#SanitizeMol
        throw_warning: bool (default False)
            parameter specifying whether warnings will be thrown if a SMILES is
            invalid
            
        Returns
        -------
        new_smiles: list of valid SMILES (if it is valid and has <75 characters)
        and NaNs if SMILES string is invalid
        valid: number of valid smiles, regardless of the its size
            
        """
        new_smiles = []
        valid = 0
        for sm in smiles:
            try:
                mol = Chem.MolFromSmiles(sm, sanitize=sanitize)
                s = Chem.MolToSmiles(mol)
                valid+=1
                if len(s) <= 75:
                    new_smiles.append(s)   

            except:
                print('Invalid')

        return new_smiles,valid

    def plot_hist(self,prediction, n_to_generate,property_identifier):
        """
        Function that plots the predictions's distribution of the generated SMILES 
        strings
        
        Args
        ----------
        prediction: list with the desired property predictions.
        n_to_generate: number of generated SMILES.
        property_identifier: String identifying the property 
        
        Returns
        ----------
        Plot property distribution
        """
        prediction = np.array(prediction,dtype='float64').reshape((-1,))
        x_label = ''
        plot_title = '' 
        

        if property_identifier == "usp7":

            print("\nMax of pIC50: ", round(np.max(prediction),4))
            print("Mean of pIC50: ", round(np.mean(prediction),4))
            print("Std of pIC50: ", round(np.std(prediction),4))
            print("Min of pIC50: ", round(np.min(prediction),4))
            x_label = "Predicted pIC50"
            plot_title = "Distribution of predicted pIC50 for generated molecules"
            
        elif property_identifier == "sas":
            print("\nMax SA score: ", round(np.max(prediction),4))
            print("Mean SA score: ", round(np.mean(prediction),4))
            print("Std SA score: ", round(np.std(prediction),4))
            print("Min SA score: ", round(np.min(prediction),4))
            x_label = "Calculated SA score"
            plot_title = "Distribution of SA score for generated molecules"
        elif property_identifier == "qed":
            print("Max QED: ", np.max(prediction))
            print("Mean QED: ", np.mean(prediction))
            print("Min QED: ", np.min(prediction))
            x_label = "Calculated QED"
            plot_title = "Distribution of QED for generated molecules"  
            
        elif property_identifier == "logp":
            percentage_in_threshold = np.sum((prediction >= 0.0) & 
                                         (prediction <= 5.0))/len(prediction)
            print("Percentage of predictions within drug-like region:", percentage_in_threshold)
            print("Average of log_P: ", np.mean(prediction))
            print("Median of log_P: ", np.median(prediction))
            plt.axvline(x=0.0)
            plt.axvline(x=5.0)
            x_label = "Predicted LogP"
            plot_title = "Distribution of predicted LogP for generated molecules"
            
        sns.axes_style("darkgrid")
        ax = sns.kdeplot(prediction, shade=True,color = 'g')
        ax.set(xlabel=x_label,
               title=plot_title)
        plt.show()
       
    
    def pad_seq_pred(smiles,tokens,FLAGS):
        """ Performs the padding for each sampled SMILES.

        Args
        ----------
            smiles (str): SMILES string
            tokens (list): List of each possible symbol in the SMILES
            FLAGS (argparse): Implementation parameters

        Returns
        -------
            smiles (str): Padded SMILES string
        """
        
        if isinstance(smiles, str) == True:
            smiles = [smiles]
            
        maxLength = FLAGS.max_str_len
    
        for i in range(0,len(smiles)):
            smiles[i] = 'G' + smiles[i] + 'E'
            if len(smiles[i]) < maxLength:
                smiles[i] = smiles[i] + tokens[-1]*(maxLength - len(smiles[i]))
        return smiles
    
    def tokenize_pred(FLAGS,smiles,token_table):
        """ Transforms the SMILES string into a list of tokens.

        Args
        ----------
            FLAGS (argparse): Implementation parameters
            smiles (str): Sampled SMILES string
            token_table (list): List of each possible symbol in the SMILES

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
            while (i < N):
                for j in range(len(token_table)):
                    symbol = token_table[j]
                    if symbol == smile[i:i + len(symbol)]:
                        tokens.append(symbol)
                        i += len(symbol)
                        break
            while (len(tokens) < FLAGS.max_str_len):
                tokens.append(token_table[-1])
                
            tokenized.append(tokens)
    
        return tokenized
    
    def plot_hist_both(prediction_usp7_unb,prediction_usp7_b,prediction_sas_unb,prediction_sas_b,FLAGS):
        """
        Function that plots the predictions's distribution of the generated SMILES 
        strings, obtained by the unbiased and biased generators.
        Parameters
        ----------
        prediction_usp7_unb: list with the usp7 affinity predictions of unbiased 
                        generator.
        prediction_usp7_b: list with the usp7 affinity predictions of biased generator.
        prediction_sas_unb: list with the sas predictions of unbiased 
                        generator.
        prediction_sas_b: list with the sas predictions of biased generator.
            FLAGS (argparse): Implementation parameters

        Returns
        ----------
        This functions returns the difference between the averages of the predicted
        properties and the % of valid SMILES
        """
        n_to_generate = FLAGS.mols_to_generate
        prediction_usp7_unb = np.array(prediction_usp7_unb)
        prediction_usp7_b= np.array(prediction_usp7_b)
        
        prediction_sas_unb = np.array(prediction_sas_unb)
        prediction_sas_b= np.array(prediction_sas_b)
        
        print("\nProportion of valid SMILES (UNB,B):", len(prediction_usp7_unb)/n_to_generate,len(prediction_usp7_b)/n_to_generate)
  
        legend_usp7_unb = 'Unbiased pIC50 values'
        legend_usp7_b = 'Biased pIC50 values'
        print("\n\nMax of pIC50: (UNB,B)", np.max(prediction_usp7_unb),np.max(prediction_usp7_b))
        print("Mean of pIC50: (UNB,B)", np.mean(prediction_usp7_unb),np.mean(prediction_usp7_b))
        print("Min of pIC50: (UNB,B)", np.min(prediction_usp7_unb),np.min(prediction_usp7_b))
    
        label_usp7 = 'Predicted pIC50'
        plot_title_usp7 = 'Distribution of predicted pIC50 for generated molecules'
            
  
        legend_sas_unb = 'Unbiased'
        legend_sas_b = 'Biased'
        print("\n\nMax of SA score: (UNB,B)", np.max(prediction_sas_unb),np.max(prediction_sas_b))
        print("Mean of SA score: (UNB,B)", np.mean(prediction_sas_unb),np.mean(prediction_sas_b))
        print("Min of SA score: (UNB,B)", np.min(prediction_sas_unb),np.min(prediction_sas_b))
    
        label_sas = 'Predicted SA score'
        plot_title_sas = 'Distribution of SA score values for generated molecules'  
     
        sns.axes_style("darkgrid")
        v1_usp7 = pd.Series(prediction_usp7_unb, name=legend_usp7_unb)
        v2_usp7 = pd.Series(prediction_usp7_b, name=legend_usp7_b)
               
        ax = sns.kdeplot(v1_usp7, shade=True,color='g',label=legend_usp7_unb)
        sns.kdeplot(v2_usp7, shade=True,color='r',label =legend_usp7_b )
    
        ax.set(xlabel=label_usp7, 
               title=plot_title_usp7)

        plt.show()
        
        v1_sas = pd.Series(prediction_sas_unb, name=legend_sas_unb)
        v2_sas = pd.Series(prediction_sas_b, name=legend_sas_b)
               
        ax = sns.kdeplot(v1_sas, shade=True,color='g',label=legend_sas_unb)
        sns.kdeplot(v2_sas, shade=True,color='r',label =legend_sas_b )
    
        ax.set(xlabel=label_sas, 
               title=plot_title_sas)
        plt.show()
        
    
    def rmse(y_true, y_pred):
        """
        This function implements the root mean squared error measure
        ----------
        y_true: True label   
        y_pred: Model predictions 
        Returns
        -------
        Returns the rmse metric to evaluate regressions
        """
        from keras import backend
        return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

    def mse(y_true, y_pred):
        """
        This function implements the mean squared error measure
        ----------
        y_true: True label   
        y_pred: Model predictions 
        Returns
        -------
        Returns the mse metric to evaluate regressions
        """
        from keras import backend
        return backend.mean(backend.square(y_pred - y_true), axis=-1)

    def r_square(y_true, y_pred):
        """
        This function implements the coefficient of determination (R^2) measure
        ----------
        y_true: True label   
        y_pred: Model predictions 
        Returns
        -------
        Returns the R^2 metric to evaluate regressions
        """
        from keras import backend as K
        SS_res =  K.sum(K.square(y_true - y_pred)) 
        SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
        return (1 - SS_res/(SS_tot + K.epsilon()))

    def ccc(y_true,y_pred):
        """
        This function implements the concordance correlation coefficient (ccc)
        ----------
        y_true: True label   
        y_pred: Model predictions 
        Returns
        -------
        Returns the ccc measure that is more suitable to evaluate regressions.
        """
        from keras import backend as K
        num = 2*K.sum((y_true-K.mean(y_true))*(y_pred-K.mean(y_pred)))
        den = K.sum(K.square(y_true-K.mean(y_true))) + K.sum(K.square(y_pred-K.mean(y_pred))) + K.int_shape(y_pred)[-1]*K.square(K.mean(y_true)-K.mean(y_pred))
        return num/den

    def update_weights(scaled_rewards_pic50,scaled_rewards_sas,weights):
        """ Updates the preference weights for each property
        Args
        ----------
            scaled_rewards_pic50 (list): Scaled rewards of pIC50 
            scaled_rewards_sas (list): Scaled rewards of SAS
            weights (list): Preference weights of the previous batch

        Returns
        -------
            weights (list): Updated weights
        """
        mean_pic50_previous = np.mean(scaled_rewards_pic50[-5:-1])
        mean_pic50_current = scaled_rewards_pic50[-1:][0]
        
        mean_sas_previous = np.mean(scaled_rewards_sas[-5:-1])
        mean_sas_current = scaled_rewards_sas[-1:][0]
        
        growth_pic50 = (mean_pic50_current - mean_pic50_previous)/mean_pic50_previous
        
        growth_sas = (mean_sas_current - mean_sas_previous)/mean_sas_previous
        
        if mean_pic50_current*weights[0] > mean_sas_current*weights[1] and growth_sas < 0.01:
            weights[0] = weights[0] - 0.05
            weights[1] = weights[1] + 0.05
        elif mean_pic50_current*weights[0] < mean_sas_current*weights[1] and growth_pic50 < 0.01:
            weights[0] = weights[0] + 0.05
            weights[1] = weights[1] - 0.05
            
        print('Preference Weights (pIC50,SAS): ',weights) 
        
        return weights
    
    def check(list1,list2, val1,val2):
        """ Verifies if a point with values (val1,val2) is non-dominated
        Args
        ----------
            list1 (list): pIC50 values
            list2 (list): SAS values
            val1 (list): point pIC50
            val2 (list): point SAS
        Returns
        -------
            (bool): False if it is non-dominated and vice-versa
        """      
        # traverse in the list
        for idx in range(0,len(list1)):

            if list1[idx] > val1 and list2[idx] < val2:
                return True 
        return False
   
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
        """ Compute softmax activation for the vector x
    
        Args:
            x (array): vector of floating values
        
        Returns:
            Vector of the same size of x with their values softamax transformed
        """
  
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
           
    def generate_smiles(self,generator,predictor_usp7,transformer,tokenDict,FLAGS):
        """ Generation of new SMILES and properties prediction  
    
        Args:
            generator (object): Generator object
            predictor_usp7 (object): Predictor object
            transformer (object): Transformer object
            tokenDict (dict): Dictionary mapping characters into integers
            FLAGS (argparse): Implementation parameters       
            
        Returns:
            generated_smiles (DataFrame): It contains the SMILES, pic50, sas and
                                          mol object of the valid molecules
            
        """
        
        generated_smiles = pd.DataFrame()
        
        _,trajectories = generator.generate(FLAGS.mols_to_generate)
        
        smiles= [smile[7:-5] for smile in trajectories]
        
        smiles_sanitized,valid = self.canonical_smiles(smiles, sanitize=True, throw_warning=False) # validar 
        
        generated_smiles['smiles'] = smiles_sanitized

        contextual_embeddings = transformer.predict_step(smiles_sanitized)
        
        #prediction usp7 affinity
        prediction_usp7 = predictor_usp7.predict(contextual_embeddings)

        # prediction SA score
        mols_list = self.smiles2mol(smiles_sanitized)

        prediction_sas = SAscore(mols_list)
        
        try:
            generated_smiles['pIC50 usp7'] = prediction_usp7
            generated_smiles['sas'] = prediction_sas
            generated_smiles['mols_obj'] = mols_list
        except:
            print('ERROR: invalid molecular evaluation')

        unique_smiles = list(np.unique(smiles_sanitized))
        percentage_unq = (len(unique_smiles)/len(smiles_sanitized))*100
        print('% Unique: ', round(percentage_unq,4))
        
        return generated_smiles
    
    @staticmethod             
    def tokenize_and_pad(FLAGS,smiles,token_table,model,padd=True):
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
            if model == 'mlm':
                smile = '<CLS>' + smile 
            elif model == 'encoder_in' or model == 'test_data':
                smile = smile 
            elif model == 'decoder_in':
                smile = '[Start]' + smile
            elif model == 'decoder_out':
                smile = smile + '[End]'

            N = len(smile)
            i = 0
            j= 0
            tokens = []

            while (i < N):
                for j in range(len(token_table)):
                    symbol = token_table[j]
                    if symbol == smile[i:i + len(symbol)]:
                        tokens.append(symbol)
                        i += len(symbol)
                        break
            

            if padd == True:
                while len(tokens) < FLAGS.max_str_len:
                    tokens.append(token_table[0])
            tokenized.append(tokens)
                    
        return tokenized,filtered_smiles
    
    
    def plot_fg(dict_fg,topk,data_identifier):

        sorted_fg = dict(sorted(dict_fg.items(), key=lambda item: item[1],reverse = True))
        values = list(sorted_fg.keys())[:topk]
        keys = list(sorted_fg.values())[:topk]
                
        plt.barh(values, keys,color='#86bf91')
 
        # setting label of y-axis
        plt.ylabel("Functional Groups")
         
        # setting label of x-axis
        plt.xlabel("Relative frequency")
        
        if data_identifier == 'generated':
            title = "Most frequent FG's of generated mols"    
        elif data_identifier == 'known':
            title = "Most frequent FG's of known inhibitors"   
        
        plt.title(title)
        plt.show()
        
    def compute_dict_atom_token(smi):
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
            
            
        
    def index_remove_tokens(smiles_tokens): #smiles_molecule tokens
        """ This function finds the indexes that are not atoms in molecules.
        -------
        Args:
        - smiles_token (list of lists) -> list of smiles tokens
    
        -------
        Returns:
        - remove_ind (list of lists of int) -> list of indexes that are not atoms
    
        """
        remove = ["#", "]", "(", "[", "=", ")"]
        remove_ind = []
        for index_smile in range (len(smiles_tokens)):
            if smiles_tokens[index_smile] in remove:
                remove_ind.append(index_smile)
            elif smiles_tokens[index_smile].isnumeric():
                remove_ind.append(index_smile)
    
        return remove_ind
    
    def compute_properties(mols_list,scaler):
        """ Computes and normalize the drug-like properties: logP, TPSA, and QED
        Args
        ----------
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
            
        scaled_prop = scaler.transform(properties_set)
        return scaled_prop
    
    def identify_fg(training_mols):
        """ Identifyes the functional groups within the training set mols
    
        Args:
            training_mols (list): List of training mols
        
        Returns:
            fgs_all (list): List with indexes of atoms that belong to FG's'
        """
        fgs_all = []
        for smi in training_mols:
            m =  Chem.MolFromSmiles(smi)
            fgs = ifg.identify_functional_groups(m)
            fgs_list = [idx for fg in fgs for idx in fg.atomIds]
            fgs_all.append(fgs_list)
            
            
        return fgs_all
    
    
    def moving_average_new(raw_values,window_size):
        """
        Parameters
        ----------
        raw_values : (list) set of attention scores
        window_size : (int) window to analyze

        Returns
        -------
        moving_averages : applies moving average filter to the attention scores
        """

        arr  = list(raw_values)
       
        i = 0
        # Initialize an empty list to store moving averages
        moving_averages = []
          
        # Loop through the array to consider
        # every window of size 3
        while i < len(arr) - window_size + 1:
            
            # Store elements from i to i+window_size
            # in list to get the current window
            window = arr[i : i + window_size]
          
            # Calculate the average of current window
            window_average = round(sum(window) / window_size, 2)
              
            # Store the average of current
            # window in moving average list
            moving_averages.append(window_average)
              
            # Shift window to right by one position
            i += 1
        
        return moving_averages

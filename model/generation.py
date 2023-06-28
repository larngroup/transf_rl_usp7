    # -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:49:39 2021

@author: tiago
"""

# internal
from .base_model import BaseModel
from dataloader.dataloader import DataLoader
from utils.utils import Utils 
from model.generator import Generator  
from model.transformer import Transformer
from model.transformer_mlm import Transformer_mlm

# external
import tensorflow as tf
import numpy as np
from rdkit import Chem
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from rdkit.Chem.Draw import DrawingOptions
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import QED
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.IFG import ifg
import warnings

warnings.filterwarnings('ignore')
tf.config.run_functions_eagerly(True)

class generation_process(BaseModel):
    """Conditional Generation Object"""
    
    def __init__(self, FLAGS):
        super().__init__(FLAGS)
        
        self.FLAGS = FLAGS
        self.token_table = Utils().tokens # Load the table of possible tokens
        self.tokenDict = Utils.smilesDict(self.token_table)
        
        # Load pre-trained models
        self.generator_unbiased = DataLoader().load_generator(self.FLAGS,'unbiased')
        
        if FLAGS.option == 'mlm' or FLAGS.option == 'mlm_exp1' or FLAGS.option == 'mlm_exp2':
            self.predictor_usp7 = DataLoader().load_predictor(self.FLAGS)
            self.transformer_model = Transformer_mlm(self.FLAGS)
        elif FLAGS.option == 'standard' or FLAGS.option == 'standard_exp1' or FLAGS.option == 'standard_exp2':
            self.predictor_usp7 = DataLoader().load_predictor(self.FLAGS)
            self.transformer_model = Transformer(self.FLAGS)
        

    def custom_loss(self,aux_matrix):
        """ Computes the loss function to update the generator through the 
        policy gradient algorithm

        Args
        ----------
            aux_matrix (array): Auxiliary matrix to adjust the dimensions and 
                                padding when performing computations.

        Returns
        -------
            lossfunction (float): Value of the loss 
        """
        def lossfunction(y_true,y_pred):
            y_pred = tf.cast(y_pred, dtype='float64')
            y_true = tf.cast(y_true, dtype='float64')
            y_true = tf.reshape(y_true, (-1,))
           
            return (-1/self.FLAGS.batch_size)*K.sum(y_true*K.log(tf.math.reduce_sum(tf.multiply(tf.add(y_pred,10**-9),aux_matrix),[1,2])))
       
        return lossfunction

    def get_unbiased_model(self,aux_array,trainable = True):
        """ Builds the pre-trained Generator to be optimized with Reinforcement
            Learning

        Args
        ----------
            aux_matrix (array): Auxiliary matrix to adjust the loss function
                                dimensions when performing computations.

        Returns
        -------
            generator_biased (model): Generator model to be updated during with
                                      the policy-gradient algorithm
        """
        
        self.FLAGS.optimizer_fn = self.FLAGS.optimizer_fn[0]

        if self.FLAGS.optimizer_fn[0] == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=float(self.FLAGS.optimizer_fn[1]),
                                                 beta_1=float(self.FLAGS.optimizer_fn[2]),
                                                 beta_2=float(self.FLAGS.optimizer_fn[3]),
                                                 epsilon=float(self.FLAGS.optimizer_fn[4]),
                                                 clipvalue=4)
        # elif self.FLAGS.optimizer_fn[0] == 'radam':
        #     optimizer_fun = tfa.optimizers.RectifiedAdam(learning_rate=float(FLAGS.optimizer_fn[1]),
        #                                              beta_1=float(FLAGS.optimizer_fn[2]),
        #                                              beta_2=float(FLAGS.optimizer_fn[3]),
        #                                              epsilon=float(FLAGS.optimizer_fn[4]),
        #                                              weight_decay=float(FLAGS.optimizer_fn[5]))


        # elif FLAGS.optimizer_fn[0] == 'adamw':
        #     optimizer_fun = tfa.optimizers.AdamW(learning_rate=float(FLAGS.optimizer_fn[1]),
        #                                      beta_1=float(FLAGS.optimizer_fn[2]),
        #                                      beta_2=float(FLAGS.optimizer_fn[3]), epsilon=float(FLAGS.optimizer_fn[4]),
        #                                      weight_decay=float(FLAGS.optimizer_fn[5]))
        
        self.generator_biased=Generator(self.FLAGS,False)
        
        if trainable == True:
            self.generator_biased.model.compile(
                optimizer=self.optimizer,
                loss = self.custom_loss(aux_array))
        
        self.generator_biased.model.load_weights(self.FLAGS.models_path['generator_unbiased_path'])

        
        return self.generator_biased.model
        

    def policy_gradient(self, gamma=1):   
        """ Implements the policy gradient algorithm to bias the Generator. """
        
        self.generator_biased = Sequential()
        
        policy = 1
        cumulative_rewards = []

        # Initialize the variables that will contain the output of each prediction
        dimen = len(self.token_table)
        states = []
        
        pol_pic50 = []
        pol_sas = []
        pol_rewards_pic50 = []
        pol_rewards_sas = [] 
        pol_pic50_reward_scaled = []
        pol_sas_reward_scaled = []
        
        all_rewards = []
        losses_generator = []
        memory_smiles = []
        
        # Re-compile the model to adapt the loss function and optimizer to the RL problem
        self.generator_biased.model = self.get_unbiased_model(np.arange(len(self.token_table)))
                
        for i in range(self.FLAGS.n_iterations):
            print(f'\nEpoch {i+1}/{self.FLAGS.n_iterations}')
            cur_reward = 0
            cur_pic50 = 0
            cur_sas = 0
            cur_reward_pic50 = 0 
            cur_reward_sas = 0  
            cur_reward_pic50_scaled = 0 
            cur_reward_sas_scaled = 0 
           
            aux_2 = np.zeros([self.FLAGS.max_str_len,len(self.token_table)])
            inputs = np.zeros([self.FLAGS.max_str_len,1])
            
            ii = 0
            
            for m in range(self.FLAGS.batch_size):
                # Sampling new trajectory
                correct_mol = False
                
                while correct_mol != True:
                    trajectory_ints,trajectory = self.generator_biased.generate(1)
      
                    try:                     
                        seq = trajectory[0][7:-5] # Remove <start> and <end> tokens
                        if '<Padd>' in seq: # '<Padd>' is the padding character
                            seq = Utils.remove_padding(trajectory)   
                        print('Sampled molecule: ',seq)
                        mol = Chem.MolFromSmiles(seq)
     
                        trajectory = '<Start>' + Chem.MolToSmiles(mol) + '<End>'
                    
                        if len(memory_smiles) > self.FLAGS.memory_length:
                                memory_smiles.remove(memory_smiles[0])                                    
                        memory_smiles.append(seq)
                        
                        if len(trajectory) <= self.FLAGS.max_str_len:
                            correct_mol = True
                                               
                    except:
                        print("Invalid SMILES!")

                # Processing the sampled molecule         
                mol_padded = Utils.pad_seq(seq,self.token_table,self.FLAGS)
                tokens = Utils.tokenize(self.FLAGS,mol_padded,self.token_table)   
                processed_mol = Utils.smiles2idx(tokens,self.tokenDict)
            
                # Apply the Multi-Head Attention to extract token scores and contextual embeddings
                token_scores,important_tokens,contextual_embeddings,tokens_transformer,_ = self.transformer_model.predict_step_rl(seq,trajectory_ints)
            
                rewards,sas,pic50 = Utils.get_reward_MO(self.predictor_usp7,seq,contextual_embeddings,memory_smiles)

                reward,rescaled_sas,rescaled_pic50 = Utils.scalarization(rewards,self.FLAGS.weights,self.FLAGS.range_pic50,self.FLAGS.range_sas)

                discounted_reward = reward
                cur_reward += reward
                cur_reward_pic50 += rewards[0]
                cur_reward_sas += rewards[1]
                cur_pic50 += pic50
                cur_sas += sas 
                cur_reward_pic50_scaled += rescaled_pic50
                cur_reward_sas_scaled += rescaled_sas
                
                # "Decompose" the trajectory while accumulating the loss
                inp_p = np.zeros([self.FLAGS.max_str_len,1])
                
                for p in range(1,len(trajectory_ints)):
                    if self.FLAGS.option == 'experiment_1':
                        states.append(discounted_reward)
                    else:
                        if p in important_tokens and pic50>=self.FLAGS.upper_pic50_thresh:
                            states.append(discounted_reward*self.FLAGS.reward_factor)

                        elif p in important_tokens and pic50 <= self.FLAGS.lower_pic50_thresh:

                            states.append(discounted_reward/self.FLAGS.reward_factor)

                        else:
                            states.append(discounted_reward)
                    
                    inp_p[p-1,0] = processed_mol[0,p-1]
                  
                    aux_2_matrix = np.zeros([self.FLAGS.max_str_len,len(self.token_table)])
                    aux_2_matrix[p-1,int(processed_mol[0,p])] = 1

                    if ii == 0:
                        aux_2 = aux_2_matrix
                        inputs = np.copy(inp_p)
    
                    else:
                        inputs = np.dstack([inputs,inp_p])
         
                        aux_2 = np.dstack([aux_2,aux_2_matrix])
    
                    ii += 1
                    
            inputs = np.moveaxis(inputs,-1,0)
            new_states = np.array(states)
            
            aux_2 = np.moveaxis(aux_2,-1,0)
            
            self.generator_biased.model.compile(optimizer = self.optimizer, loss = self.custom_loss(tf.convert_to_tensor(aux_2, dtype=tf.float64, name=None)))
           
            loss_generator = self.generator_biased.model.train_on_batch(inputs,new_states) # (inputs,targets) update the weights with a batch

            # Clear out variables
            states = []
            inputs = np.empty(0).reshape(0,0,dimen)

            cur_reward = cur_reward / self.FLAGS.batch_size
            cur_pic50 = cur_pic50 / self.FLAGS.batch_size
            cur_sas = cur_sas / self.FLAGS.batch_size
            cur_reward_pic50 = cur_reward_pic50 / self.FLAGS.batch_size
            cur_reward_sas = cur_reward_sas / self.FLAGS.batch_size
            cur_reward_pic50_scaled = cur_reward_pic50_scaled  / self.FLAGS.batch_size
            cur_reward_sas_scaled  = cur_reward_sas_scaled  / self.FLAGS.batch_size

            # serialize model
            Utils.serialize_model(self.generator_biased,self.FLAGS)

            if len(all_rewards) > 2: # decide the threshold of the next generated batch 
                threshold_greedy = Utils.compute_thresh(all_rewards[-3:],self.FLAGS.threshold_set)
 
            all_rewards.append(Utils.moving_average(all_rewards, cur_reward)) 
           
            pol_pic50.append(Utils.moving_average(pol_pic50, cur_pic50)) 
            pol_sas.append(Utils.moving_average(pol_sas, cur_sas))                   
            pol_rewards_pic50.append(Utils.moving_average(pol_rewards_pic50,cur_reward_pic50))
            pol_rewards_sas.append(Utils.moving_average(pol_rewards_sas,cur_reward_sas))
            pol_pic50_reward_scaled.append(Utils.moving_average(pol_pic50_reward_scaled, cur_reward_pic50_scaled))  
            pol_sas_reward_scaled.append(Utils.moving_average(pol_sas_reward_scaled, cur_reward_sas_scaled)) 
            
            losses_generator.append(Utils.moving_average(losses_generator, loss_generator))
                        
            if i%2==0 and i > 0 and self.FLAGS.option != 'experiment_2':
                self.weights = Utils.update_weights(pol_pic50_reward_scaled,pol_sas_reward_scaled,self.FLAGS.weights)
                # Utils.plot_training_progress(all_rewards,losses_generator,pol_pic50,pol_sas,pol_rewards_pic50,pol_rewards_sas,pol_pic50_reward_scaled,pol_sas_reward_scaled)
            if i%25==0 and i > 0:
                Utils.plot_training_progress(all_rewards,losses_generator,pol_pic50,pol_sas,pol_rewards_pic50,pol_rewards_sas,pol_pic50_reward_scaled,pol_sas_reward_scaled)
        
        Utils.plot_training_progress(all_rewards,losses_generator,pol_pic50,pol_sas,pol_rewards_pic50,pol_rewards_sas,pol_pic50_reward_scaled,pol_sas_reward_scaled)
        cumulative_rewards.append(np.mean(all_rewards[-10:]))

        policy+=1
    
        return cumulative_rewards

    def samples_generation(self,smiles_data=None):
        """
        It generates new hits, computes the desired properties, draws the 
        specified number of hits and saves the set of optimized compounds to 
        a file.
        
        Parameters:
        -----------
        smiles_data (DataFrame): It contains a set of generated molecules 
    
        """
        
        training_data = DataLoader().load_generator_smiles(self.FLAGS.models_path['generator_data_path'])
        
        if smiles_data is None:
            if self.FLAGS.option == 'unbiased':
                generator = self.generator_unbiased
            else:
                generator = DataLoader().load_generator(self.FLAGS,'biased')
            
            smiles_data = Utils().generate_smiles(generator,self.predictor_usp7,self.transformer_model,self.tokenDict,self.FLAGS)
            
            
        vld = (len(smiles_data['smiles'])/self.FLAGS.mols_to_generate)*100
        
        print("\nValid: ", round(vld,4))
        
        tanimoto_int = Utils().external_diversity(list(smiles_data['smiles']))
        print('\nInternal Tanimoto Diversity: ',round(tanimoto_int,4))
        
        tanimoto_ext = Utils().external_diversity(list(smiles_data['smiles']),list(training_data))
        print('\nExternal Tanimoto Diversity: ',round(tanimoto_ext,4))
    
        Utils().plot_hist(smiles_data['sas'],self.FLAGS.mols_to_generate,"sas")
        Utils().plot_hist(smiles_data['pIC50 usp7'],self.FLAGS.mols_to_generate,"usp7")
        
        qed_values = []
        logp_values = []
        with open(self.FLAGS.path_generated_mols+'mols.smi', 'w') as f:
            f.write("Number of molecules: %s\n" % str(len(smiles_data['smiles'])))
            f.write("Percentage of valid and unique molecules: %s\n\n" % str(vld))
            f.write("SMILES, pIC50, SAS, MW, logP, QED\n")
            for i,smi in enumerate(smiles_data['smiles']):
                mol = list(smiles_data['mols_obj'])[i]
                q = QED.qed(mol)
                mw, logP = Descriptors.MolWt(mol), Crippen.MolLogP(mol)
                
                qed_values.append(q)
                logp_values.append(logP)
                data = str(list(smiles_data['smiles'])[i]) + " ," +  str(np.round(smiles_data['pIC50 usp7'][i],2)) + " ," + str(np.round(smiles_data['sas'][i],2)) + " ,"  + str(np.round(mw,2)) + " ," + str(np.round(logP,2)) + " ," + str(np.round(q,2))
                f.write("%s\n" % data)  
                
        print("\nMax QED: ", round(np.max(qed_values),4))
        print("Mean QED: ", round(np.mean(qed_values),4))
        print("Std QED: ", round(np.std(qed_values),4))
        print("Min QED: ", round(np.min(qed_values),4))
    
        print("\nMax logP: ", round(np.max(logp_values),4))
        print("Mean logP: ", round(np.mean(logp_values),4))
        print("Std logP: ", round(np.std(logp_values),4))
        print("Min logP: ", round(np.min(logp_values),4))   

        if self.FLAGS.draw_mols == True:
            df_sorted = smiles_data.sort_values('pIC50 usp7',ascending = False)
            self.drawMols(list(df_sorted['smiles'])[:self.FLAGS.mols_to_draw])

    
    def compare_models(self):
        """
        Comparison of the molecules generated by the unbiased and biased models
        """
        generator_biased = DataLoader().load_generator(self.FLAGS,'biased')
        
        print('\nUnbiased Generation...')
        unbiased_smiles_data = Utils().generate_smiles(self.generator_unbiased,self.predictor_usp7,self.transformer_model,self.tokenDict,self.FLAGS)
        
        print('\nBiased Generation...')        
        self.biased_smiles_data = Utils().generate_smiles(generator_biased,self.predictor_usp7,self.transformer_model,self.tokenDict,self.FLAGS)
        
        # plot both distributions together and compute the % of valid generated by the biased model 
        Utils.plot_hist_both(unbiased_smiles_data['pIC50 usp7'],self.biased_smiles_data['pIC50 usp7'],unbiased_smiles_data['sas'],self.biased_smiles_data['sas'],self.FLAGS)
        
        self.samples_generation(self.biased_smiles_data)
        
        if self.FLAGS.show_images:
            self.compare_fg()
            
        
    def drawMols(self,smiles_generated=None):
        """
        Function that draws the chemical structure of given compounds

        Parameters:
        -----------
        smiles_generated (list): It contains a set of generated molecules
        
        Returns
        -------
        This function returns a figure with the specified number of molecules
        """
        
        DrawingOptions.atomLabelFontSize = 50
        DrawingOptions.dotsPerAngstrom = 100
        DrawingOptions.bondLineWidth = 3
        DrawingOptions.addStereoAnnotation = True  
        DrawingOptions.addAtomIndices = True
        
        
        if smiles_generated is None:
            smiles_generated = ['CC(C)C1CCC2(C)CCC3(C)C(CCC4C5(C)CCC(O)C(C)(C)C5CCC43C)C12',
                                  'Cc1cccc(C)c1C(=O)NC1CCCNC1=O','CC1CCC(C)C12C(=O)Nc1ccccc12',
                                  'NC(Cc1ccc(O)cc1)C(=O)O','CC(=O)OCC(=O)C1(OC(C)=O)CCC2C3CCC4=CC(=O)CCC4(C)C3CCC21C',
                                  'CC(=O)NCC1CN(c2cc3c(cc2F)c(=O)c(C(=O)O)cn3C2CC2)C(=O)N1',
                                  'CC1(C)CCC(C)(C)c2cc(C(=O)Nc3ccc(C(=O)O)cc3)ccc21','CC(=O)NCC1CN(c2cc3c(cc2F)c(=O)c(C(=O)O)cn3C2CC2)C(=O)N1',
                                  'CCC(c1ccccc1)c1ccc(OCCN(CC)CC)cc1','Cc1cccc(C)c1NC(=O)CN(C)C(=O)C(C)C','CC(C)C(CO)Nc1ccnc2cc(Cl)ccc12',
                                  'Cc1ccc(C(=O)Nc2ccc(C(C)C)cc2)cc1Nc1nccc(-c2ccc(C(F)(F)F)cc2)n1','CN(C)CCCNC(=O)CCC(=O)Nc1ccccc1',
                                  'CC(C)CC(=O)N(Cc1ccccc1)C1CCN(Cc2ccccc2)CC1']
            
            known_drugs = ['C[C@@H]1[C@H]2C3=CC[C@@H]4[C@@]5(C)CC[C@H](O)C(C)(C)[C@@H]5CC[C@@]4(C)[C@]3(C)CC[C@@]2(C(=O)O)CC[C@H]1C',
                           'O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1','CCC1(c2ccc(N)cc2)CCC(=O)NC1=O',
                           'CC(N)(Cc1ccc(O)cc1)C(=O)O','CC(=O)O[C@]1(C(C)=O)CC[C@H]2[C@@H]3C=C(C)C4=CC(=O)CC[C@]4(C)[C@H]3CC[C@@]21C',
                           'COc1c(N2C[C@@H]3CCCN[C@@H]3C2)c(F)cc2c(=O)c(C(=O)O)cn(C3CC3)c12',
                           'C=C(c1ccc(C(=O)O)cc1)c1cc2c(cc1C)C(C)(C)CCC2(C)C','O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O',
                           'CCN(CC)CCOc1ccc(Cc2ccccc2)cc1', 'CCN(CC)CC(=O)Nc1c(C)cccc1C', 'CCN(CCO)CCCC(C)Nc1ccnc2cc(Cl)ccc12', 
                           'Cc1cn(-c2cc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cccnc5)n4)c3)cc(C(F)(F)F)c2)cn1','O=C(CCCCCCC(=O)Nc1ccccc1)NO',
                           'CCC(=O)N(c1ccccc1)C1CCN(CCc2ccccc2)CC1']
            
            legends = ['Ursolic acid', 'Thalidomide', 'Aminoglutethimide',
                       'Racemetyrosine', 'Megestrol acetate', 'Moxifloxacin',
                       'Bexarotene', 'Ciproflaxicin', 'Tesmilifene', 'Lidocaine', 
                       'Hydroxycloroquine', 'Nilotilib', 'Vorinostat', 'Fentanyl']
            
           
            drugs_mols = Utils().smiles2mol(known_drugs)
       
            img = Draw.MolsToGridImage(drugs_mols, molsPerRow=3, subImgSize=(300,300),legends=legends)
            img.show()
            
        generated_mols = Utils().smiles2mol(smiles_generated)
            
        img = Draw.MolsToGridImage(generated_mols, molsPerRow=3, subImgSize=(300,300))
        img.show()
        
    def select_best_stereoisomers(self):
        """
        Identifies the best stereoisomer for each promising hit 
        """
        df = DataLoader().load_promising_mols(self.config)
        # # sort predictions
        df_sorted = df.sort_values('pic50',ascending = False)
        
        smiles_sorted = df_sorted['smiles'].tolist()
        
        opts = StereoEnumerationOptions(tryEmbedding=True,unique=True,onlyStereoGroups = False)

        for idx,smi in enumerate(smiles_sorted):
            print(smi)
            mols_list = Utils.smiles2mol([smi])
            stereo_isomers = list(EnumerateStereoisomers(mols_list[0],options=opts))
            
                   
            smiles_augmented = [Chem.MolToSmiles(stereo_mol) for stereo_mol in stereo_isomers]

            if len(smiles_augmented) <2:
                keep_mols = stereo_isomers
                keep_mols = Utils.smiles2mol(smiles_augmented)
            
            else:
                
                #prediction usp7 affinity
                prediction_usp7 = self.predictor_usp7.predict(smiles_augmented)
                
                # sort and get the original indexes
                out_arr = np.argsort(prediction_usp7)
                
                keep_indices = list(out_arr[-self.config.max_stereoisomers:])
                
                keep_smiles = [smiles_augmented[k_idx] for k_idx in keep_indices]
                
                keep_mols = Utils.smiles2mol(keep_smiles)
                
            if self.config.draw_mols == 'true':
                # DrawingOptions.addStereoAnnotation = True
                DrawingOptions.atomLabelFontSize = 50
                DrawingOptions.dotsPerAngstrom = 100
                DrawingOptions.bondLineWidth = 3
                                             

                
                legends = []
                for i in keep_indices:
                     legends.append('pIC50 for USP7: ' + str(prediction_usp7[i]))
           
                img1 = Draw.MolsToGridImage([mols_list[0]], molsPerRow=1, subImgSize=(300,300))
                img2 = Draw.MolsToGridImage(keep_mols, molsPerRow=3, subImgSize=(300,300))
                img1.show()
                
                img2.show()
                img1.save('generated\mols_canonical_best' + str(idx) + '.png')
                img2.save('generated\mols_stereoisomers_best' + str(idx) + '.png')
  
        
    def compare_fg(self):
        """
        Function that compares the functional groups frequency in generated
        and known USP7 inhibitors

        """

        generated_hits = list(self.biased_smiles_data['smiles'])
        known_inhibitors = DataLoader().load_usp7_mols(self.FLAGS)
        
        dict_fg_known = {}
        for smi in known_inhibitors:
            m =  Chem.MolFromSmiles(smi)
            fgs = ifg.identify_functional_groups(m)
            for i in range(0,len(fgs)):
                fg = fgs[i][1]
                
                if len(fg) == 1:
                    fg = fg.capitalize()
                    
                if fg in dict_fg_known:
                    dict_fg_known[fg] += 1   
                else:
                    dict_fg_known[fg] = 1  
        
        total_known = sum(dict_fg_known.values())
        dict_known_normalized = {key: value / total_known for key, value in dict_fg_known.items()}
        
        dict_fg_generated = {}
        for smi in generated_hits:

            m =  Chem.MolFromSmiles(smi)
            fgs = ifg.identify_functional_groups(m)
            for i in range(0,len(fgs)):
                fg = fgs[i][1]
                if len(fg) == 1:
                    fg = fg.capitalize()
                    
                if fg in dict_fg_generated:
                    dict_fg_generated[fg] += 1   
                else:
                    dict_fg_generated[fg] = 1  
        
        total_generated = sum(dict_fg_generated.values())
        dict_generated_normalized = {key: value / total_generated for key, value in dict_fg_generated.items()}
        
        Utils.plot_fg(dict_known_normalized,10,'known')
        Utils.plot_fg(dict_generated_normalized,10,'generated')
                
        td = 0
   
        fps_A = []
        for i, row in enumerate(generated_hits):
            try:
                # print(i,row)
                mol = Chem.MolFromSmiles(row)
                fps_A.append(AllChem.GetMorganFingerprint(mol, 3))
            except:
                print('ERROR: Invalid SMILES!')
    
        
        fps_B = []
        for j, row in enumerate(known_inhibitors):
            try:
                mol = Chem.MolFromSmiles(row)
                fps_B.append(AllChem.GetMorganFingerprint(mol, 3))
            except:
                print('ERROR: Invalid SMILES!') 
        
        td_all = []
        for jj in range(len(fps_A)):
            for xx in range(len(fps_B)):
                ts = 1 - DataStructs.TanimotoSimilarity(fps_A[jj], fps_B[xx]) 
                td_all.append(ts)
         
        td_all_arr = np.array(td_all,dtype='float64').reshape((-1,))    
        sns.axes_style("darkgrid")
        ax = sns.kdeplot(td_all_arr, shade=True,color = 'g')
        ax.set(xlabel='Tanimito Similarity',
               title='Distribution of TS between the generated compounds and known inhibitors ')
        plt.show() 

                                    

  

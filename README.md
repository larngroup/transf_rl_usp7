# De novo Molecular Design Applying Enhanced Reinforcement Learning Guided by Multi-Head Self-Attention Mechanisms

<p align="justify"> We propose a generative framework that combines DL methodologies and RL to guide the generation of molecules with optimized properties. This work introduces an intuitive RL setting in which the agent learns directly through its actions. In other words, we intend to move the evaluation of the actions from the level of molecules to the level of the tokens that constitute them. In this way, we implemented a strategy that provides different rewards for different parts of the molecule for the model to learn how to generate compounds with the active sites typically involved in the interaction with the target. The framework comprises an RNN-based Generator and a biological affinity Predictor connected by a Transformer-encoder. The latter model will apply MHSA to the sampled molecules in order to extract the attention scores and the informative embedding vectors that characterize the molecules. By distributing the attention scores along the molecule, it will be possible to indicate to the Generator the different levels of importance of the different parts of the molecule. The practical case addressed was the generation of a set of putative hit compounds against the USP7 due to the importance of this enzyme for the proliferation of different types of tumours. </p>


## Model Dynamics Architecture
<p align="center"><img src="/figures/figure1.jpg" width="90%" height="90%"/></p>

## Data Availability
### Molecular Generator
- **train_chembl_22_clean_1576904_sorted_std_final:** Train set extracted from ChEMBL 22
- **test_chembl_22_clean_1576904_sorted_std_final:** Test set extracted from ChEMBL 22
### USP7 inhibitors
- **usp7_new.smi:** pIC50 against USP7 + SMILES from ChEMBL 
- **usp_inhibitors_1.xlsx:** known USP7 inhibitors collected from the state-of-art 

## Requirements:
- Python 3.8.12
- Tensorflow 2.3.0
- Numpy 
- Pandas
- Scikit-learn
- Itertools
- Matplotlib
- Seaborn
- Bunch
- tqdm
- rdkit 2021.03.4

## Usage 
Codes to run the experiments described in the research article with the pre-trained Generator, Transformers and Predictors models.
### MLM (best approach)
```
python main.py --option mlm --upper_pic50_thresh 6.5 --lower_pic50_thresh 6.0 --batch_size 7 --top_tokens_rate 0.33 --reward_factor 3 --optimizer_fn adam 0.0005 0.9 0.999 1e-08
```
### Experiment I - MLM
```
python main.py --option mlm_exp1 --upper_pic50_thresh 6.5 --lower_pic50_thresh 6.0 --batch_size 7 --top_tokens_rate 0.33 --reward_factor 3 --optimizer_fn adam 0.0005 0.9 0.999 1e-08
```
### Experiment II - MLM
```
python main.py --option mlm_exp2 --upper_pic50_thresh 6.5 --lower_pic50_thresh 6.0 --batch_size 7 --top_tokens_rate 0.33 --reward_factor 3 --optimizer_fn adam 0.0005 0.9 0.999 1e-08
```
### Standard
```
python main.py --option standard --upper_pic50_thresh 6.0 --lower_pic50_thresh 5.5 --batch_size 7 --top_tokens_rate 0.2 --reward_factor 3 --optimizer_fn adam 0.0005 0.9 0.999 1e-08
```
### Experiment I - Standard
```
python main.py --option standard_exp1 --upper_pic50_thresh 6.0 --lower_pic50_thresh 5.5 --batch_size 7 --top_tokens_rate 0.2 --reward_factor 3 --range_pic50 [0.48,1.1] --range_sas [1.4,3.75] --weights [0.5,0.5] --optimizer_fn adam 0.0005 0.9 0.999 1e-08
```
### Experiment II - Standard
```
python main.py --option standard_exp2 --upper_pic50_thresh 6.0 --lower_pic50_thresh 5.5 --batch_size 7 --top_tokens_rate 0.2 --reward_factor 3 --range_pic50 [0.48,1.1] --range_sas [1.4,3.75] --weights [0.5,0.5] --optimizer_fn adam 0.0005 0.9 0.999 1e-08
```

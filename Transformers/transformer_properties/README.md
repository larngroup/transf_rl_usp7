# De novo Molecular Design Applying Enhanced Reinforcement Learning Guided by Multi-Head Self-Attention Mechanisms - Transformer Standard

<p align="justify"> We propose a generative framework that combines DL methodologies and RL to guide the generation of molecules with optimized properties. This work introduces an intuitive RL setting in which the agent learns directly through its actions. In other words, we intend to move the evaluation of the actions from the level of molecules to the level of the tokens that constitute them. In this way, we implemented a strategy that provides different rewards for different parts of the molecule for the model to learn how to generate compounds with the active sites typically involved in the interaction with the target. The framework comprises an RNN-based Generator and a biological affinity Predictor connected by a Transformer-encoder. The latter model will apply MHSA to the sampled molecules in order to extract the attention scores and the informative embedding vectors that characterize the molecules. By distributing the attention scores along the molecule, it will be possible to indicate to the Generator the different levels of importance of the different parts of the molecule. The practical case addressed was the generation of a set of putative hit compounds against the USP7 due to the importance of this enzyme for the proliferation of different types of tumours. </p>


## Model Dynamics Architecture
<p align="center"><img src="/figures/figure1.jpg" width="90%" height="90%"/></p>

## Data Availability
### Molecular Dataset
- **train_chembl_22_clean_1576904_sorted_std_final:** Train set extracted from ChEMBL 22
- **test_chembl_22_clean_1576904_sorted_std_final:** Test set extracted from ChEMBL 22

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
Codes to run the training and validation (grid-search) experiments
### Training
```
python main.py --option train --n_epochs 25 --d_model 256 --n_layers 4 --n_heads 4 --dropout 0.1 --activation_func gelu --ff_dim 1024 --optimizer_fn adam 0.0001 0.9 0.99 1e-08
```
### Validation
```
python main.py --option validation --n_epochs 25 --d_model 128 256 512 --n_layers 4 6 --n_heads 4 6 8 --dropout 0.1 0.15 0.2 --activation_func relu gelu --ff_dim 768 1024 2048 --optimizer_fn adam 0.0001 0.9 0.99 1e-08
```


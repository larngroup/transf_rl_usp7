# De novo Molecular Design Applying Enhanced Reinforcement Learning Guided by Multi-Head Self-Attention Mechanisms - Predictor

<p align="justify"> We propose a generative framework that combines DL methodologies and RL to guide the generation of molecules with optimized properties. This work introduces an intuitive RL setting in which the agent learns directly through its actions. In other words, we intend to move the evaluation of the actions from the level of molecules to the level of the tokens that constitute them. In this way, we implemented a strategy that provides different rewards for different parts of the molecule for the model to learn how to generate compounds with the active sites typically involved in the interaction with the target. The framework comprises an RNN-based Generator and a biological affinity Predictor connected by a Transformer-encoder. The latter model will apply MHSA to the sampled molecules in order to extract the attention scores and the informative embedding vectors that characterize the molecules. By distributing the attention scores along the molecule, it will be possible to indicate to the Generator the different levels of importance of the different parts of the molecule. The practical case addressed was the generation of a set of putative hit compounds against the USP7 due to the importance of this enzyme for the proliferation of different types of tumours. </p>


## Model Dynamics Architecture
<p align="center"><img src="/figures/figure1.jpg" width="90%" height="90%"/></p>

## Data Availability
### Molecular Dataset
- **usp7_new.csv:** Biological activity dataset against USP7 extracted from ChEMBL

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
### Training best Standard-based Predictor
```
python main.py --model standard --option train_model --bi_rnn True --normalization_strategy robust --optimizer_fn adam 0.0001 0.9 0.99 1e-08 --reduction_lr 0.8 4 1e-6 --rnn_1 256 --rnn_2 512 --dropout_rnn 0.15 --rnn_type gru
```
### Training best MLM-based Predictor
```
python main.py --model mlm --option train_model --normalization_strategy min_max --optimizer_fn adam 0.0001 0.9 0.99 1e-08 --reduction_lr 0.8 4 1e-6 --n_layers 3 --units 512  --activation_fc relu --dropout_fc 0.1 --paticence 5
```
### Validation Standard-based Predictor
```
python main.py --model standard --option grid_search --bi_rnn True False --normalization_strategy min_max percentile robust --optimizer_fn adam 0.0001 0.9 0.99 1e-08 --reduction_lr 0.8 4 1e-6 --rnn_1 512 256 128 --rnn_2 512 256 128 --dropout_rnn 0.1 0.15 0.2 --rnn_type gru lstm
```
### Validation MLM-based Predictor
```
python main.py --model mlm --option grid_search --normalization_strategy min_max percentile robust --optimizer_fn adam 0.0001 0.9 0.99 1e-08 --reduction_lr 0.8 4 1e-6 --n_layers 5 4 3 --units 512 256 128  --activation_fc gelu relu --dropout_fc 0.1 0.15 0.2
```

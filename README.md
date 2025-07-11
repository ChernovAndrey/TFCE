# TFCE & TFCE-MLP

This repository contains the implementation of **TFCE** and **TFCE-MLP**, developed for recommendation systems.  
It is a fork of the original [AlphaRec](https://github.com/LehengTHU/AlphaRec) repository.

## 🆕 Multi-Dataset Training

This repository now supports training on multiple datasets simultaneously! This feature allows you to:

- Train recommendation models on several datasets at once
- Maintain separate interaction graphs for each dataset
- Ensure negative sampling is isolated within each dataset
- Use proportional sampling based on dataset sizes

### Multi-Dataset Training Usage

```bash
# Train on multiple Amazon datasets simultaneously
python main.py --rs_type General --model_name MF \
  --multi_datasets amazon_movie amazon_book amazon_game \
  --multi_datasets_path data/General/ \
  --proportional_sampling \
  --batch_size 4096 --lr 0.001 --max_epoch 100

# Train with equal dataset weights
python main.py --rs_type General --model_name LightGCN \
  --multi_datasets amazon_movie amazon_book amazon_game \
  --multi_datasets_path data/General/ \
  --equal_sampling \
  --batch_size 2048 --lr 0.001 --max_epoch 200

# Train with custom dataset sampling weights
python main.py --rs_type General --model_name LightGCN \
  --multi_datasets amazon_movie amazon_book \
  --multi_datasets_path data/General/ \
  --dataset_sampling_weights 0.7 0.3 \
  --batch_size 2048 --lr 0.001 --max_epoch 200
```

### Key Features

- **Dataset Isolation**: Each dataset maintains its own interaction graph
- **Negative Sampling**: Negative samples are drawn only from the same dataset as the positive sample
- **Proportional Sampling**: Training samples are drawn proportionally to dataset sizes
- **Flexible Weighting**: Custom sampling weights can be specified for each dataset

### Multi-Dataset Arguments

- `--multi_datasets`: List of dataset names to train on simultaneously
- `--multi_datasets_path`: Base path where datasets are stored
- `--proportional_sampling`: Use proportional sampling based on dataset sizes
- `--equal_sampling`: Use equal sampling weights for all datasets
- `--dataset_sampling_weights`: Custom weights for dataset sampling (overrides other sampling methods)

---

## 🛠 Installation

Our experiments were tested on **Python 3.9.12** with **PyTorch 1.13.1+cu117**.  
⚠️ Python versions above 3.10 may cause issues with the `reckit` package.

### Step 1: Environment Setup

Set up a virtual environment and install PyTorch manually (as per your CUDA version) from the [official site](https://pytorch.org/get-started/previous-versions/).  
Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```


2. Before using the general recommendation, run the following command to install the evaluator:
```bash
pushd models/General/base
python setup.py build_ext --inplace
popd
```

### Dataset downloading

Please download the datasets from the following anonymous link and put the unzipped dataset in the `data` folder:

https://drive.google.com/drive/folders/1iGKeTx3vqCtbeVdWkHOwgpbY3-s7QDy_?usp=sharing

Example of the file structure:
```
├── assets/
├── models/
├── data/
    ├── General/
        ├── amazon_movie/ # target datasets
            ├── cf_data/
            ├── item_info/
```

### Commands for running 
TFCE-MLP
```bash
# Amazon Game
nohup python main.py --rs_type General --clear_checkpoints --saveID tfcemlp --dataset amazon_game --model_name TFCEMLP --n_layers 3 --patience 20 --cuda 0 --no_wandb --train_norm --pred_norm --neg_sample 512 --lm_model v3 --model_version mlp --tau 0.2 --infonce 1 --verbose 1 --no-is_one_pos_item --n_pos_samples 3 --hidden_size 128 &> logs_tfce_tuned/amazon_game.log &

# Amazon Movie
nohup python main.py --rs_type General --clear_checkpoints --saveID tfcemlp --dataset amazon_movie --model_name TFCEMLP --n_layers 2 --patience 20 --cuda 0 --no_wandb --train_norm --pred_norm --neg_sample 512 --lm_model v3 --model_version mlp --tau 0.15 --infonce 1 --verbose 1 --no-is_one_pos_item --n_pos_samples 9 --hidden_size 128 &> logs_tfce_tuned/amazon_movie.log &

# Amazon Book
nohup python main.py --rs_type General --clear_checkpoints --saveID tfcemlp --dataset amazon_book --model_name TFCEMLP --n_layers 3 --patience 20 --cuda 0 --no_wandb --train_norm --pred_norm --neg_sample 512 --lm_model v3 --model_version mlp --tau 0.15 --infonce 1 --verbose 1 --no-is_one_pos_item --n_pos_samples 7  --hidden_size 256 &> logs_tfce_tuned/amazon_book.log &```
```

TFCE
```bash
# Amazon Game
nohup python main.py --rs_type General --clear_checkpoints --saveID tfce --dataset amazon_game --model_name TFCE --n_layers 4 --cuda 0 --no_wandb --train_norm --pred_norm --lm_model v3 --model_version mlp --infonce 1 --verbose 1 &> logs_tfce_tuned/amazon_game.log &

# Amazon Movie
nohup python main.py --rs_type General --clear_checkpoints --saveID tfce --dataset amazon_movie --model_name TFCE --n_layers 3  --cuda 0 --no_wandb --train_norm --pred_norm  --lm_model v3 --model_version mlp --infonce 1 --verbose 1 &> logs_tfce_tuned/amazon_movie.log &

# Amazon Book
nohup python main.py --rs_type General --clear_checkpoints --saveID tfce --dataset amazon_book --model_name TFCE --n_layers 5 --cuda 0 --no_wandb --train_norm --pred_norm  --lm_model v3 --model_version mlp --infonce 1 --verbose 1 &> logs_tfce_tuned/amazon_book.log &
```


#!/usr/bin/env python3
"""
Multi-Dataset Training Example for AlphaRec

This script demonstrates how to train recommendation models on multiple datasets simultaneously.
The implementation ensures:
1. Each dataset maintains its own interaction graph
2. Negative sampling is isolated within each dataset
3. Sampling ratio is proportional to dataset size (or custom weights)

Usage:
    python multi_dataset_example.py
"""

import argparse
import sys
from parse import parse_args
from utils import fix_seeds
from models.General.base.abstract_RS import AbstractRS
from models.General.base.abstract_data import MultiDatasetData

def create_multi_dataset_args():
    """Create example arguments for multi-dataset training"""
    parser = argparse.ArgumentParser()
    
    # Basic arguments
    parser.add_argument('--rs_type', type=str, default='General')
    parser.add_argument('--model_name', type=str, default='MF')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--regs', type=float, default=0.0001)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--Ks', type=int, default=20)
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--clear_checkpoints', action='store_true', default=False)
    parser.add_argument('--saveID', type=str, default='multi_dataset_example')
    parser.add_argument('--train_norm', action='store_true', default=False)
    parser.add_argument('--pred_norm', action='store_true', default=False)
    parser.add_argument('--neg_sample', type=int, default=64)
    parser.add_argument('--infonce', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='data/General/')
    parser.add_argument('--nodrop', action='store_true', default=False)
    parser.add_argument('--candidate', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--is_one_pos_item', action='store_true', default=True)
    parser.add_argument('--n_pos_samples', type=int, default=1)
    parser.add_argument('--no_wandb', action='store_true', default=True)
    
    # Multi-dataset specific arguments
    parser.add_argument('--multi_datasets', nargs='+', 
                       default=['amazon_movie', 'amazon_book', 'amazon_game'],
                       help='List of datasets to train on simultaneously')
    parser.add_argument('--multi_datasets_path', type=str, default='data/General/',
                       help='Base path for multi-dataset training')
    parser.add_argument('--proportional_sampling', action='store_true', default=False,
                       help='Use proportional sampling based on dataset sizes')
    parser.add_argument('--equal_sampling', action='store_true', default=True,
                       help='Use equal sampling weights for all datasets')
    parser.add_argument('--dataset_sampling_weights', nargs='+', type=float, default=None,
                       help='Custom weights for dataset sampling')
    
    return parser.parse_args()

def main():
    """Main function to demonstrate multi-dataset training"""
    print("=== Multi-Dataset Training Example ===")
    
    # Parse arguments
    args = create_multi_dataset_args()
    
    # Set random seed
    fix_seeds(args.seed)
    
    print(f"Training on datasets: {args.multi_datasets}")
    print(f"Model: {args.model_name}")
    print(f"Proportional sampling: {args.proportional_sampling}")
    if args.dataset_sampling_weights:
        print(f"Custom weights: {args.dataset_sampling_weights}")
    
    try:
        # Import and create the recommender system
        print(f"Loading model: {args.model_name}")
        exec(f'from models.General.{args.model_name} import {args.model_name}_RS')
        RS = eval(f'{args.model_name}_RS(args, {{}})')
        
        # Verify multi-dataset setup
        if hasattr(RS.data, 'dataset_info'):
            print("\n=== Dataset Information ===")
            for dataset_name, info in RS.data.dataset_info.items():
                print(f"{dataset_name}:")
                print(f"  Users: {info['n_users']}")
                print(f"  Items: {info['n_items']}")
                print(f"  Interactions: {info['n_interactions']}")
                print(f"  Weight: {RS.data.dataset_weights.get(dataset_name, 'N/A'):.4f}")
            
            print(f"\nCombined dataset:")
            print(f"  Total users: {RS.data.n_users}")
            print(f"  Total items: {RS.data.n_items}")
            print(f"  Total interactions: {RS.data.n_observations}")
        
        # Start training
        print("\n=== Starting Training ===")
        RS.execute()
        
        print("\n=== Training Complete ===")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("This example requires the datasets to be properly structured.")
        print("Please ensure your datasets are in the correct format and location.")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
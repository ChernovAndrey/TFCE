import argparse
import os
import torch
from models.General.base.abstract_RS import AbstractRS
from parse import parse_args
import re

def get_model_args_from_checkpoint(checkpoint_dir):
    """Load original model arguments from the checkpoint directory"""
    args_file = os.path.join(checkpoint_dir, 'args.txt')
    if not os.path.exists(args_file):
        raise FileNotFoundError(f"Arguments file not found in {args_file}")
    
    import json
    with open(args_file, 'r') as f:
        saved_args = json.load(f)
    return saved_args

def find_best_checkpoint(checkpoint_dir):
    """Find the checkpoint with the highest epoch number"""
    cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
               if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]
    
    if not cp_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    # Extract epoch numbers and find the highest
    regex = re.compile(r'\d+')
    epoch_list = []
    for cp in cp_files:
        epoch_list.append([int(x) for x in regex.findall(cp)][0])
    
    best_epoch = max(epoch_list)
    return best_epoch

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test a trained model on a new dataset')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                      help='Path to the model checkpoint directory')
    parser.add_argument('--test_data_path', type=str, required=True,
                      help='Path to the test dataset')
    parser.add_argument('--cuda', type=int, default=0,
                      help='GPU device ID. Use -1 for CPU')
    test_args = parser.parse_args()

    # Load original model arguments
    saved_args = get_model_args_from_checkpoint(test_args.checkpoint_path)
    
    # Create args namespace combining saved and new arguments
    class Args:
        pass
    combined_args = Args()
    
    # First set all saved arguments
    for key, value in saved_args.items():
        setattr(combined_args, key, value)
    
    # Override with test-specific arguments
    combined_args.data_path = os.path.dirname(test_args.test_data_path.rstrip('/')) + '/'
    combined_args.dataset = os.path.basename(test_args.test_data_path.rstrip('/'))
    combined_args.cuda = test_args.cuda
    combined_args.test_only = True
    combined_args.no_wandb = True
    combined_args.clear_checkpoints = False  # Make sure we don't clear checkpoints
    
    # Find the best checkpoint epoch
    best_epoch = find_best_checkpoint(test_args.checkpoint_path)
    print(f"Found best checkpoint at epoch {best_epoch}")
    
    # Get special arguments from the original parse_args
    args, special_args = parse_args()
    
    # Initialize model
    rs = AbstractRS(combined_args, special_args)
    
    # Set paths and load checkpoint
    print(f"Loading model from {test_args.checkpoint_path}")
    rs.base_path = test_args.checkpoint_path
    rs.data.best_valid_epoch = best_epoch  # Set the best epoch for loading
    
    # Execute testing
    rs.execute()
    
    print("Testing completed. Results saved in:", rs.base_path)

if __name__ == "__main__":
    main() 
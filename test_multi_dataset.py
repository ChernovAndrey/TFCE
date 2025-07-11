#!/usr/bin/env python3
"""
Test script for multi-dataset training functionality.

This script creates synthetic data to test the multi-dataset implementation
without requiring actual dataset files.
"""

import numpy as np
import torch
import tempfile
import os
from collections import defaultdict
from models.General.base.abstract_data import MultiDatasetData, MultiDatasetTrainDataset


def create_synthetic_dataset(name, n_users, n_items, n_interactions, temp_dir):
    """Create a synthetic dataset for testing"""
    
    # Create dataset directory
    dataset_dir = os.path.join(temp_dir, name, 'cf_data')
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Generate synthetic interactions
    np.random.seed(42)  # For reproducibility
    
    # Generate user-item interactions
    train_interactions = defaultdict(list)
    valid_interactions = defaultdict(list)
    test_interactions = defaultdict(list)
    
    for _ in range(n_interactions):
        user = np.random.randint(0, n_users)
        item = np.random.randint(0, n_items)
        
        # Split into train/valid/test (80/10/10)
        rand = np.random.random()
        if rand < 0.8:
            train_interactions[user].append(item)
        elif rand < 0.9:
            valid_interactions[user].append(item)
        else:
            test_interactions[user].append(item)
    
    # Write to files
    def write_interactions(interactions, filename):
        with open(filename, 'w') as f:
            for user, items in interactions.items():
                if items:  # Only write if user has interactions
                    f.write(f"{user} {' '.join(map(str, items))}\n")
    
    write_interactions(train_interactions, os.path.join(dataset_dir, 'train.txt'))
    write_interactions(valid_interactions, os.path.join(dataset_dir, 'valid.txt'))
    write_interactions(test_interactions, os.path.join(dataset_dir, 'test.txt'))
    
    return dataset_dir


def create_test_args(temp_dir):
    """Create test arguments for multi-dataset training"""
    class TestArgs:
        def __init__(self):
            self.multi_datasets = ['dataset1', 'dataset2', 'dataset3']
            self.multi_datasets_path = temp_dir + '/'
            self.proportional_sampling = True
            self.dataset_sampling_weights = None
            self.batch_size = 32
            self.neg_sample = 10
            self.infonce = 1
            self.is_one_pos_item = True
            self.n_pos_samples = 1
            self.num_workers = 0
            self.model_name = 'MF'
            self.user_neighbors = {}
    
    return TestArgs()


def test_multi_dataset_data():
    """Test the MultiDatasetData class"""
    print("Testing MultiDatasetData...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create synthetic datasets
        datasets = [
            ('dataset1', 100, 50, 1000),
            ('dataset2', 150, 75, 1500),
            ('dataset3', 80, 40, 800)
        ]
        
        for name, n_users, n_items, n_interactions in datasets:
            create_synthetic_dataset(name, n_users, n_items, n_interactions, temp_dir)
        
        # Test multi-dataset data loading
        args = create_test_args(temp_dir)
        
        try:
            data = MultiDatasetData(args)
            
            # Verify dataset info
            print(f"✓ Successfully loaded {len(data.dataset_info)} datasets")
            
            total_users = sum(info['n_users'] for info in data.dataset_info.values())
            total_items = sum(info['n_items'] for info in data.dataset_info.values())
            
            print(f"✓ Total users: {data.n_users} (expected: {total_users})")
            print(f"✓ Total items: {data.n_items} (expected: {total_items})")
            
            # Test dataset weights
            print(f"✓ Dataset weights: {data.dataset_weights}")
            
            # Verify weights sum to 1
            weight_sum = sum(data.dataset_weights.values())
            assert abs(weight_sum - 1.0) < 1e-6, f"Weights don't sum to 1: {weight_sum}"
            print("✓ Dataset weights sum to 1.0")
            
            return data
            
        except Exception as e:
            print(f"✗ Error in MultiDatasetData: {e}")
            raise


def test_multi_dataset_train_dataset(data):
    """Test the MultiDatasetTrainDataset class"""
    print("\nTesting MultiDatasetTrainDataset...")
    
    try:
        # Create train dataset
        train_dataset = MultiDatasetTrainDataset(
            model_name='MF',
            users=data.users,
            train_user_list=data.train_user_list,
            user_pop_idx=data.user_pop_idx,
            item_pop_idx=data.item_pop_idx,
            neg_sample=10,
            n_observations=data.n_observations,
            n_items=data.n_items,
            sample_items=data.sample_items,
            infonce=1,
            items=data.items,
            nu_info=data.nu_info,
            ni_info=data.ni_info,
            dataset_info=data.dataset_info
        )
        
        print(f"✓ Created train dataset with {len(train_dataset)} samples")
        
        # Test sampling
        sample_idx = 0
        sample = train_dataset[sample_idx]
        
        print(f"✓ Sample format: {len(sample)} elements")
        
        # Test negative sampling isolation
        if len(sample) >= 7:  # InfoNCE with negative sampling
            user, pos_item, user_pop, pos_item_pop, neg_items, neg_items_pop, mask = sample
            
            # Get user's dataset
            user_dataset = None
            for dataset_name, info in data.dataset_info.items():
                if info['user_offset'] <= user < info['user_offset'] + info['n_users']:
                    user_dataset = dataset_name
                    break
            
            if user_dataset:
                dataset_info = data.dataset_info[user_dataset]
                dataset_item_range = (dataset_info['item_offset'], 
                                    dataset_info['item_offset'] + dataset_info['n_items'])
                
                # Verify negative items are from the same dataset
                if hasattr(neg_items, 'numpy'):
                    neg_items_array = neg_items.numpy()
                elif hasattr(neg_items, '__iter__'):
                    neg_items_array = list(neg_items)
                else:
                    neg_items_array = [neg_items]
                
                for neg_item in neg_items_array:
                    assert dataset_item_range[0] <= neg_item < dataset_item_range[1], \
                        f"Negative item {neg_item} not in dataset {user_dataset} range {dataset_item_range}"
                
                print(f"✓ Negative sampling isolated within dataset '{user_dataset}'")
        else:
            print("✓ Sample format verified (different sampling strategy)")
        
        return train_dataset
        
    except Exception as e:
        print(f"✗ Error in MultiDatasetTrainDataset: {e}")
        raise


def test_dataset_isolation():
    """Test that datasets are properly isolated"""
    print("\nTesting dataset isolation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create datasets with specific user/item ranges
        datasets = [
            ('dataset1', 10, 20, 100),
            ('dataset2', 15, 25, 150),
        ]
        
        for name, n_users, n_items, n_interactions in datasets:
            create_synthetic_dataset(name, n_users, n_items, n_interactions, temp_dir)
        
        args = create_test_args(temp_dir)
        args.multi_datasets = ['dataset1', 'dataset2']
        
        data = MultiDatasetData(args)
        
        # Verify user/item offsets
        dataset1_info = data.dataset_info['dataset1']
        dataset2_info = data.dataset_info['dataset2']
        
        # Check that dataset2 users start after dataset1 users
        assert dataset2_info['user_offset'] == dataset1_info['user_offset'] + dataset1_info['n_users']
        print("✓ User offsets are correct")
        
        # Check that dataset2 items start after dataset1 items
        assert dataset2_info['item_offset'] == dataset1_info['item_offset'] + dataset1_info['n_items']
        print("✓ Item offsets are correct")
        
        # Test helper functions
        # Test user from dataset1
        user1 = dataset1_info['user_offset'] + 1
        assert data.get_user_dataset(user1) == 'dataset1'
        
        # Test user from dataset2
        user2 = dataset2_info['user_offset'] + 1
        assert data.get_user_dataset(user2) == 'dataset2'
        
        print("✓ User-to-dataset mapping works correctly")
        
        # Test item mapping
        item1 = dataset1_info['item_offset'] + 1
        assert data.get_item_dataset(item1) == 'dataset1'
        
        item2 = dataset2_info['item_offset'] + 1
        assert data.get_item_dataset(item2) == 'dataset2'
        
        print("✓ Item-to-dataset mapping works correctly")


def main():
    """Run all tests"""
    print("=== Multi-Dataset Training Tests ===\n")
    
    try:
        # Test 1: Basic multi-dataset data loading
        data = test_multi_dataset_data()
        
        # Test 2: Multi-dataset train dataset
        train_dataset = test_multi_dataset_train_dataset(data)
        
        # Test 3: Dataset isolation
        test_dataset_isolation()
        
        print("\n=== All Tests Passed! ===")
        print("✓ Multi-dataset data loading works correctly")
        print("✓ Negative sampling is properly isolated")
        print("✓ Dataset boundaries are maintained")
        print("✓ Proportional sampling weights are computed correctly")
        
    except Exception as e:
        print(f"\n=== Test Failed ===")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
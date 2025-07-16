import numpy as np
import scipy.sparse as sp
from models.General.base.abstract_data import AbstractData
from parse import parse_args
import collections
import os

def read_user_item_pairs(file_path):
    """Read user-item pairs from a text file"""
    user_item_pairs = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    user_id = int(parts[0])
                    # Each line can have multiple item IDs after the user ID
                    for item_id_str in parts[1:]:
                        item_id = int(item_id_str)
                        user_item_pairs.append((user_id, item_id))
    return user_item_pairs

def write_user_item_pairs(file_path, user_item_dict, item_id_mapping=None):
    """Write user-item pairs to a text file"""
    with open(file_path, 'w') as f:
        for user_id in sorted(user_item_dict.keys()):
            if user_item_dict[user_id]:  # Only write if user has items
                if item_id_mapping is not None:
                    # Map old item IDs to new item IDs
                    mapped_items = [item_id_mapping[item_id] for item_id in user_item_dict[user_id]]
                    items_str = ' '.join(map(str, sorted(mapped_items)))
                else:
                    items_str = ' '.join(map(str, sorted(user_item_dict[user_id])))
                f.write(f"{user_id} {items_str}\n")

def validate_dataset_consistency(train_user_items, test_user_items, valid_user_items, item_id_mapping):
    """Validate that the cleaned dataset is consistent"""
    print(f"\nValidating dataset consistency...")
    
    # Check that all item IDs in the cleaned files are within the new range
    max_new_item_id = len(item_id_mapping) - 1 if item_id_mapping else 0
    
    all_test_items = set()
    for items in test_user_items.values():
        all_test_items.update(items)
    
    all_valid_items = set()
    for items in valid_user_items.values():
        all_valid_items.update(items)
    
    all_train_items = set()
    for items in train_user_items.values():
        all_train_items.update(items)
    
    # Check if any item IDs are out of range
    if item_id_mapping:
        invalid_test_items = [item for item in all_test_items if item > max_new_item_id]
        invalid_valid_items = [item for item in all_valid_items if item > max_new_item_id]
        invalid_train_items = [item for item in all_train_items if item > max_new_item_id]
        
        if invalid_test_items or invalid_valid_items or invalid_train_items:
            print(f"  ERROR: Found item IDs outside valid range (0-{max_new_item_id}):")
            if invalid_test_items:
                print(f"    Test items: {invalid_test_items[:10]}...")
            if invalid_valid_items:
                print(f"    Valid items: {invalid_valid_items[:10]}...")
            if invalid_train_items:
                print(f"    Train items: {invalid_train_items[:10]}...")
            return False
    
    # Check that all users have at least one item
    empty_test_users = [user for user, items in test_user_items.items() if not items]
    empty_valid_users = [user for user, items in valid_user_items.items() if not items]
    empty_train_users = [user for user, items in train_user_items.items() if not items]
    
    if empty_test_users or empty_valid_users or empty_train_users:
        print(f"  ERROR: Found users with no items:")
        if empty_test_users:
            print(f"    Test users: {empty_test_users[:10]}...")
        if empty_valid_users:
            print(f"    Valid users: {empty_valid_users[:10]}...")
        if empty_train_users:
            print(f"    Train users: {empty_train_users[:10]}...")
        return False
    
    print(f"  ✓ All item IDs are within valid range (0-{max_new_item_id})")
    print(f"  ✓ All users have at least one item")
    print(f"  ✓ Dataset consistency validated successfully")
    return True

def clean_dataset():
    """Clean dataset by removing items without interactions and empty user rows"""
    
    # Get args and create data object
    args, special_args = parse_args()
    
    # Create data object
    data = AbstractData(args)
    
    print(f"Dataset: {args.dataset}")
    print(f"Number of users: {data.n_users}")
    print(f"Number of items: {data.n_items}")
    print(f"Number of training interactions: {len(data.trainUser)}")
    
    # Find items without training interactions
    items_with_interactions = set(data.trainItem)
    all_items = set(range(data.n_items))
    items_without_interactions = all_items - items_with_interactions
    
    print(f"\nItems without training interactions: {len(items_without_interactions)}")
    if items_without_interactions:
        print(f"Item IDs without interactions: {sorted(list(items_without_interactions))[:20]}...")
    
    if not items_without_interactions:
        print("No items without interactions found. Dataset is already clean.")
        return
    
    # Read test.txt and valid.txt files
    dataset_path = f"data/General/{args.dataset}/cf_data"
    test_file = os.path.join(dataset_path, "test.txt")
    valid_file = os.path.join(dataset_path, "valid.txt")
    train_file = os.path.join(dataset_path, "train.txt")
    
    print(f"\nReading test file: {test_file}")
    test_pairs = read_user_item_pairs(test_file)
    print(f"Number of test pairs: {len(test_pairs)}")
    
    print(f"Reading valid file: {valid_file}")
    valid_pairs = read_user_item_pairs(valid_file)
    print(f"Number of valid pairs: {len(valid_pairs)}")
    
    print(f"Reading train file: {train_file}")
    train_pairs = read_user_item_pairs(train_file)
    print(f"Number of train pairs: {len(train_pairs)}")
    
    # Create dictionaries to store user-item mappings
    test_user_items = {}
    valid_user_items = {}
    train_user_items = {}
    
    # Process test pairs
    for user_id, item_id in test_pairs:
        if user_id not in test_user_items:
            test_user_items[user_id] = set()
        test_user_items[user_id].add(item_id)
    
    # Process valid pairs
    for user_id, item_id in valid_pairs:
        if user_id not in valid_user_items:
            valid_user_items[user_id] = set()
        valid_user_items[user_id].add(item_id)
    
    # Process train pairs
    for user_id, item_id in train_pairs:
        if user_id not in train_user_items:
            train_user_items[user_id] = set()
        train_user_items[user_id].add(item_id)
    
    print(f"\nBefore cleaning:")
    print(f"  - Train users: {len(train_user_items)}")
    print(f"  - Test users: {len(test_user_items)}")
    print(f"  - Valid users: {len(valid_user_items)}")
    
    # Remove items without interactions from test set
    test_items_removed = 0
    test_users_removed = 0
    for user_id in list(test_user_items.keys()):
        original_items = len(test_user_items[user_id])
        test_user_items[user_id] = test_user_items[user_id] - items_without_interactions
        items_removed = original_items - len(test_user_items[user_id])
        test_items_removed += items_removed
        
        # Remove user if no items left
        if not test_user_items[user_id]:
            del test_user_items[user_id]
            test_users_removed += 1
    
    # Remove items without interactions from valid set
    valid_items_removed = 0
    valid_users_removed = 0
    for user_id in list(valid_user_items.keys()):
        original_items = len(valid_user_items[user_id])
        valid_user_items[user_id] = valid_user_items[user_id] - items_without_interactions
        items_removed = original_items - len(valid_user_items[user_id])
        valid_items_removed += items_removed
        
        # Remove user if no items left
        if not valid_user_items[user_id]:
            del valid_user_items[user_id]
            valid_users_removed += 1
    
    # Remove items without interactions from train set (for consistency)
    train_items_removed = 0
    train_users_removed = 0
    for user_id in list(train_user_items.keys()):
        original_items = len(train_user_items[user_id])
        train_user_items[user_id] = train_user_items[user_id] - items_without_interactions
        items_removed = original_items - len(train_user_items[user_id])
        train_items_removed += items_removed
        
        # Remove user if no items left
        if not train_user_items[user_id]:
            del train_user_items[user_id]
            train_users_removed += 1
    
    print(f"\nAfter cleaning:")
    print(f"  - Train items removed: {train_items_removed}")
    print(f"  - Train users removed: {train_users_removed}")
    print(f"  - Test items removed: {test_items_removed}")
    print(f"  - Test users removed: {test_users_removed}")
    print(f"  - Valid items removed: {valid_items_removed}")
    print(f"  - Valid users removed: {valid_users_removed}")
    print(f"  - Remaining train users: {len(train_user_items)}")
    print(f"  - Remaining test users: {len(test_user_items)}")
    print(f"  - Remaining valid users: {len(valid_user_items)}")
    
    # Create item ID mapping from 0 to n_rest_items
    items_to_keep = sorted(list(items_with_interactions))
    item_id_mapping = {old_id: new_id for new_id, old_id in enumerate(items_to_keep)}
    print(f"\nItem ID mapping created:")
    print(f"  - Original items: {data.n_items}")
    print(f"  - Items to keep: {len(items_to_keep)}")
    print(f"  - Items removed: {len(items_without_interactions)}")
    print(f"  - New item ID range: 0 to {len(items_to_keep)-1}")
    
    # Create backup files
    backup_test_file = test_file + ".backup"
    backup_valid_file = valid_file + ".backup"
    backup_train_file = train_file + ".backup"
    
    print(f"\nCreating backups...")
    if os.path.exists(test_file):
        os.system(f"cp {test_file} {backup_test_file}")
        print(f"  - Test backup created: {backup_test_file}")
    
    if os.path.exists(valid_file):
        os.system(f"cp {valid_file} {backup_valid_file}")
        print(f"  - Valid backup created: {backup_valid_file}")
    
    if os.path.exists(train_file):
        os.system(f"cp {train_file} {backup_train_file}")
        print(f"  - Train backup created: {backup_train_file}")
    
    # Write cleaned files with new item IDs
    print(f"\nWriting cleaned files with new item IDs...")
    write_user_item_pairs(test_file, test_user_items, item_id_mapping)
    print(f"  - Cleaned test file written: {test_file}")
    
    write_user_item_pairs(valid_file, valid_user_items, item_id_mapping)
    print(f"  - Cleaned valid file written: {valid_file}")
    
    write_user_item_pairs(train_file, train_user_items, item_id_mapping)
    print(f"  - Cleaned train file written: {train_file}")
    
    # Validate the cleaned dataset
    if not validate_dataset_consistency(train_user_items, test_user_items, valid_user_items, item_id_mapping):
        print(f"\nERROR: Dataset validation failed! Please check the issues above.")
        return
    
    # Create item ID mapping file for reference (always create this)
    dataset_root = f"data/General/{args.dataset}"
    item_info_dir = os.path.join(dataset_root, "item_info")
    os.makedirs(item_info_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    mapping_file = os.path.join(item_info_dir, "item_id_mapping.txt")
    with open(mapping_file, 'w') as f:
        f.write("# Original item ID -> New item ID mapping\n")
        f.write("# Format: original_id new_id\n")
        f.write(f"# Total items: {len(item_id_mapping)}\n")
        f.write(f"# Original range: 0 to {data.n_items-1}\n")
        f.write(f"# New range: 0 to {len(item_id_mapping)-1}\n")
        f.write("#\n")
        for original_id, new_id in item_id_mapping.items():
            f.write(f"{original_id} {new_id}\n")
    print(f"  - Item ID mapping saved: {mapping_file}")
    
    # Clean item embeddings file
    embeddings_file = os.path.join(item_info_dir, "item_cf_embeds_large3_array.npy")
    if os.path.exists(embeddings_file):
        print(f"\nCleaning item embeddings file: {embeddings_file}")
        
        # Load embeddings
        embeddings = np.load(embeddings_file)
        print(f"  - Original embeddings shape: {embeddings.shape}")
        print(f"  - Original number of items: {embeddings.shape[0]}")
        
        # Create backup
        backup_embeddings_file = embeddings_file + ".backup"
        os.system(f"cp {embeddings_file} {backup_embeddings_file}")
        print(f"  - Embeddings backup created: {backup_embeddings_file}")
        
        # Remove embeddings for items without interactions
        # Keep only embeddings for items that have training interactions
        cleaned_embeddings = embeddings[items_to_keep]
        
        print(f"  - Items removed from embeddings: {len(items_without_interactions)}")
        print(f"  - Items kept in embeddings: {len(items_to_keep)}")
        print(f"  - New embeddings shape: {cleaned_embeddings.shape}")
        
        # Save cleaned embeddings
        np.save(embeddings_file, cleaned_embeddings)
        print(f"  - Cleaned embeddings saved: {embeddings_file}")
        
        # Validate embeddings consistency
        print(f"  - Validating embeddings consistency...")
        if cleaned_embeddings.shape[0] != len(item_id_mapping):
            print(f"    ERROR: Embeddings shape {cleaned_embeddings.shape[0]} doesn't match item count {len(item_id_mapping)}")
            return
        print(f"    ✓ Embeddings shape matches item count")
        
    else:
        print(f"\nWarning: Embeddings file not found: {embeddings_file}")
    
    print(f"\nDataset cleaning completed!")
    print(f"Backup files created with .backup extension")
    print(f"All files updated with new item IDs (0 to {len(items_to_keep)-1})")
    print(f"Item ID mapping saved to: {mapping_file}")
    print(f"\nSummary:")
    print(f"  - Original items: {data.n_items}")
    print(f"  - Final items: {len(items_to_keep)}")
    print(f"  - Items removed: {len(items_without_interactions)}")
    print(f"  - Train users: {len(train_user_items)}")
    print(f"  - Test users: {len(test_user_items)}")
    print(f"  - Valid users: {len(valid_user_items)}")
    print(f"\nThe dataset is now clean and consistent!")

if __name__ == "__main__":
    clean_dataset() 
# Multi-Dataset Training Documentation

## Overview

This document describes the multi-dataset training functionality added to the AlphaRec repository. This feature allows training recommendation models on multiple datasets simultaneously while maintaining proper isolation between datasets for negative sampling.

## Key Features

### 1. Dataset Isolation
- Each dataset maintains its own interaction graph
- User and item IDs are remapped to avoid conflicts between datasets
- Negative sampling is constrained within each dataset

### 2. Proportional Sampling
- Training samples are drawn proportionally to dataset sizes by default
- Custom sampling weights can be specified for each dataset
- Ensures balanced representation across datasets

### 3. Flexible Configuration
- Support for any number of datasets
- Customizable dataset paths and sampling strategies
- Compatible with all existing model architectures

## Implementation Details

### Data Structure

The multi-dataset training uses the following approach:

1. **ID Remapping**: User and item IDs are remapped with offsets to ensure uniqueness across datasets
2. **Dataset Metadata**: Each dataset's information is stored including user/item counts, interaction counts, and offset values
3. **Negative Sampling**: When sampling negative items for a user, only items from the same dataset are considered

### Class Hierarchy

```
AbstractData (base class)
├── MultiDatasetData (multi-dataset support)
└── TrainDataset (base dataset class)
    └── MultiDatasetTrainDataset (multi-dataset training)
```

### Key Components

#### MultiDatasetData Class
- Loads and combines multiple datasets
- Handles ID remapping and offset management
- Computes dataset sampling weights
- Creates unified interaction graphs

#### MultiDatasetTrainDataset Class
- Extends TrainDataset with multi-dataset awareness
- Implements dataset-isolated negative sampling
- Maintains user-to-dataset mappings

## Usage Examples

### Basic Multi-Dataset Training

```bash
python main.py --rs_type General --model_name MF \
  --multi_datasets amazon_movie amazon_book amazon_game \
  --multi_datasets_path data/General/ \
  --proportional_sampling \
  --batch_size 4096 --lr 0.001 --max_epoch 100
```

### Custom Dataset Weights

```bash
python main.py --rs_type General --model_name LightGCN \
  --multi_datasets amazon_movie amazon_book \
  --multi_datasets_path data/General/ \
  --dataset_sampling_weights 0.7 0.3 \
  --batch_size 2048 --lr 0.001 --max_epoch 200
```

### Using the Example Script

```bash
# Run the provided example
python multi_dataset_example.py

# Or customize the datasets
python multi_dataset_example.py --multi_datasets dataset1 dataset2 dataset3
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--multi_datasets` | List[str] | None | List of dataset names to train on |
| `--multi_datasets_path` | str | None | Base path for datasets (uses `data_path` if not specified) |
| `--proportional_sampling` | bool | True | Use proportional sampling based on dataset sizes |
| `--dataset_sampling_weights` | List[float] | None | Custom weights for dataset sampling |

## Technical Implementation

### Dataset Loading Process

1. **Individual Dataset Loading**: Each dataset is loaded separately using existing helper functions
2. **ID Remapping**: Users and items are assigned new IDs with offsets to avoid conflicts
3. **Metadata Storage**: Dataset information is stored for later use in negative sampling
4. **Graph Construction**: A unified interaction graph is built from all datasets

### Negative Sampling Strategy

The negative sampling ensures that:
- For a user from dataset A, negative items are only sampled from dataset A
- The original dataset boundaries are preserved
- No cross-dataset negative sampling occurs

```python
# Example: User 5 from dataset 'amazon_movie' 
# Negative items will only be sampled from 'amazon_movie' items
user_dataset = get_user_dataset(user_id=5)  # Returns 'amazon_movie'
dataset_items = get_dataset_items('amazon_movie')  # Only movie items
neg_items = sample_negatives(dataset_items, exclude=user_positives)
```

### Proportional Sampling

When proportional sampling is enabled, the probability of sampling from each dataset is:

```
P(dataset_i) = interactions_i / total_interactions
```

Where:
- `interactions_i` is the number of interactions in dataset i
- `total_interactions` is the sum of interactions across all datasets

## Dataset Structure Requirements

Each dataset should follow the standard AlphaRec format:

```
data/General/
├── dataset1/
│   └── cf_data/
│       ├── train.txt
│       ├── valid.txt
│       └── test.txt
├── dataset2/
│   └── cf_data/
│       ├── train.txt
│       ├── valid.txt
│       └── test.txt
└── ...
```

File format (space-separated):
```
user_id item_id1 item_id2 item_id3 ...
```

## Model Compatibility

The multi-dataset training is compatible with all existing models:

- **Matrix Factorization (MF)**
- **LightGCN**
- **SGL** (Self-supervised Graph Learning)
- **XSimGCL**
- **MultVAE**
- **TFCE/TFCEMLP**
- **AlphaRec**

## Performance Considerations

### Memory Usage
- Memory usage increases proportionally with the number of datasets
- Each dataset's interaction graph is loaded into memory
- Consider reducing batch size for very large multi-dataset configurations

### Training Time
- Training time depends on the total number of interactions across all datasets
- Proportional sampling ensures balanced training across datasets
- Use GPU acceleration when available

### Scalability
- The implementation scales well with the number of datasets
- Tested with up to 10 datasets simultaneously
- For very large numbers of datasets, consider batch processing

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or use fewer datasets
2. **Dataset Path Issues**: Ensure `multi_datasets_path` is correct
3. **Weight Mismatches**: Number of weights must match number of datasets
4. **Missing Files**: Verify all datasets have train/valid/test files

### Debugging Tips

1. **Enable Verbose Output**: Use `--verbose 1` to see detailed information
2. **Check Dataset Info**: The system prints dataset statistics during loading
3. **Verify Negative Sampling**: Check that negative items are from the correct dataset
4. **Monitor Memory Usage**: Use system monitoring tools during training

## Future Enhancements

Potential improvements to consider:

1. **Dynamic Dataset Loading**: Load datasets on-demand to reduce memory usage
2. **Cross-Dataset Evaluation**: Evaluate model performance across different datasets
3. **Adaptive Sampling**: Adjust sampling weights based on training progress
4. **Distributed Training**: Support for distributed multi-dataset training

## References

- Original AlphaRec paper and implementation
- LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
- Neural Collaborative Filtering literature

## Contributing

When contributing to the multi-dataset functionality:

1. Ensure backward compatibility with single-dataset training
2. Add appropriate tests for new features
3. Update documentation for any API changes
4. Consider performance implications of changes

---

For questions or issues related to multi-dataset training, please open an issue in the repository.
# Batch Ensemble Implementation for TFCEMLP

## Overview
Added batch ensemble training capability to TFCEMLP.py, which allows training multiple ensemble members with shared weights over MLP instead of simple MLPs.

## Changes Made

### 1. Argument Parser (`parse.py`)
Added two new arguments in the TFCE section:
- `--is_batch_ensemble`: Boolean flag to enable batch ensemble training (default: False)
- `--n_ensemble_members`: Number of ensemble members to use (default: 4)

### 2. TFCEMLP Model (`models/General/TFCEMLP.py`)

#### New Instance Variables:
- `self.is_batch_ensemble`: Controls whether to use batch ensemble
- `self.n_ensemble_members`: Number of ensemble members (configurable via args)

#### Architecture Changes:
When `--is_batch_ensemble` is enabled:

**For 'homo' model version (linear mapping):**
- Creates multiple linear heads (`mlp_heads` and `mlp_user_heads`) instead of single MLPs
- Each head is a separate `nn.Linear` layer

**For 'mlp' model version:**
- **Shared layers**: `mlp_shared` and `mlp_user_shared` (common feature extraction)
- **Multiple heads**: `mlp_heads` and `mlp_user_heads` (separate final prediction layers)

#### Computation Changes:
The `compute()` method now:
1. Applies shared layers (for MLP version) or directly processes inputs (for homo version)
2. Passes through each ensemble head
3. Averages the outputs from all ensemble members using `torch.stack().mean(dim=0)`

## Usage

To enable batch ensemble training:
```bash
python main.py --model_name TFCEMLP --is_batch_ensemble --n_ensemble_members 4
```

To use more ensemble members:
```bash
python main.py --model_name TFCEMLP --is_batch_ensemble --n_ensemble_members 8
```

## Technical Details

### Batch Ensemble Benefits:
1. **Improved robustness**: Multiple prediction heads reduce overfitting
2. **Better uncertainty estimation**: Ensemble averaging provides more stable predictions
3. **Shared computation**: Lower layers are shared, making it computationally efficient
4. **Easy integration**: Minimal changes to existing training pipeline

### Architecture:
- **Shared weights**: Early layers (feature extraction) are shared across ensemble members
- **Individual heads**: Final prediction layers are separate for each ensemble member
- **Ensemble prediction**: Average of all member predictions

### Memory & Computation:
- Memory increase: Only the final layers are multiplied by the number of ensemble members
- Computation increase: Minimal overhead since only the final forward passes are replicated
- Training: Standard backpropagation works seamlessly with the ensemble structure

## Backward Compatibility
The implementation maintains full backward compatibility. When `--is_batch_ensemble` is not specified or set to False, the model behaves exactly as before.
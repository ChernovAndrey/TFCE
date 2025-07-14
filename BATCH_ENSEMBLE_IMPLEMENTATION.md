# BatchEnsemble Implementation for TFCEMLP

## Overview
Added BatchEnsemble training capability to TFCEMLP.py following the method from Wen et al. (ICLR 2020). This implementation uses rank-1 adaptation with shared weights instead of multiple separate MLPs, providing efficient ensemble training with reduced memory and computational overhead.

## Changes Made

### 1. Argument Parser (`parse.py`)
Added two new arguments in the TFCE section:
- `--is_batch_ensemble`: Boolean flag to enable BatchEnsemble training (default: False)
- `--n_ensemble_members`: Number of ensemble members to use (default: 4)

### 2. TFCEMLP Model (`models/General/TFCEMLP.py`)

#### New Instance Variables:
- `self.is_batch_ensemble`: Controls whether to use BatchEnsemble
- `self.n_ensemble_members`: Number of ensemble members (configurable via args)

#### Architecture Changes (BatchEnsemble Method):
When `--is_batch_ensemble` is enabled, the model implements the BatchEnsemble approach where each weight matrix W_i is defined as:

**W_i = W ⊙ (r_i ⊗ s_i^T)**

Where:
- W is the shared weight matrix
- ⊙ is the Hadamard product
- r_i and s_i are rank-1 adaptation vectors for ensemble member i
- ⊗ denotes the outer product

**For 'homo' model version (linear mapping):**
- **Shared weights**: `mlp_shared_weight`, `mlp_user_shared_weight`
- **Rank-1 vectors**: `mlp_r_vectors`, `mlp_s_vectors`, `mlp_user_r_vectors`, `mlp_user_s_vectors`
- Each ensemble member gets its own r_i and s_i vectors

**For 'mlp' model version:**
- **Layer 1 shared weights**: `mlp_layer1_shared_weight`, `mlp_user_layer1_shared_weight`
- **Layer 1 rank-1 vectors**: `mlp_layer1_r_vectors`, `mlp_layer1_s_vectors`, etc.
- **Layer 2 shared weights**: `mlp_layer2_shared_weight`, `mlp_user_layer2_shared_weight`
- **Layer 2 rank-1 vectors**: `mlp_layer2_r_vectors`, `mlp_layer2_s_vectors`, etc.

#### Computation Changes:
The `compute()` method now:
1. For each ensemble member i, creates ensemble-specific weight matrix using Hadamard product
2. Applies the ensemble-specific transformation using `F.linear()`
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

### BatchEnsemble Benefits:
1. **Memory efficiency**: Shared weights with only rank-1 adaptation vectors per ensemble member
2. **Computational efficiency**: Hadamard product operations are efficient and parallelizable
3. **Improved robustness**: Multiple ensemble members reduce overfitting
4. **Better uncertainty estimation**: Ensemble averaging provides more stable predictions
5. **Easy integration**: Minimal changes to existing training pipeline

### Architecture Details:
- **Shared weights**: All ensemble members share the same base weight matrices
- **Rank-1 adaptation**: Each member has unique r_i and s_i vectors for adaptation
- **Hadamard product**: Element-wise multiplication creates member-specific weights
- **Ensemble prediction**: Average of all member predictions

### Memory & Computation:
- Memory increase: Only 2×(input_dim + output_dim) parameters per ensemble member (for r_i and s_i)
- Computation: Sequential application of ensemble-specific weights during forward pass
- Training: Standard backpropagation works with the rank-1 parameterization
- Efficiency: Much more efficient than deep ensembles while maintaining similar performance

## Backward Compatibility
The implementation maintains full backward compatibility. When `--is_batch_ensemble` is not specified or set to False, the model behaves exactly as before.
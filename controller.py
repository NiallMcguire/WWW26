#!/usr/bin/env python3
"""
FIXED Controller for Brain Passage Retrieval
CHANGES:
1. Always passes global_eeg_dims to model creation (fixes EEG encoder initialization)
2. Adds --use_labram_patching flag for true LaBraM-style channel patching
3. Adds parameter verification after model creation
"""

import torch
import numpy as np
import random
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import json
from datetime import datetime

# Import our custom modules
from mv_dataloader import DynamicMaskingDataloader, simple_collate_fn, compute_global_eeg_dimensions
from mv_models import create_model  # Use FIXED model
from mv_training import train_model, finish_wandb


def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Set random seed to {seed}")


def create_output_directory(base_name="simple_experiment"):
    """Create timestamped output directory for experiment results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{base_name}_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    print(f"Created output directory: {output_dir}")
    return output_dir


def create_dynamic_dataloaders(data_path, tokenizer, batch_size=8, max_text_len=256,
                               max_eeg_len=50, train_ratio=0.8, debug=False,
                               num_vectors=32, dataset_type='auto', training_masking_level=90,
                               global_eeg_dims=None, split_by_subject=False):
    """Create training, validation, and test dataloaders with DYNAMIC masking support"""
    print(f"Loading data from: {data_path}")

    # Compute global EEG dimensions if not provided
    if global_eeg_dims is None:
        global_eeg_dims = compute_global_eeg_dimensions(data_path, max_eeg_len, dataset_type)
        print(f"Computed global EEG dimensions: {global_eeg_dims[0]}x{global_eeg_dims[1]}x{global_eeg_dims[2]}")

    # Convert training masking level to probability
    training_masking_prob = training_masking_level / 100.0
    print(f"Training with {training_masking_level}% masking probability")

    # Create datasets
    train_dataset = DynamicMaskingDataloader(
        data_path=data_path, tokenizer=tokenizer, max_text_len=max_text_len,
        max_eeg_len=max_eeg_len, split='train', train_ratio=train_ratio,
        debug=debug, global_eeg_dims=global_eeg_dims, num_vectors=num_vectors,
        dataset_type=dataset_type, initial_masking_probability=training_masking_prob
    )
    train_dataset.split_by_subject = split_by_subject

    val_dataset = DynamicMaskingDataloader(
        data_path=data_path, tokenizer=tokenizer, max_text_len=max_text_len,
        max_eeg_len=max_eeg_len, split='val', train_ratio=train_ratio,
        debug=debug, global_eeg_dims=global_eeg_dims, num_vectors=num_vectors,
        dataset_type=dataset_type, initial_masking_probability=training_masking_prob
    )
    val_dataset.split_by_subject = split_by_subject

    test_dataset = DynamicMaskingDataloader(
        data_path=data_path, tokenizer=tokenizer, max_text_len=max_text_len,
        max_eeg_len=max_eeg_len, split='test', train_ratio=train_ratio,
        debug=debug, global_eeg_dims=global_eeg_dims, num_vectors=num_vectors,
        dataset_type=dataset_type, initial_masking_probability=training_masking_prob
    )
    test_dataset.split_by_subject = split_by_subject

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=simple_collate_fn, num_workers=0,
                                  pin_memory=torch.cuda.is_available())

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=simple_collate_fn, num_workers=0,
                                pin_memory=torch.cuda.is_available())

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=simple_collate_fn, num_workers=0,
                                 pin_memory=torch.cuda.is_available())

    print(f"Created training dataloader with {len(train_dataset)} samples")
    print(f"Created validation dataloader with {len(val_dataset)} samples")
    print(f"Created test dataloader with {len(test_dataset)} samples")

    return train_dataloader, val_dataloader, test_dataloader, global_eeg_dims


def verify_model_parameters(model):
    """
    Verify that all expected modules are registered and have parameters
    Returns parameter breakdown by module
    """
    print("\n=== PARAMETER VERIFICATION ===")

    param_breakdown = {}

    for name, module in model.named_modules():
        if len(list(module.parameters())) > 0 and len(list(module.children())) == 0:  # Leaf modules
            num_params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            param_breakdown[name] = {
                'total': num_params,
                'trainable': trainable_params
            }

    # Print breakdown
    print("\nParameter breakdown by module:")
    for name, counts in sorted(param_breakdown.items()):
        if counts['total'] > 0:
            trainable_pct = (counts['trainable'] / counts['total'] * 100) if counts['total'] > 0 else 0
            print(f"  {name}: {counts['total']:,} total, {counts['trainable']:,} trainable ({trainable_pct:.1f}%)")

    # Check critical modules
    critical_modules = ['eeg_encoder', 'text_encoder', 'text_projection', 'eeg_projection']
    print("\nCritical module check:")
    for module_name in critical_modules:
        if hasattr(model, module_name):
            module = getattr(model, module_name)
            if module is not None:
                num_params = sum(p.numel() for p in module.parameters())
                print(f"  ✓ {module_name}: {num_params:,} parameters")
            else:
                print(f"  ✗ {module_name}: None (not initialized!)")
        else:
            print(f"  ✗ {module_name}: Not found")

    # Check for LaBraM components if used
    if hasattr(model, 'use_labram_patching') and model.use_labram_patching:
        labram_modules = ['channel_patcher', 'cnn_preprocessor', 'positional_embeddings']
        print("\nLaBraM component check:")
        for module_name in labram_modules:
            if hasattr(model, module_name):
                module = getattr(model, module_name)
                if module is not None:
                    num_params = sum(p.numel() for p in module.parameters())
                    print(f"  ✓ {module_name}: {num_params:,} parameters")
                else:
                    print(f"  ✗ {module_name}: None (not initialized!)")
            else:
                print(f"  ✗ {module_name}: Not found")

    return param_breakdown


def save_experiment_config(config, output_dir):
    """Save experiment configuration to JSON file"""
    config_path = output_dir / "experiment_config.json"

    # Convert non-serializable values
    serializable_config = {k: (v if isinstance(v, (str, int, float, bool, list, dict, type(None))) else str(v))
                           for k, v in config.items()}

    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    print(f"Saved experiment config to: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description='FIXED Brain Passage Retrieval with proper EEG encoder initialization')

    # Data arguments
    parser.add_argument('--data_paths', nargs='+', help='Paths to ICT pairs .npy files (can specify multiple)')
    parser.add_argument('--dataset_types', nargs='+', default=['auto'],
                        choices=['auto', 'original', 'nieuwland', 'alice', 'derco', 'narrative'],
                        help='Dataset types (one for each data_path)')

    # Model arguments
    parser.add_argument('--colbert_model_name', default='colbert-ir/colbertv2.0', help='ColBERT model name')
    parser.add_argument('--hidden_dim', type=int, default=768, help='Hidden dimension size')
    parser.add_argument('--pooling_strategy', default='max', choices=['multi', 'cls', 'max', 'mean'],
                        help='Pooling strategy')
    parser.add_argument('--encoder_type', default='dual', choices=['dual'],
                        help='Encoder type (only dual supported in fixed version)')
    parser.add_argument('--query_type', default='eeg', choices=['eeg', 'text'],
                        help='Query representation type')

    # LoRA arguments
    parser.add_argument('--no_lora', action='store_true', help='Disable LoRA adaptation')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha scaling factor')
    parser.add_argument('--use_pretrained_text', action='store_true',
                        help='Use pretrained ColBERT for text encoding')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--max_text_len', type=int, default=256, help='Max text sequence length')
    parser.add_argument('--max_eeg_len', type=int, default=50, help='Max EEG sequence length')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data for training')
    parser.add_argument('--eeg_arch', default='transformer', choices=['simple', 'complex', 'transformer'],
                        help='EEG encoder architecture')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')

    # NEW: LaBraM patching arguments
    parser.add_argument('--use_labram_patching', action='store_true',
                        help='Use LaBraM-style channel patching (each channel processed independently)')
    parser.add_argument('--labram_patch_length', type=int, default=200,
                        help='Length of temporal patches for LaBraM (default: 200 samples)')

    # Multi-masking validation arguments
    parser.add_argument('--enable_multi_masking_validation', action='store_true',
                        help='Enable validation across multiple masking levels')
    parser.add_argument('--validation_masking_levels', nargs='+', type=int,
                        default=[0, 25, 50, 75, 90, 100],
                        help='Masking percentages for validation')
    parser.add_argument('--multi_masking_frequency', type=int, default=3,
                        help='Run multi-masking validation every N epochs')
    parser.add_argument('--primary_masking_level', type=int, default=90,
                        help='Primary masking level for early stopping')
    parser.add_argument('--training_masking_level', type=int, default=90,
                        help='Masking level used during training')

    # Test evaluation arguments
    parser.add_argument('--test_masking_levels', nargs='+', type=int,
                        default=[0, 25, 50, 75, 90, 100],
                        help='Masking percentages for testing')
    parser.add_argument('--enable_test_evaluation', action='store_true',
                        help='Enable comprehensive test evaluation after training')

    # Experiment arguments
    parser.add_argument('--output_dir', default=None, help='Output directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--split_by_subject', action='store_true',
                        help='Split data by subject (out-of-subject evaluation)')

    args = parser.parse_args()

    # Validate inputs
    if not args.data_paths:
        raise ValueError("Must specify --data_paths")

    # Set seeds
    set_seeds(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir) if args.output_dir else create_output_directory("fixed_brain_retrieval")
    output_dir.mkdir(exist_ok=True)

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.colbert_model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.colbert_model_name)
    except:
        print(f"ColBERT tokenizer not found, falling back to bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    special_tokens = ['[Q]', '[D]', '[MASK]'] if '[MASK]' not in tokenizer.get_vocab() else ['[Q]', '[D]']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")

    # Create dataloaders with global_eeg_dims
    print(f"\nCreating dataloaders...")
    train_dataloader, val_dataloader, test_dataloader, global_eeg_dims = create_dynamic_dataloaders(
        data_path=args.data_paths,  # Pass list of paths
        tokenizer=tokenizer, batch_size=args.batch_size,
        max_text_len=args.max_text_len, max_eeg_len=args.max_eeg_len,
        train_ratio=args.train_ratio, debug=args.debug,
        dataset_type=args.dataset_types,  # Pass list of types
        training_masking_level=args.training_masking_level,
        split_by_subject=args.split_by_subject
    )

    print(f"\n✓ Global EEG dimensions: {global_eeg_dims}")
    print(f"  Words: {global_eeg_dims[0]}, Time: {global_eeg_dims[1]}, Channels: {global_eeg_dims[2]}")

    # Create experiment configuration
    config = {
        'experiment_type': 'fixed_brain_retrieval',
        'timestamp': datetime.now().isoformat(),
        'data_paths': [str(p) for p in args.data_paths],
        'dataset_types': args.dataset_types,
        'colbert_model_name': args.colbert_model_name,
        'hidden_dim': args.hidden_dim,
        'batch_size': args.batch_size,
        'max_text_len': args.max_text_len,
        'max_eeg_len': args.max_eeg_len,
        'train_ratio': args.train_ratio,
        'seed': args.seed,
        'device': str(device),
        'tokenizer_vocab_size': len(tokenizer),
        'train_samples': len(train_dataloader.dataset),
        'val_samples': len(val_dataloader.dataset),
        'test_samples': len(test_dataloader.dataset),
        'eeg_arch': args.eeg_arch,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'patience': args.patience,
        'use_lora': not args.no_lora,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'pooling_strategy': args.pooling_strategy,
        'encoder_type': args.encoder_type,
        'query_type': args.query_type,
        'use_pretrained_text': args.use_pretrained_text,
        'use_labram_patching': args.use_labram_patching,
        'labram_patch_length': args.labram_patch_length,
        'global_eeg_dims': list(global_eeg_dims),
        'training_masking_level': args.training_masking_level,
        'split_by_subject': args.split_by_subject
    }

    # ===== FIX: CREATE MODEL WITH GLOBAL_EEG_DIMS =====
    print(f"\n=== MODEL CREATION (FIXED) ===")
    print(f"Creating model with global_eeg_dims: {global_eeg_dims}")

    if args.use_labram_patching:
        print(
            f"⚠ WARNING: LaBraM patching will create {global_eeg_dims[2] * (global_eeg_dims[1] // args.labram_patch_length)} patches per word")
        print(f"  This is {global_eeg_dims[2]}x more expensive than standard approach!")
        print(f"  Inference will be MUCH slower. Consider using smaller models or efficient attention.")

    model = create_model(
        colbert_model_name=args.colbert_model_name,
        hidden_dim=args.hidden_dim,
        eeg_arch=args.eeg_arch,
        device=device,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        pooling_strategy=args.pooling_strategy,
        encoder_type=args.encoder_type,
        global_eeg_dims=global_eeg_dims,  # ✓ CRITICAL: Pass dimensions
        query_type=args.query_type,
        use_pretrained_text=args.use_pretrained_text,
        use_labram_patching=args.use_labram_patching,  # ✓ NEW: LaBraM patching
        labram_patch_length=args.labram_patch_length  # ✓ NEW: Patch length
    )

    if not args.use_pretrained_text:
        model.set_tokenizer_vocab_size(len(tokenizer))

    # ===== FIX: VERIFY PARAMETERS BEFORE OPTIMIZER =====
    param_breakdown = verify_model_parameters(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Model created successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable percentage: {trainable_params / total_params * 100:.1f}%")

    config.update({
        'total_params': total_params,
        'trainable_params': trainable_params,
        'param_breakdown': {k: v['total'] for k, v in param_breakdown.items()}
    })
    save_experiment_config(config, output_dir)

    # ===== CREATE OPTIMIZER AFTER MODEL IS FULLY INITIALIZED =====
    print(f"\n=== OPTIMIZER CREATION ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Verify optimizer has parameters
    optimizer_param_count = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
    print(f"✓ Optimizer created with {optimizer_param_count:,} parameters")

    if optimizer_param_count != trainable_params:
        print(
            f"⚠ WARNING: Optimizer parameter count ({optimizer_param_count:,}) != trainable params ({trainable_params:,})")
        print(f"  Some parameters may not be training!")
    else:
        print(f"✓ All trainable parameters registered in optimizer")

    # TRAIN
    print(f"\n=== TRAINING START ===")
    trained_model = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        num_epochs=args.epochs,
        patience=args.patience,
        device=device,
        debug=args.debug,
        config=config,
        enable_multi_masking_validation=args.enable_multi_masking_validation,
        multi_masking_frequency=args.multi_masking_frequency,
        validation_masking_levels=args.validation_masking_levels,
        primary_masking_level=args.primary_masking_level
    )

    # SAVE MODEL
    model_save_path = output_dir / f"fixed_model_{args.pooling_strategy}_{args.eeg_arch}.pt"
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': config,
        'tokenizer_vocab_size': len(tokenizer)
    }, model_save_path)
    print(f"Saved trained model to: {model_save_path}")

    # TEST EVALUATION
    if args.enable_test_evaluation:
        print(f"\n=== TEST SET EVALUATION ===")
        from mv_training import test_model
        test_results = test_model(
            model=trained_model,
            test_dataloader=test_dataloader,
            device=device,
            debug=args.debug,
            test_masking_levels=args.test_masking_levels,
            primary_masking_level=args.primary_masking_level
        )
        print(f"Test evaluation completed. Results logged to wandb.")

    finish_wandb()
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
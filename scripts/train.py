# train.py
"""
Main training script for TahoeFormer.

This script handles:
- Loading configuration from a YAML file.
- Setting up logging (Weights & Biases).
- Initializing the model (LitEnformerSMILES, using Morgan Fingerprints).
- Initializing dataloaders (TahoeSMILESDataset).
- Setting up PyTorch Lightning Callbacks (ModelCheckpoint, EarlyStopping, MetricLogger).
- Running the training and testing loops using PyTorch Lightning Trainer.
"""

import argparse
import yaml
import os
import torch
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split
import wandb

from pl_models import LitEnformerSMILES, MetricLogger
from datasets import TahoeSMILESDataset, ENFORMER_INPUT_SEQ_LENGTH

import warnings
warnings.filterwarnings('ignore', '.*does not have many workers.*')
warnings.filterwarnings('ignore', '.*Detecting val_dataloader.*')

# --- Default Configs --- (can be overridden by config YAML)
DEFAULT_CONFIG = {
    'data': {
        'regions_csv_path': 'data/Enformer_genomic_regions_TSSCenteredGenes_FixedOverlapRemoval.csv',
        'pbulk_parquet_path': 'data/pseudoBulk_celllineXdrug_top3k_for_testing.parquet',
        'drug_meta_csv_path': 'data/drug_metadata.csv',
        'fasta_file_path': 'data/hg38.fa',
        'enformer_input_seq_length': 196_608,
        'morgan_fp_radius': 2, # For TahoeSMILESDataset
        'morgan_fp_nbits': 2048, # For TahoeSMILESDataset
        'filter_drugs_by_ids': None,
        # Column name defaults for TahoeSMILESDataset
        'regions_gene_col': 'gene_id',
        'regions_chr_col': 'seqnames',
        'regions_start_col': 'start',
        'regions_end_col': 'end',
        'regions_strand_col': None,
        'regions_set_col': 'set', # New: Column name for train/val/test split in regions_csv
        'pbulk_gene_col': 'gene_id',
        'pbulk_dose_col': 'dose_nM',
        'pbulk_expr_col': 'value',
        'pbulk_cell_line_col': 'cell_line_id',
        'drug_meta_id_col': 'drug_id',
        'drug_meta_smiles_col': 'canonical_smiles'
    },
    'model': {
        'enformer_model_name': 'EleutherAI/enformer-official-rough',
        'enformer_target_length': -1,
        'num_output_tracks_enformer_head': 1,
        'morgan_fingerprint_dim': 2048, # For LitEnformerSMILES model
        'dose_input_dim': 1,
        'fusion_hidden_dim': 256,
        'final_output_tracks': 1,
        'learning_rate': 5e-6, 
        'loss_alpha': 1.0,
        'weight_decay': 0.01,
        'eval_gene_sets': None 
    },
    'training': {
        'batch_size': 2,
        'num_workers': 0,
        'pin_memory': False,
        'max_epochs': 50,
        'gpus': -1, # -1 for all available GPUs, or specify count e.g., 1, 2
        'accelerator': 'auto',
        'strategy': 'ddp_find_unused_parameters_true', 
        'precision': '16-mixed', # '32' or '16-mixed' or 'bf16-mixed'
        'val_check_interval': 1.0, 
        'limit_train_batches': 1.0, 
        'limit_val_batches': 1.0,   
        'limit_test_batches': 1.0,  
        'deterministic': True, 
        'seed': 42
    },
    'logging': {
        'wandb_project': 'TahoeformerDebug',
        'wandb_entity': None, # W&B info (username or team)
        'save_dir': 'outputs/model_checkpoints',
        'checkpoint_monitor_metric': 'validation_pearson_epoch',
        'checkpoint_monitor_mode': 'max',
        'save_top_k': 1,
        'early_stopping_metric': 'validation_pearson_epoch',
        'early_stopping_mode': 'max',
        'early_stopping_patience': 10
    }
}

def delete_checkpoint_at_end(trainer):
    """ Delete checkpoint after training and testing if desired """
    checkpoint_callbacks = [cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)]
    if checkpoint_callbacks:
        checkpoint_callback = checkpoint_callbacks[0]
        if hasattr(checkpoint_callback, 'best_model_path') and checkpoint_callback.best_model_path and os.path.exists(checkpoint_callback.best_model_path):
            print(f"Deleting best checkpoint: {checkpoint_callback.best_model_path}")
            os.remove(checkpoint_callback.best_model_path)
        else:
            print("No best model path found to delete or path does not exist.")
    else:
        print("No ModelCheckpoint callback found.")

def parse_optional_gene_list(filepath):
    """ Parses a file containing one gene name per row, returns a list. Returns empty list if path is None or invalid. """
    if filepath is None or not os.path.exists(filepath):
        return []
    gene_list = []
    with open(filepath, 'r') as file:
        for gene in file:
            gene_list.append(gene.strip())
    return gene_list

def load_config(config_path=None):
    """Loads configuration from YAML, merging with defaults. Ensures deep copy of defaults."""
    # Manual deep copy for 2 levels, as DEFAULT_CONFIG is structured
    config = {}
    for k, v in DEFAULT_CONFIG.items():
        if isinstance(v, dict):
            config[k] = v.copy() # Copies the inner dictionary
        else:
            config[k] = v

    if config_path:
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        if user_config: # Ensure user_config is not None (e.g. if YAML is empty)
            for key, value in user_config.items():
                if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                    config[key].update(value) # Merge level 2 dicts
                else:
                    config[key] = value # Overwrite or add new keys/values
    return config

def build_model(config):
    """
    Builds the LitEnformerSMILES model using Morgan Fingerprints.
    Model parameters are sourced from the 'model' section of the config.
    """
    model_params = config['model']
    # Ensure morgan_fingerprint_dim from data config (for dataset) matches model config
    # Model will use its own `morgan_fingerprint_dim` parameter.
    # The dataset's `morgan_fp_nbits` should align with this.
    print(f"Building LitEnformerSMILES model with morgan_fingerprint_dim: {model_params.get('morgan_fingerprint_dim')}")

    return LitEnformerSMILES(
        enformer_model_name=model_params.get('enformer_model_name'),
        enformer_target_length=model_params.get('enformer_target_length'),
        num_output_tracks_enformer_head=model_params.get('num_output_tracks_enformer_head'),
        morgan_fingerprint_dim=model_params.get('morgan_fingerprint_dim', 2048), # Default from model if not in config
        dose_input_dim=model_params.get('dose_input_dim'),
        fusion_hidden_dim=model_params.get('fusion_hidden_dim'),
        final_output_tracks=model_params.get('final_output_tracks'),
        learning_rate=model_params.get('learning_rate'),
        loss_alpha=model_params.get('loss_alpha'),
        weight_decay=model_params.get('weight_decay'),
        eval_gene_sets=model_params.get('eval_gene_sets')
    )

def load_tahoe_smiles_dataloaders(config):
    """
    Initializes TahoeSMILESDataset (now using Morgan Fingerprints) and creates DataLoaders.
    Dataset parameters are sourced from the 'data' section of the config.
    Training parameters (batch_size, num_workers) from 'training' section.
    """
    data_config = config['data']
    train_config = config['training']

    # Pass Morgan fingerprint params to TahoeSMILESDataset
    dataset_args = {
        'regions_csv_path': data_config['regions_csv_path'],
        'pbulk_parquet_path': data_config['pbulk_parquet_path'],
        'drug_meta_csv_path': data_config['drug_meta_csv_path'],
        'fasta_file_path': data_config['fasta_file_path'],
        'enformer_input_seq_length': data_config.get('enformer_input_seq_length'),
        'morgan_fp_radius': data_config.get('morgan_fp_radius', 2),
        'morgan_fp_nbits': data_config.get('morgan_fp_nbits', 2048),
        'filter_drugs_by_ids': data_config.get('filter_drugs_by_ids'),
        # Pass column name configurations
        'regions_gene_col': data_config.get('regions_gene_col', 'gene_name'),
        'regions_chr_col': data_config.get('regions_chr_col', 'seqnames'),
        'regions_start_col': data_config.get('regions_start_col', 'starts'),
        'regions_end_col': data_config.get('regions_end_col', 'ends'),
        'regions_strand_col': data_config.get('regions_strand_col', None),
        'regions_set_col': data_config.get('regions_set_col', 'set'), # Added for set-based splitting
        'pbulk_gene_col': data_config.get('pbulk_gene_col', 'gene_id'),
        'pbulk_dose_col': data_config.get('pbulk_dose_col', 'dose_nM'),
        'pbulk_expr_col': data_config.get('pbulk_expr_col', 'value'),
        'pbulk_cell_line_col': data_config.get('pbulk_cell_line_col', 'cell_line_id'),
        'drug_meta_id_col': data_config.get('drug_meta_id_col', 'drug_id'),
        'drug_meta_smiles_col': data_config.get('drug_meta_smiles_col', 'canonical_smiles')
    }
    
    # print(f"Initializing TahoeSMILESDataset with morgan_fp_nbits: {dataset_args['morgan_fp_nbits']}")

    # Instantiate dataset for each split using the 'target_set' parameter
    print("Initializing train dataset...")
    train_dataset = TahoeSMILESDataset(**dataset_args, target_set='train')
    print("Initializing validation dataset...")
    val_dataset = TahoeSMILESDataset(**dataset_args, target_set='valid')
    # In the original Enformer_genomic_regions_TSSCenteredGenes_FixedOverlapRemoval.csv, 
    # the validation set is often named 'valid'. If it's 'validation' in your file, adjust accordingly.
    print("Initializing test dataset...")
    test_dataset = TahoeSMILESDataset(**dataset_args, target_set='test')

    # Check if datasets are potentially empty
    if len(train_dataset) == 0:
        print("WARNING: Train dataset is empty. This could be due to filtering by set='train' or other data issues. Training might fail or be skipped.")
    if len(val_dataset) == 0:
        print("WARNING: Validation dataset is empty (set='valid'). Validation loop will likely be skipped.")
    if len(test_dataset) == 0:
        print("WARNING: Test dataset is empty (set='test'). Testing loop will likely be skipped.")

    # Original logic for empty full_dataset is no longer directly applicable
    # We proceed hoping at least train_dataset has data.
    # PyTorch Lightning handles empty val_loader or test_loader gracefully.

    # Removed random_split logic
    # train_size = int(0.8 * len(full_dataset))
    # val_size = int(0.1 * len(full_dataset))
    # test_size = len(full_dataset) - train_size - val_size
    
    # # Ensure at least one sample in each split if dataset is very small
    # if train_size == 0 and len(full_dataset) > 0: train_size = 1
    # if val_size == 0 and len(full_dataset) > train_size : val_size = 1
    # if test_size == 0 and len(full_dataset) > train_size + val_size: test_size = 1
    
    # # Adjust sizes to sum up to len(full_dataset) if rounding caused issues
    # if train_size + val_size + test_size != len(full_dataset):
    #     train_size = len(full_dataset) - val_size - test_size

    # if train_size <=0 or val_size <=0 : # Test size can be 0 if not testing here
    #     print(f"Warning: Dataset too small for a meaningful split (total: {len(full_dataset)}). Train size: {train_size}, Val size: {val_size}. This might lead to issues.")
    #     # Fallback if dataset is extremely small to avoid crash, though training might not be meaningful.
    #     if len(full_dataset) > 0:
    #         train_dataset = full_dataset
    #         val_dataset = full_dataset # Using full for val too in this extreme case
    #         test_dataset = full_dataset
    #     else:
    #         raise ValueError("Dataset is empty, cannot create dataloaders.")
    # else:
    #     train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    #         full_dataset, 
    #         [train_size, val_size, test_size],
    #         generator=torch.Generator().manual_seed(config['training'].get('seed', 42))
    #     )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get('batch_size', 2),
        shuffle=True,
        num_workers=train_config.get('num_workers', 0),
        pin_memory=train_config.get('pin_memory', False),
        drop_last=True # Important for DDP and BatchNorm if batches can be size 1 per GPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.get('batch_size', 2) * 2, # Often use larger batch for val
        shuffle=False,
        num_workers=train_config.get('num_workers', 0),
        pin_memory=train_config.get('pin_memory', False)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config.get('batch_size', 2) * 2,
        shuffle=False,
        num_workers=train_config.get('num_workers', 0),
        pin_memory=train_config.get('pin_memory', False)
    )
    return train_loader, val_loader, test_loader

def load_trainer_and_callbacks(config, experiment_name_for_wandb, run_name_for_wandb):
    """ Loads PyTorch Lightning Trainer and associated callbacks. """
    

    metric_logger = MetricLogger(save_dir_prefix=os.path.join(config['logging']['save_dir'], "metrics"))


    checkpoint_dir = os.path.join(config['logging']['save_dir'], 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    monitor_metric = config['logging']['checkpoint_monitor_metric'] # MetricLogger logs with _epoch suffix
    monitor_mode = config['logging']['checkpoint_monitor_mode']
    save_top_k = config['logging'].get('save_top_k', 1)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{{epoch}}-{{{monitor_metric}:.4f}}",
        save_top_k=save_top_k,
        monitor=monitor_metric,
        mode=monitor_mode
    )


    early_stop_monitor_metric = config['logging']['early_stopping_metric']
    early_stop_mode = config['logging']['early_stopping_mode']
    min_delta = 0.001
    patience = config['logging']['early_stopping_patience']

    early_stopping_callback = EarlyStopping(
        monitor=early_stop_monitor_metric,
        min_delta=min_delta,
        patience=patience,
        verbose=True,
        mode=early_stop_mode
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = [checkpoint_callback, metric_logger, early_stopping_callback, lr_monitor]


    wandb_logger = None
    if config['logging'].get('wandb_project'):
        wandb_logger = WandbLogger(
            name=run_name_for_wandb,
            project=config['logging']['wandb_project'],
            group=experiment_name_for_wandb,
            config=config, # Log the entire config dictionary
            save_dir=config['logging']['save_dir'], # Optional: ensure logger saves to the same base dir
            id=run_name_for_wandb # Use the unique run_name as the W&B run ID
        )

    trainer = Trainer(
        max_epochs=config['training']['max_epochs'],
        precision=config['training']['precision'],
        accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
        gradient_clip_val=config['training'].get('gradient_clip_val', 0.5),
        callbacks=callbacks,
        logger=wandb_logger, # Use the configured logger
        num_sanity_val_steps=config['training'].get('num_sanity_val_steps', 0), # Often 0 if val metrics are complex
        log_every_n_steps=config['training'].get('log_every_n_steps', 50),
        check_val_every_n_epoch=config['training'].get('check_val_every_n_epoch', 1),
        deterministic=config['training']['deterministic'], # For reproducibility
        strategy=config['training']['strategy'],
        accelerator=config['training']['accelerator'],
        devices=config['training'].get('gpus', 'auto')
    )
    

    if config['training'].get('accumulate_grad_batches', 1) > 1:
        effective_batch_size = config['training']['batch_size'] * config['training'].get('accumulate_grad_batches', 1)
        print(f"Gradient Accumulation: Effective batch size will be {effective_batch_size}")
        # Log to wandb config if logger is active and wandb.run exists
        if wandb_logger and wandb.run:
             wandb.config.update({'effective_train_batch_size': effective_batch_size}, allow_val_change=True)

    return trainer

def run_experiment(config: wandb.config): 
    """ Main training and evaluation loop. """
    print("Starting experiment with configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")


    train_loader, val_loader, test_loader = load_tahoe_smiles_dataloaders(config)


    eval_gene_sets = {
        'train_eval_set': parse_optional_gene_list(config.get('eval_train_gene_path')),
        'valid_eval_set': parse_optional_gene_list(config.get('eval_valid_gene_path')),
        'test_eval_set': parse_optional_gene_list(config.get('eval_test_gene_path'))
    }
    eval_gene_sets = {k: v for k, v in eval_gene_sets.items() if v} # Keep only non-empty lists


    model = build_model(config)


    experiment_name = config.get('experiment_name', 'DefaultExperiment')
    run_name = config.get('run_name', f"{experiment_name}_default_run_id") # Fallback run_name

    trainer = load_trainer_and_callbacks(config, experiment_name, run_name)


    if config.get('validate_before_train', False) and val_loader.dataset:
        print("Running pre-training validation loop...")
        trainer.validate(model, dataloaders=val_loader)


    print("Starting training...")
    if train_loader.dataset:
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader if val_loader.dataset else None
        )
    else:
        print("Skipping training as train_loader is empty.")


    print("Starting testing...")
    if test_loader.dataset:
         
        best_model_path = trainer.checkpoint_callback.best_model_path if hasattr(trainer.checkpoint_callback, 'best_model_path') else None
        if best_model_path and os.path.exists(best_model_path):
            print(f"Loading best model for testing from: {best_model_path}")
            trainer.test(model, dataloaders=test_loader, ckpt_path=best_model_path)
        elif not best_model_path:
             print("No best_model_path found from checkpoint callback. Testing with current model state (if any training happened).")
             trainer.test(model, dataloaders=test_loader) # Test with current model if no checkpoint or if training was skipped
        else: # path exists but is false for some reason or doesnt exist
            print(f"Best model path {best_model_path} not found. Testing with current model state.")
            trainer.test(model, dataloaders=test_loader)
    else:
        print("Skipping testing as test_loader is empty.")

    if config.get('delete_checkpoint_after_run', False):
        delete_checkpoint_at_end(trainer)


def main():
    parser = argparse.ArgumentParser(description='Run PyTorch Lightning Enformer-SMILES experiment.')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the YAML configuration file.')
    # Allow seed override from command line, though config is primary source
    parser.add_argument("--seed", type=int, help="Override seed from config file.") 
    args = parser.parse_args()


    effective_config = load_config(args.config_path)
    

    if args.seed is not None:
        # Ensure 'training' key exists if seed is to be put there
        if 'training' not in effective_config:
            effective_config['training'] = {}
        effective_config['training']['seed'] = args.seed
    

    seed = effective_config.get('training', {}).get('seed', 42)
    pl.seed_everything(seed, workers=True)
    

    if effective_config.get('training', {}).get('deterministic', False):
         torch.use_deterministic_algorithms(True, warn_only=True) # Ensure deterministic ops if requested

    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    experiment_name = effective_config.get('experiment_name', 'EnformerSMILESExperiment')
    

    run_name = f"{experiment_name}_Seed-{seed}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    effective_config['run_name'] = run_name
    effective_config['experiment_name'] = experiment_name # Ensure experiment_name is also in config


    default_save_dir = os.path.join(current_script_dir, f"../results/{experiment_name}/{run_name}")
    
    if 'logging' not in effective_config:
        effective_config['logging'] = {}
    effective_config['logging']['save_dir'] = effective_config.get('logging', {}).get('save_dir', default_save_dir)
    os.makedirs(effective_config['logging']['save_dir'], exist_ok=True)
    print(f"Results and checkpoints will be saved in: {effective_config['logging']['save_dir']}")

 
    run_experiment(effective_config) # Pass the fully populated effective_config

    if effective_config.get('logging', {}).get('wandb_project') and wandb.run is not None:
        wandb.finish()

if __name__ == '__main__':
    main()

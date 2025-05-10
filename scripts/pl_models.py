# pl_models.py
"""
This module defines PyTorch Lightning modules for the Tahoeformer project.
It includes a base model class (`LitBaseModel`) and the main experimental model
(`LitEnformerSMILES`) which combines an Enformer-based DNA sequence model with
drug information (SMILES string processed into Morgan Fingerprints) and dose information 
to predict gene expression.

Key components:
- masked_mse: A utility loss function for Mean Squared Error that handles NaN targets.
- LitBaseModel: A base LightningModule providing common training, validation, test steps,
  optimizer configuration, and basic metric logging hooks.
- LitEnformerSMILES: The primary model for predicting drug-induced gene expression changes,
  using Enformer for DNA and Morgan fingerprints for drugs.
- MetricLogger: A PyTorch Lightning Callback for detailed logging of predictions.
"""

import pandas as pd
import os
import torch
import torch.nn as nn
import lightning.pytorch as pl
from enformer_pytorch.finetune import HeadAdapterWrapper
from enformer_pytorch import Enformer
from torchmetrics.regression import PearsonCorrCoef, R2Score
from warnings import warn
import wandb
import numpy as np # Added for MetricLogger consistency

# --- Utility Functions ---
def masked_mse(y_hat, y):
    """
    Computes Mean Squared Error (MSE) while ignoring NaN values in the target tensor.

    Args:
        y_hat (torch.Tensor): The predicted values.
        y (torch.Tensor): The target values, which may contain NaNs.

    Returns:
        torch.Tensor: A scalar tensor representing the masked MSE. Returns 0.0 if all targets are NaN.
    """
    mask = torch.isnan(y)
    if mask.all(): # Handle case where all targets in batch are NaN
        return torch.tensor(0.0, device=y_hat.device, requires_grad=True)
    mse = torch.mean((y[~mask] - y_hat[~mask])**2)
    return mse

# --- Base Lightning Module ---
class LitBaseModel(pl.LightningModule):
    """
    A base PyTorch Lightning module providing common boilerplate for training and evaluation.

    This class implements a generic training/validation/test step, loss calculation using
    `masked_mse`, optimizer configuration (AdamW), and hooks for accumulating outputs
    for detailed metric logging via the `MetricLogger` callback.

    Derived classes are expected to implement the `forward` method.

    Hyperparameters:
        learning_rate (float): The learning rate for the optimizer.
        loss_alpha (float): A coefficient for the primary loss term (MSE). Useful if 
                            additional loss terms were to be added.
        weight_decay (float, optional): Weight decay for the AdamW optimizer. If None,
                                        AdamW's internal default is used.
        eval_gene_sets (dict, optional): A dictionary where keys are set names (e.g., 'oncogenes')
                                         and values are lists of gene IDs. Used by `MetricLogger`
                                         to compute metrics for specific gene subsets.
    """
    def __init__(self, learning_rate=5e-6, loss_alpha=1.0, weight_decay=None,
                 eval_gene_sets=None): # eval_gene_sets: dict {'train': [genes], 'valid': [genes], 'test': [genes]}
        """
        Initializes the LitBaseModel.

        Args:
            learning_rate (float, optional): Learning rate. Defaults to 5e-6.
            loss_alpha (float, optional): Alpha for MSE loss. Defaults to 1.0.
            weight_decay (float, optional): Weight decay for AdamW. If None, uses optimizer default.
                                          Defaults to None.
            eval_gene_sets (dict, optional): Dictionary of gene sets for targeted evaluation.
                                           Keys are names, values are lists of gene IDs.
                                           Defaults to None.
        """
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.loss_alpha = loss_alpha # alpha for mse vs. other terms (if any)
        self.weight_decay = weight_decay
        self.eval_gene_sets = eval_gene_sets if eval_gene_sets else {}

        # Results accumulated per epoch for MetricLogger
        self.epoch_outputs = []

    def loss_fn(self, y_hat, y):
        """
        Calculates the loss for the model.

        Currently uses `masked_mse` scaled by `self.loss_alpha`.

        Args:
            y_hat (torch.Tensor): Predicted values from the model.
            y (torch.Tensor): Ground truth target values.

        Returns:
            torch.Tensor: The computed loss value.
        """
        mse_term = masked_mse(y_hat, y)
        # Potentially: add other loss terms here, weighted by (1-loss_alpha) if desired
        return self.loss_alpha * mse_term

    def _common_step(self, batch, batch_idx, step_type):
        """
        A common step for training, validation, and testing.

        This method unpacks the batch, performs a forward pass, calculates the loss,
        logs the loss, and accumulates outputs for epoch-level metric calculation
        (for validation and test steps).

        Args:
            batch: The batch of data from the DataLoader. Expected to be a tuple containing
                   DNA sequence, Morgan fingerprints, dose, target expression, 
                   and metadata (gene_id, drug_id, cell_line).
            batch_idx (int): The index of the current batch.
            step_type (str): A string indicating the type of step ('train', 'val', or 'test').

        Returns:
            torch.Tensor: The loss for the current batch.
        """
        # Batch structure will change after dataset modification:
        # (dna_seq, morgan_fingerprints, dose, target_expression, gene_id, drug_id, cell_line)
        dna_seq, morgan_fingerprints, dose, target_expression, gene_id, drug_id, cell_line = batch
        
        y_hat = self(dna_seq, morgan_fingerprints, dose) # Call forward method of derived class

        loss = self.loss_fn(y_hat, target_expression)
        self.log(f'{step_type}_loss', loss, batch_size=target_expression.shape[0], on_step=(step_type=='train'), on_epoch=True, prog_bar=(step_type!='train'))
        
        if step_type != 'train':
            # Prepare data for MetricLogger
            batch_size = target_expression.shape[0]
            for i in range(batch_size):
                item_data = {
                    'pred': y_hat[i].detach(),
                    'target': target_expression[i].detach(),
                    'gene_id': gene_id[i], 
                    'drug_id': drug_id[i], 
                    'cell_line': cell_line[i], 
                    'rank': self.trainer.global_rank 
                }
                self.epoch_outputs.append(item_data)
        return loss

    def training_step(self, batch, batch_idx):
        """PyTorch Lightning training step. Calls `_common_step`."""
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """PyTorch Lightning validation step. Calls `_common_step`."""
        return self._common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """PyTorch Lightning test step. Calls `_common_step`."""
        return self._common_step(batch, batch_idx, 'test')
    
    def on_validation_epoch_start(self):
        """Clears accumulated outputs at the start of each validation epoch."""
        self.epoch_outputs = []
    
    def on_test_epoch_start(self):
        """Clears accumulated outputs at the start of each test epoch."""
        self.epoch_outputs = []

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.

        Uses AdamW with the specified learning rate and weight decay.

        Returns:
            torch.optim.Optimizer: The configured AdamW optimizer.
        """
        if self.weight_decay is None:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

# --- Enformer + Morgan Fingerprints Model ---
class LitEnformerSMILES(LitBaseModel): # Consider renaming to LitEnformerMorgan for clarity
    """
    A PyTorch Lightning module that combines genomic sequence information (via Enformer)
    with drug chemical structure (represented by Morgan fingerprints) and drug dose
    to predict gene expression changes.

    The model architecture consists of three main branches:
    1. DNA Module: Uses a pre-trained Enformer model (with an adapted head) to extract
       features from a one-hot encoded DNA sequence centered around a gene's TSS.
    2. Drug Module: Uses pre-computed Morgan fingerprints as the drug representation.
    3. Dose Module: Directly uses the numerical dose value.

    Features from these three branches are concatenated and passed through a multi-layer
    fusion head (MLP with ReLU, BatchNorm, Dropout) to produce the final prediction
    of gene expression.

    Inherits common training and evaluation logic from `LitBaseModel`.
    """
    def __init__(self,
                 enformer_model_name: str = 'EleutherAI/enformer-official-rough',
                 enformer_target_length: int = -1,
                 num_output_tracks_enformer_head: int = 1, 
                 morgan_fingerprint_dim: int = 2048, # dim of the Morgan fingerprint vector
                 dose_input_dim: int = 1, 
                 fusion_hidden_dim: int = 256, 
                 final_output_tracks: int = 1, 
                 learning_rate=5e-6, 
                 loss_alpha=1.0, 
                 weight_decay=None,
                 eval_gene_sets=None):
        """
        Initializes the LitEnformerSMILES (or LitEnformerMorgan) model.

        Args:
            enformer_model_name (str, optional): Name or path of the pre-trained Enformer model.
            enformer_target_length (int, optional): Target length for Enformer's internal pooling.
            num_output_tracks_enformer_head (int, optional): Output features from Enformer head.
            morgan_fingerprint_dim (int, optional): Dimensionality of the Morgan fingerprint vector
                                                   (e.g., 2048 for ECFP4). Defaults to 2048.
            dose_input_dim (int, optional): Dimensionality of the drug dose input. Defaults to 1.
            fusion_hidden_dim (int, optional): Hidden dimension for the fusion MLP. Defaults to 256.
            final_output_tracks (int, optional): Number of final output values. Defaults to 1.
            learning_rate (float, optional): Learning rate. Defaults to 5e-6.
            loss_alpha (float, optional): Weight for MSE loss. Defaults to 1.0.
            weight_decay (float, optional): Weight decay. Defaults to None.
            eval_gene_sets (dict, optional): Gene sets for targeted evaluation. Defaults to None.
        """
        super().__init__(learning_rate, loss_alpha, weight_decay, eval_gene_sets)
        self.save_hyperparameters(
            "enformer_model_name", "enformer_target_length", 
            "num_output_tracks_enformer_head", "morgan_fingerprint_dim",
            "dose_input_dim", "fusion_hidden_dim", "final_output_tracks",
            "learning_rate", "loss_alpha", "weight_decay"
        )

        # 1. DNA Module (Enformer with HeadAdapter)
        enformer_pretrained = Enformer.from_pretrained(
            self.hparams.enformer_model_name,
            target_length=self.hparams.enformer_target_length 
        )
        self.dna_module = HeadAdapterWrapper(
            enformer=enformer_pretrained,
            num_tracks=self.hparams.num_output_tracks_enformer_head,
            post_transformer_embed=False, 
            output_activation=nn.Identity()
        )

        # 2. Drug Module (Morgan Fingerprints are provided as input directly)
        # No layers needed here as fingerprints are pre-computed.
        # The self.hparams.morgan_fingerprint_dim defines the expected input dimension.

        # 3. Fusion Head 
        # Input dimension uses morgan_fingerprint_dim
        fusion_input_dim = self.hparams.num_output_tracks_enformer_head + self.hparams.morgan_fingerprint_dim + self.hparams.dose_input_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hparams.fusion_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hparams.fusion_hidden_dim),
            nn.Dropout(0.25),
            nn.Linear(self.hparams.fusion_hidden_dim, self.hparams.fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.hparams.fusion_hidden_dim // 2),
            nn.Dropout(0.1),
            nn.Linear(self.hparams.fusion_hidden_dim // 2, self.hparams.final_output_tracks)
        )

    def forward(self, dna_seq, morgan_fingerprints, dose):
        """
        Defines the forward pass of the LitEnformerSMILES model using Morgan Fingerprints.

        Args:
            dna_seq (torch.Tensor): Batch of one-hot encoded DNA sequences.
                                    Shape: (batch_size, sequence_length, 4).
            morgan_fingerprints (torch.Tensor): Batch of pre-computed Morgan fingerprint vectors.
                                                Shape: (batch_size, morgan_fingerprint_dim).
            dose (torch.Tensor): Batch of drug dose values.
                                 Shape: (batch_size, dose_input_dim).

        Returns:
            torch.Tensor: The model's prediction. Shape: (batch_size, final_output_tracks).
        """
        # --- DNA Processing ---
        dna_out_intermediate = self.dna_module(dna_seq, freeze_enformer=False)
        center_seq_idx = dna_out_intermediate.shape[1] // 2 
        dna_features = dna_out_intermediate[:, center_seq_idx, :] 
        
        # --- Drug Processing (Morgan Fingerprints) ---
        # Morgan fingerprints are directly used as features.
        smiles_features = morgan_fingerprints # Shape: (batch_size, morgan_fingerprint_dim)

        # --- Dose Processing ---
        if dose.ndim == 1:
            dose = dose.unsqueeze(-1)
        
        # --- Feature Combination & Final Prediction ---
        combined_features = torch.cat([dna_features, smiles_features, dose], dim=1)
        prediction = self.fusion_head(combined_features)
        return prediction

# --- Metrics Logging Callback ---
class MetricLogger(pl.Callback):
    """
    A PyTorch Lightning Callback for comprehensive metric calculation and logging.

    This callback accumulates predictions and targets during validation and test epochs.
    At the end of these epochs, it:
    1. Processes the accumulated outputs into a pandas DataFrame.
    2. Saves the raw predictions and targets for the epoch to a CSV file.
    3. Logs a sample of these raw predictions as a W&B Table if WandbLogger is used.
    4. Calculates overall performance metrics (MSE, Pearson, R2) for the epoch.
    5. If `eval_gene_sets` are provided in the LightningModule, calculates metrics for these specific gene subsets.
    6. Calculates metrics per cell line if 'cell_line' information is available in the outputs.
    7. Logs all calculated metrics to the LightningModule's logger.

    Attributes:
        save_dir_prefix (str): Prefix for the directory where metric CSVs will be saved.
        current_epoch_data (list): List to accumulate dictionaries of pred/target/metadata per item.
    """
    def __init__(self, save_dir_prefix="results"):
        """
        Initializes the MetricLogger callback.

        Args:
            save_dir_prefix (str, optional): Directory prefix for saving metrics files.
                                           Defaults to "results".
        """
        super().__init__()
        self.save_dir_prefix = save_dir_prefix
        self.current_epoch_data = [] 

    def _process_epoch_outputs(self, pl_module, stage):
        """
        Processes the raw outputs collected during an epoch into a pandas DataFrame.

        Converts tensor data for 'pred' and 'target' columns to NumPy/Python native types.

        Args:
            pl_module (pl.LightningModule): The LightningModule instance.
            stage (str): The current stage (e.g., "validation", "test").

        Returns:
            pd.DataFrame: A DataFrame containing the processed epoch outputs.
                          Returns an empty DataFrame if no outputs were collected.
        """
        if not hasattr(pl_module, 'epoch_outputs') or not pl_module.epoch_outputs:
            warn(f"No outputs collected (pl_module.epoch_outputs is missing or empty) during {stage} epoch for MetricLogger.")
            return pd.DataFrame()

        df = pd.DataFrame(pl_module.epoch_outputs)
        
        for col in ['pred', 'target']:
            if col in df.columns and not df[col].empty:
                if isinstance(df[col].iloc[0], torch.Tensor):
                    df[col] = df[col].apply(lambda x: x.cpu().float().numpy().item() if x.numel() == 1 else x.cpu().float().numpy())
        return df

    def on_validation_epoch_end(self, trainer, pl_module):
        """Hook called at the end of the validation epoch."""
        if hasattr(pl_module, 'epoch_outputs') and pl_module.epoch_outputs:
            self.current_epoch_data = self._process_epoch_outputs(pl_module, "validation")
            if not self.current_epoch_data.empty:
                self._log_and_save_metrics(trainer, pl_module, "validation")
        else:
            warn("MetricLogger: pl_module.epoch_outputs not found or empty at on_validation_epoch_end.")

    def on_test_epoch_end(self, trainer, pl_module):
        """Hook called at the end of the test epoch."""
        if hasattr(pl_module, 'epoch_outputs') and pl_module.epoch_outputs:
            self.current_epoch_data = self._process_epoch_outputs(pl_module, "test")
            if not self.current_epoch_data.empty:
                self._log_and_save_metrics(trainer, pl_module, "test")
        else:
            warn("MetricLogger: pl_module.epoch_outputs not found or empty at on_test_epoch_end.")


    def _log_and_save_metrics(self, trainer, pl_module, stage):
        """
        Calculates, logs, and saves metrics for the current stage and epoch.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning Trainer instance.
            pl_module (pl.LightningModule): The LightningModule instance.
            stage (str): The current stage (e.g., "validation", "test").
        """
        epoch = trainer.current_epoch if trainer.current_epoch is not None else -1 
        save_dir = getattr(pl_module.hparams, 'save_dir', 
                           os.path.join(self.save_dir_prefix, f"run_{trainer.logger.version if trainer.logger else 'local'}"))
        os.makedirs(save_dir, exist_ok=True)

        raw_preds_path = os.path.join(save_dir, f"{stage}_predictions_epoch_{epoch}.csv")
        self.current_epoch_data.to_csv(raw_preds_path, index=False)
        
        if trainer.logger and hasattr(trainer.logger, 'experiment') and isinstance(trainer.logger.experiment, wandb.sdk.wandb_run.Run):
            try:
                trainer.logger.experiment.log({f"{stage}_raw_predictions_epoch_{epoch}": wandb.Table(dataframe=self.current_epoch_data.head(1000))})
            except Exception as e:
                warn(f"MetricLogger: Failed to log raw predictions table to W&B: {e}")

        overall_metrics = self._calculate_metrics_for_group(self.current_epoch_data, pl_module.device)
        if overall_metrics:
            pl_module.log_dict({f"{stage}_{k}_epoch": v for k, v in overall_metrics.items()}, sync_dist=True)

        if hasattr(pl_module, 'eval_gene_sets') and pl_module.eval_gene_sets and isinstance(pl_module.eval_gene_sets, dict) and 'gene_id' in self.current_epoch_data.columns:
            for split_name, gene_list in pl_module.eval_gene_sets.items():
                if not gene_list: continue
                split_df = self.current_epoch_data[self.current_epoch_data['gene_id'].isin(gene_list)]
                if not split_df.empty:
                    split_metrics = self._calculate_metrics_for_group(split_df, pl_module.device)
                    if split_metrics:
                        pl_module.log_dict({f"{stage}_{split_name}_genes_{k}_epoch": v for k, v in split_metrics.items()}, sync_dist=True)
        
        if 'cell_line' in self.current_epoch_data.columns:
            for cell_line, group_df in self.current_epoch_data.groupby('cell_line'):
                cl_metrics = self._calculate_metrics_for_group(group_df, pl_module.device)
                if cl_metrics:
                     pl_module.log_dict({f"{stage}_{cell_line}_cell_line_{k}_epoch": v for k,v in cl_metrics.items()}, sync_dist=True)


    def _calculate_metrics_for_group(self, df_group, device):
        """
        Calculates regression metrics (MSE, Pearson, R2) for a given group of predictions.

        Args:
            df_group (pd.DataFrame): DataFrame containing 'pred' and 'target' columns for the group.
            device (torch.device): The device to perform calculations on.

        Returns:
            dict: A dictionary of calculated metrics (mse, pearson, r2). Returns empty if data is insufficient.
        """
        if df_group.empty or 'pred' not in df_group.columns or 'target' not in df_group.columns:
            return {}

        preds_np = np.array(df_group['pred'].tolist(), dtype=np.float32)
        targets_np = np.array(df_group['target'].tolist(), dtype=np.float32)

        preds = torch.tensor(preds_np).to(device)
        targets = torch.tensor(targets_np).to(device)

        if preds.ndim == 1: 
            preds = preds.squeeze()
            targets = targets.squeeze()
        
        if preds.numel() == 0 or targets.numel() == 0 or preds.shape != targets.shape :
            warn(f"Skipping metrics calculation for a group due to mismatched or empty preds/targets. Pred shape: {preds.shape}, Target shape: {targets.shape}")
            return {}
        
        mse_val_tensor = masked_mse(preds.unsqueeze(-1) if preds.ndim==1 else preds, 
                                    targets.unsqueeze(-1) if targets.ndim==1 else targets)
        calculated_metrics = {'mse': mse_val_tensor.item()}

        if preds.numel() < 2: 
             warn(f"Skipping Pearson/R2 for a group with < 2 samples. Found {preds.numel()} samples. Only MSE will be reported.")
             return calculated_metrics 

        preds_for_corr = preds.squeeze()
        targets_for_corr = targets.squeeze()

        if preds_for_corr.shape != targets_for_corr.shape or preds_for_corr.ndim > 1 and preds_for_corr.shape[1] >1:
            warn(f"Skipping Pearson/R2 due to incompatible shapes after squeeze for correlation. Pred: {preds_for_corr.shape}, Target: {targets_for_corr.shape}")
            return calculated_metrics

        try:
            pearson_fn = PearsonCorrCoef().to(device) 
            pearson_val = pearson_fn(preds_for_corr, targets_for_corr)
            calculated_metrics['pearson'] = pearson_val.item()
        except Exception as e:
            warn(f"Could not compute Pearson Correlation: {e}. Preds shape: {preds_for_corr.shape}, Targets shape: {targets_for_corr.shape}")

        try:
            r2_fn = R2Score().to(device) 
            r2_val = r2_fn(preds_for_corr, targets_for_corr)
            calculated_metrics['r2'] = r2_val.item()
        except Exception as e:
            warn(f"Could not compute R2 Score: {e}. Preds shape: {preds_for_corr.shape}, Targets shape: {targets_for_corr.shape}")

        return calculated_metrics   

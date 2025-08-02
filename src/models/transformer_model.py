import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel
import math

from config import TRANSFORMER_FEATURES, N_TRIALS_OPTUNA

# --- Data Handling Helpers ---
def collate_fn(batch):
    """
    Collates a batch of (X_seq, y_seq) tuples.
    Pads sequences to the length of the longest sequence in the batch.
    Returns:
        padded_X: (batch_size, max_len, input_dim)
        padded_y: (batch_size, max_len)
        mask: (batch_size, max_len) - True for padded positions (to be ignored)
    """
    # Sort batch by sequence length (descending) for efficiency (optional but good practice)
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    
    X_seqs = [torch.tensor(item[0], dtype=torch.float32) for item in batch]
    y_seqs = [torch.tensor(item[1], dtype=torch.float32) for item in batch]
    
    # Pad sequences
    # batch_first=True makes output (batch_size, max_seq_len, feature_dim)
    padded_X = pad_sequence(X_seqs, batch_first=True, padding_value=0.0)
    padded_y = pad_sequence(y_seqs, batch_first=True, padding_value=0.0)
    
    # Create padding mask (True where value is padding_value)
    # lengths tensor
    lengths = torch.tensor([len(x) for x in X_seqs])
    max_len = padded_X.size(1)
    
    # Create mask: (batch_size, max_len)
    # mask[i, j] is True if j >= length[i]
    mask = torch.arange(max_len)[None, :] >= lengths[:, None]
    
    return padded_X, padded_y, mask

class TripDataset(Dataset):
    def __init__(self, X_seq_list, y_seq_list):
        self.X_seq_list = X_seq_list
        self.y_seq_list = y_seq_list
        
    def __len__(self):
        return len(self.X_seq_list)
        
    def __getitem__(self, idx):
        return self.X_seq_list[idx], self.y_seq_list[idx]

# --- Model Definitions ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, max_len=30000):
        super(SimpleTransformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, src, src_key_padding_mask=None):
        # src: (batch_size, seq_len, input_dim)
        # src_key_padding_mask: (batch_size, seq_len) - True indicates padding
        
        embedded = self.embedding(src) * math.sqrt(self.d_model)
        embedded = self.pos_encoder(embedded)
        
        # Pass mask to encoder
        transformer_output = self.transformer_encoder(embedded, src_key_padding_mask=src_key_padding_mask)
        
        output = self.fc_out(transformer_output) # (batch_size, seq_len, 1)
        return output.squeeze(-1) # (batch_size, seq_len)


class TransformerModel(BaseModel):
    def __init__(self, vehicle_name):
        super().__init__('Transformer', vehicle_name)
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"TransformerModel initializing on {self.device}")

    def _get_features_and_target(self, data, model_type):
        features_to_use = TRANSFORMER_FEATURES.copy()
        X_sequences = []
        y_sequences = []

        for trip_id in data['trip_id'].unique():
            trip_data = data[data['trip_id'] == trip_id].copy()
            X_seq = trip_data[features_to_use].values
            X_sequences.append(X_seq)

            if model_type == 'hybrid':
                y_seq = (trip_data['target'] - trip_data['physics_prediction']).values
            else: 
                y_seq = trip_data['target'].values
            y_sequences.append(y_seq)

        return X_sequences, y_sequences

    def _train_epoch(self, model, dataloader, criterion, optimizer):
        model.train()
        total_loss = 0
        total_samples = 0
        
        for X_batch, y_batch, mask in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            mask = mask.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(X_batch, src_key_padding_mask=mask)
            
            # Mask the output and target to ignore padding in loss calculation
            # ~mask selects valid positions (False in mask means valid data)
            valid_mask = ~mask
            loss = criterion(output[valid_mask], y_batch[valid_mask])
            
            loss.backward()
            optimizer.step()
            
            # Accumulate loss weighted by number of valid samples
            batch_samples = valid_mask.sum().item()
            total_loss += loss.item() * batch_samples
            total_samples += batch_samples
            
        return total_loss / total_samples

    def _evaluate(self, model, dataloader):
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch, mask in dataloader:
                X_batch = X_batch.to(self.device)
                mask = mask.to(self.device)
                
                output = model(X_batch, src_key_padding_mask=mask)
                
                # Extract valid predictions
                valid_mask = ~mask
                valid_preds = output[valid_mask].cpu().numpy()
                valid_targets = y_batch[valid_mask.cpu()].numpy()
                
                all_preds.extend(valid_preds)
                all_targets.extend(valid_targets)
                
        return np.sqrt(mean_squared_error(all_targets, all_preds))

    def find_best_params(self, train_data, storage_path):
        X_full_seq, y_full_seq = self._get_features_and_target(train_data, model_type='hybrid')
        
        data_pairs = list(zip(X_full_seq, y_full_seq))
        X_full_flat_for_scaler = np.vstack(X_full_seq)
        kf_indices = list(KFold(n_splits=5, shuffle=True, random_state=42).split(data_pairs))

        def objective(trial):
            params = {
                'd_model': trial.suggest_categorical('d_model', [32, 64]),
                'nhead': trial.suggest_categorical('nhead', [2, 4]),
                'num_encoder_layers': trial.suggest_int('num_encoder_layers', 1, 2),
                'dim_feedforward': trial.suggest_categorical('dim_feedforward', [128, 256]),
                'dropout': trial.suggest_float('dropout', 0.1, 0.3),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),
                'n_epochs': trial.suggest_int('n_epochs', 20, 60),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32])
            }

            fold_rmses = []
            
            for train_idx, val_idx in kf_indices:
                train_pairs = [data_pairs[i] for i in train_idx]
                val_pairs = [data_pairs[i] for i in val_idx]
                
                # Prepare scaler relative to this fold's train data
                scaler = StandardScaler()
                X_train_flat = np.vstack([p[0] for p in train_pairs])
                scaler.fit(X_train_flat)
                
                # Apply scaling
                X_train_scaled = [scaler.transform(p[0]) for p in train_pairs]
                y_train_seq = [p[1] for p in train_pairs]
                X_val_scaled = [scaler.transform(p[0]) for p in val_pairs]
                y_val_seq = [p[1] for p in val_pairs]
                
                # Create DataLoaders
                train_ds = TripDataset(X_train_scaled, y_train_seq)
                val_ds = TripDataset(X_val_scaled, y_val_seq)
                
                train_dl = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_fn)
                val_dl = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn)

                model = SimpleTransformer(
                    input_dim=len(TRANSFORMER_FEATURES),
                    d_model=params['d_model'],
                    nhead=params['nhead'],
                    num_encoder_layers=params['num_encoder_layers'],
                    dim_feedforward=params['dim_feedforward'],
                    dropout=params['dropout']
                ).to(self.device)
                
                optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
                criterion = nn.MSELoss()

                # Train
                for _ in range(params['n_epochs']):
                    self._train_epoch(model, train_dl, criterion, optimizer)
                
                # Evaluate
                fold_rmse = self._evaluate(model, val_dl)
                fold_rmses.append(fold_rmse)

            return np.mean(fold_rmses)

        study = optuna.create_study(
            direction='minimize',
            study_name=f"{self.model_name}-{self.vehicle_name}-tuning",
            storage=storage_path,
            load_if_exists=True
        )
        study.optimize(objective, n_trials=N_TRIALS_OPTUNA)
        
        print(f"Best Transformer params: {study.best_params}")
        return study.best_params

    def train(self, train_data, params, model_type='hybrid'):
        X_train_seq, y_train_seq = self._get_features_and_target(train_data, model_type)
        
        # Scale
        X_train_flat = np.vstack(X_train_seq)
        self.scaler.fit(X_train_flat)
        X_train_scaled = [self.scaler.transform(x) for x in X_train_seq]
        
        # Setup Data
        dataset = TripDataset(X_train_scaled, y_train_seq)
        batch_size = params.get('batch_size', 32)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        
        # Model
        self.model = SimpleTransformer(
            input_dim=len(TRANSFORMER_FEATURES),
            d_model=params['d_model'],
            nhead=params['nhead'],
            num_encoder_layers=params['num_encoder_layers'],
            dim_feedforward=params['dim_feedforward'],
            dropout=params['dropout']
        ).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        criterion = nn.MSELoss()
        
        n_epochs = params.get('n_epochs', 50)
        print(f"Training Transformer for {n_epochs} epochs with batch size {batch_size}...")
        
        for epoch in range(n_epochs):
            avg_loss = self._train_epoch(self.model, dataloader, criterion, optimizer)
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
                
        return self.model, self.scaler

    def predict(self, test_data, model_type='hybrid'):
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
            
        X_test_seq, _ = self._get_features_and_target(test_data, model_type)
        X_test_scaled = [self.scaler.transform(x) for x in X_test_seq]
        
        # Dummy targets for dataset (not used for prediction)
        y_dummy = [np.zeros(len(x)) for x in X_test_scaled]
        
        dataset = TripDataset(X_test_scaled, y_dummy)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
        
        self.model.eval()
        all_preds = []
        
        with torch.no_grad():
            for X_batch, _, mask in dataloader:
                X_batch = X_batch.to(self.device)
                mask = mask.to(self.device)
                
                output = self.model(X_batch, src_key_padding_mask=mask)
                
                # Unwrap the batch to individual sequences
                # We need to preserve the order and exact length of each trip
                # The dataloader batch is padded, so we use the mask to slice valid parts
                
                for i in range(X_batch.size(0)):
                    # Get mask for this sequence (True is padding)
                    # We want indices where mask is False
                    valid_len = (~mask[i]).sum().item()
                    pred_seq = output[i, :valid_len].cpu().numpy()
                    all_preds.extend(pred_seq)
                    
        predictions_flat = np.array(all_preds)
        
        if model_type == 'hybrid':
            physics_predictions_flat = test_data['physics_prediction'].values
            # Check length to ensure safety
            if len(predictions_flat) != len(physics_predictions_flat):
                print(f"Warning: Prediction length {len(predictions_flat)} != Physics length {len(physics_predictions_flat)}")
                # This could happen if dataset order got messed up. 
                # Since we didn't shuffle the loader (shuffle=False), it should be fine.
            
            final_predictions = predictions_flat + physics_predictions_flat
            return final_predictions
        else:
            return predictions_flat
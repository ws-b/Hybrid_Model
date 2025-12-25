import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel
import math

from config import TRANSFORMER_FEATURES, N_TRIALS_OPTUNA

# Positional Encoding 정의
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
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# PyTorch 트랜스포머 인코더 모델 정의
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(SimpleTransformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # TransformerEncoderLayer 정의
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True  # (batch_size, seq_len, features)
        )
        # TransformerEncoder 정의
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        # 최종 출력을 위한 Fully Connected Layer
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, src):
        # src shape: (batch_size, seq_len, input_dim)
        
        # 1. Embedding & Positional Encoding
        embedded = self.embedding(src) * math.sqrt(self.d_model) # Scaling for transformer
        embedded = self.pos_encoder(embedded)
        
        # 2. Transformer Encoder
        # PyTorch의 Transformer는 (seq_len, batch_size, d_model)을 기본으로 하지만,
        # batch_first=True로 설정했으므로 (batch_size, seq_len, d_model)을 유지합니다.
        transformer_output = self.transformer_encoder(embedded) # -> (batch_size, seq_len, d_model)
        
        # 3. Output Layer
        # 시퀀스의 모든 타임스텝에 대해 예측을 수행합니다.
        output = self.fc_out(transformer_output) # -> (batch_size, seq_len, 1)
        
        # (batch_size, seq_len) 형태로 flatten
        return output.squeeze(-1)


class TransformerModel(BaseModel):
    def __init__(self, vehicle_name):
        super().__init__('Transformer', vehicle_name)
        # scaler는 모델 외부에서, 데이터를 전처리할 때 사용됩니다.
        self.scaler = StandardScaler()
        # self.model은 train() 함수에서 인스턴스화됩니다.

    def _get_features_and_target(self, data, model_type):
        """
        데이터프레임에서 Trip ID별로 시퀀스(List of Arrays)를 생성합니다.
        """
        features_to_use = TRANSFORMER_FEATURES.copy()
        
        X_sequences = [] # 피처 시퀀스 리스트
        y_sequences = [] # 타겟 시퀀스 리스트

        for trip_id in data['trip_id'].unique():
            trip_data = data[data['trip_id'] == trip_id].copy()
            
            # 피처 (N_timesteps, N_features)
            X_seq = trip_data[features_to_use].values
            X_sequences.append(X_seq)

            # 타겟 (N_timesteps,)
            if model_type == 'hybrid':
                y_seq = (trip_data['target'] - trip_data['physics_prediction']).values
            else: # ml_only
                y_seq = trip_data['target'].values
                
            y_sequences.append(y_seq)

        return X_sequences, y_sequences

    def find_best_params(self, train_data, storage_path):
        """
        Optuna를 사용하여 K-fold CV로 하이퍼파라미터를 튜닝합니다.
        """
        
        # 튜닝을 위해 전체 학습 데이터셋을 사용
        X_full_seq, y_full_seq = self._get_features_and_target(train_data, model_type='hybrid')
        
        # K-Fold CV를 위해 (X, y) 페어를 인덱싱 가능하게 변환
        # (시퀀스 길이가 다르므로 리스트를 유지)
        data_pairs = list(zip(X_full_seq, y_full_seq))
        
        # 스케일러 핏을 위해 튜닝 데이터 전체를 사용 (stack)
        # 스케일링은 튜닝 루프 내 각 fold에서 수행
        X_full_flat_for_scaler = np.vstack(X_full_seq)
        
        # KFold는 인덱스를 분리
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
                'n_epochs': trial.suggest_int('n_epochs', 30, 100), # Epoch 횟수도 튜닝
            }

            fold_rmses = []
            
            # K-fold CV 수행
            for train_idx, val_idx in kf_indices:
                
                fold_train_pairs = [data_pairs[i] for i in train_idx]
                fold_val_pairs = [data_pairs[i] for i in val_idx]
                
                X_train_seq = [p[0] for p in fold_train_pairs]
                y_train_seq = [p[1] for p in fold_train_pairs]
                X_val_seq = [p[0] for p in fold_val_pairs]
                y_val_seq = [p[1] for p in fold_val_pairs]

                # 1. 스케일러 (Fold의 Train 데이터로만 fit)
                scaler = StandardScaler()
                X_train_flat = np.vstack(X_train_seq)
                scaler.fit(X_train_flat)

                # 2. 모델, 옵티마이저, 손실 함수
                model = SimpleTransformer(
                    input_dim=len(TRANSFORMER_FEATURES),
                    d_model=params['d_model'],
                    nhead=params['nhead'],
                    num_encoder_layers=params['num_encoder_layers'],
                    dim_feedforward=params['dim_feedforward'],
                    dropout=params['dropout']
                )
                optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
                criterion = nn.MSELoss()

                # 3. 학습 (Epoch 루프)
                model.train()
                for epoch in range(params['n_epochs']):
                    epoch_loss = 0
                    # Trip(시퀀스)별로 학습 (배치 처리 대신)
                    for X_trip, y_trip in zip(X_train_seq, y_train_seq):
                        # 데이터 스케일링 및 텐서 변환
                        X_trip_scaled = scaler.transform(X_trip)
                        
                        # (seq_len, features) -> (1, seq_len, features) (배치 크기 1)
                        X_tensor = torch.tensor(X_trip_scaled, dtype=torch.float32).unsqueeze(0)
                        # (seq_len,) -> (1, seq_len)
                        y_tensor = torch.tensor(y_trip, dtype=torch.float32).unsqueeze(0)

                        optimizer.zero_grad()
                        output = model(X_tensor) # (1, seq_len)
                        loss = criterion(output, y_tensor)
                        
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()

                # 4. 검증
                model.eval()
                val_predictions_flat = []
                val_targets_flat = []
                
                with torch.no_grad():
                    for X_trip, y_trip in zip(X_val_seq, y_val_seq):
                        X_trip_scaled = scaler.transform(X_trip)
                        X_tensor = torch.tensor(X_trip_scaled, dtype=torch.float32).unsqueeze(0)
                        
                        preds_seq = model(X_tensor) # (1, seq_len)
                        
                        val_predictions_flat.extend(preds_seq.squeeze(0).cpu().numpy())
                        val_targets_flat.extend(y_trip)
                
                fold_rmse = np.sqrt(mean_squared_error(val_targets_flat, val_predictions_flat))
                fold_rmses.append(fold_rmse)

            # Optuna는 평균 RMSE를 최소화하는 방향으로 최적화
            return np.mean(fold_rmses)

        # Optuna Study 실행
        study = optuna.create_study(
            direction='minimize',
            study_name=f"{self.model_name}-{self.vehicle_name}-tuning",
            storage=storage_path,
            load_if_exists=True
        )
        study.optimize(objective, n_trials=N_TRIALS_OPTUNA, timeout=10800) # 3시간 타임아웃
        
        print(f"Best Transformer params: {study.best_params}")
        return study.best_params

    def train(self, train_data, params, model_type='hybrid'):
        """
        주어진 하이퍼파라미터로 전체 학습 데이터셋에 대해 모델을 학습시킵니다.
        """
        X_train_seq, y_train_seq = self._get_features_and_target(train_data, model_type)

        # 1. 스케일러 (전체 학습 데이터로 fit)
        X_train_flat = np.vstack(X_train_seq)
        self.scaler.fit(X_train_flat)

        # 2. 모델, 옵티마이저, 손실 함수
        self.model = SimpleTransformer(
            input_dim=len(TRANSFORMER_FEATURES),
            d_model=params['d_model'],
            nhead=params['nhead'],
            num_encoder_layers=params['num_encoder_layers'],
            dim_feedforward=params['dim_feedforward'],
            dropout=params['dropout']
        )
        optimizer = optim.Adam(self.model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        criterion = nn.MSELoss()

        # 3. 학습 (Epoch 루프)
        n_epochs = params.get('n_epochs', 50) # 튜닝에서 n_epochs를 찾았다고 가정
        
        self.model.train()
        print(f"Training Transformer for {n_epochs} epochs...")
        for epoch in range(n_epochs):
            epoch_loss = 0
            # Trip(시퀀스)별로 학습
            for X_trip, y_trip in zip(X_train_seq, y_train_seq):
                X_trip_scaled = self.scaler.transform(X_trip)
                X_tensor = torch.tensor(X_trip_scaled, dtype=torch.float32).unsqueeze(0)
                y_tensor = torch.tensor(y_trip, dtype=torch.float32).unsqueeze(0)

                optimizer.zero_grad()
                output = self.model(X_tensor)
                loss = criterion(output, y_tensor)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/len(X_train_seq):.4f}")

        # 학습된 모델과 스케일러 반환 (main.py에서 저장)
        return self.model, self.scaler

    def predict(self, test_data, model_type='hybrid'):
        """
        학습된 모델로 테스트 데이터에 대한 예측을 수행합니다.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
            
        X_test_seq, _ = self._get_features_and_target(test_data, model_type)

        self.model.eval()
        
        all_predictions_flat = []
        
        with torch.no_grad():
            for X_trip in X_test_seq:
                # 스케일링 및 텐서 변환
                X_trip_scaled = self.scaler.transform(X_trip)
                X_tensor = torch.tensor(X_trip_scaled, dtype=torch.float32).unsqueeze(0) # (1, seq_len, features)
                
                # 모델 예측
                preds_seq = self.model(X_tensor) # (1, seq_len)
                
                # 결과를 리스트에 저장
                all_predictions_flat.extend(preds_seq.squeeze(0).cpu().numpy())

        # (N_total_samples,) 형태의 1D 배열로 변환
        predictions_flat = np.array(all_predictions_flat)
        
        if model_type == 'hybrid':
            # 하이브리드 모델: ML 예측(residual) + 물리 모델 예측
            physics_predictions_flat = test_data['physics_prediction'].values
            final_predictions = predictions_flat + physics_predictions_flat
            return final_predictions
        else:
            # ML-Only 모델: ML 예측
            return predictions_flat
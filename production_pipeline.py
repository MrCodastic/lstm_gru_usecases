"""
End-to-End Production Pipeline for Energy Forecasting (GRU).
Steps: Data Ingestion -> Feature Engineering -> Training -> Validation -> ONNX Export
"""

import os
import glob
import logging
import kagglehub
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.onnx
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- 0. Configuration & Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    # Data Params
    DATASET_HANDLE = "robikscube/hourly-energy-consumption"
    TARGET_REGION_FILE = "*AEP_hourly.csv"
    SPLIT_DATE = '2017-01-01'
    SEQ_LEN = 24  # Lookback window (24 hours)
    
    # Model Hyperparameters (Tuned via Optuna)
    INPUT_DIM = 4     # Load + Hour_Sin + Hour_Cos + Is_Weekend
    HIDDEN_DIM = 64
    LAYER_DIM = 2
    OUTPUT_DIM = 1
    DROPOUT = 0.2
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    EPOCHS = 15
    
    # Paths
    MODEL_SAVE_PATH = "energy_gru.pth"
    ONNX_SAVE_PATH = "energy_gru.onnx"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Data Pipeline Class ---
class DataPipeline:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_cols = ['load', 'hour_sin', 'hour_cos', 'is_weekend']

    def fetch_and_load(self):
        """Downloads data using KaggleHub and loads the specific CSV."""
        logger.info(f"Downloading dataset: {Config.DATASET_HANDLE}...")
        path = kagglehub.dataset_download(Config.DATASET_HANDLE)
        
        csv_files = glob.glob(os.path.join(path, Config.TARGET_REGION_FILE))
        if not csv_files:
            raise FileNotFoundError("Target CSV not found in downloaded data.")
            
        df = pd.read_csv(csv_files[0])
        logger.info(f"Loaded file: {csv_files[0]}")
        return df

    def preprocess(self, df):
        """Cleans, resamples, and adds cyclical features."""
        # Standardize
        df = df.rename(columns={'Datetime': 'timestamp', 'AEP_MW': 'load'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Handle Duplicates/Gaps
        df = df[~df.index.duplicated(keep='first')]
        df = df.resample('H').mean().interpolate()

        # Feature Engineering
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(float)
        
        return df[self.feature_cols] # Ensure column order

    def prepare_tensors(self, df):
        """Splits data, scales it, and creates sequences for PyTorch."""
        train_df = df.loc[df.index < Config.SPLIT_DATE].copy()
        test_df = df.loc[df.index >= Config.SPLIT_DATE].copy()

        # Fit Scaler ONLY on Training data to prevent leakage
        train_scaled = self.scaler.fit_transform(train_df)
        test_scaled = self.scaler.transform(test_df)

        def create_sequences(data, seq_len):
            xs, ys = [], []
            for i in range(len(data) - seq_len):
                xs.append(data[i:i+seq_len])
                ys.append(data[i+seq_len][0]) # Target is 'load' (col 0)
            return np.array(xs), np.array(ys)

        X_train, y_train = create_sequences(train_scaled, Config.SEQ_LEN)
        X_test, y_test = create_sequences(test_scaled, Config.SEQ_LEN)

        # Convert to Tensors
        tensors = {
            'X_train': torch.FloatTensor(X_train),
            'y_train': torch.FloatTensor(y_train).unsqueeze(1),
            'X_test': torch.FloatTensor(X_test),
            'y_test': torch.FloatTensor(y_test).unsqueeze(1),
            'y_test_raw': test_df['load'].values[Config.SEQ_LEN:] # For validation comparison
        }
        return tensors

# --- 2. Model Definition ---
class ProductionGRU(nn.Module):
    def __init__(self, config):
        super(ProductionGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=config.INPUT_DIM,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.LAYER_DIM,
            batch_first=True,
            dropout=config.DROPOUT
        )
        self.fc = nn.Linear(config.HIDDEN_DIM, config.OUTPUT_DIM)

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        out, _ = self.gru(x)
        # We only care about the last time step prediction
        return self.fc(out[:, -1, :])

# --- 3. Training & Validation Logic ---
class Trainer:
    def __init__(self, model, config):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    def fit(self, train_loader):
        logger.info(f"Starting training on {self.config.DEVICE}...")
        self.model.train()
        
        for epoch in range(self.config.EPOCHS):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.config.DEVICE), y_batch.to(self.config.DEVICE)
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                
                # CRITICAL: Gradient Clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{self.config.EPOCHS} | Loss: {epoch_loss/len(train_loader):.5f}")

    def save_model(self):
        torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
        logger.info(f"Model weights saved to {self.config.MODEL_SAVE_PATH}")

    def export_onnx(self, sample_input):
        """Exports the model to ONNX format for production deployment."""
        self.model.eval()
        sample_input = sample_input.to(self.config.DEVICE)
        
        torch.onnx.export(
            self.model,
            sample_input,
            self.config.ONNX_SAVE_PATH,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_sequence'],
            output_names=['predicted_load'],
            dynamic_axes={'input_sequence': {0: 'batch_size'}, 'predicted_load': {0: 'batch_size'}}
        )
        logger.info(f"Model exported to ONNX: {self.config.ONNX_SAVE_PATH}")

# --- 4. Main Execution Pipeline ---
def run_pipeline():
    # A. ETL Process
    pipeline = DataPipeline()
    try:
        raw_df = pipeline.fetch_and_load()
        processed_df = pipeline.preprocess(raw_df)
        tensors = pipeline.prepare_tensors(processed_df)
    except Exception as e:
        logger.error(f"Data Pipeline Failed: {e}")
        return

    # B. Setup Data Loaders
    train_dataset = TensorDataset(tensors['X_train'], tensors['y_train'])
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # C. Initialize & Train
    model = ProductionGRU(Config)
    trainer = Trainer(model, Config)
    trainer.fit(train_loader)
    trainer.save_model()

    # D. Validation Checks
    logger.info("Running Validation Checks...")
    model.eval()
    with torch.no_grad():
        preds_scaled = model(tensors['X_test'].to(Config.DEVICE)).cpu().numpy()

    # Inverse Transform Logic
    # We need a dummy array because scaler expects 4 columns, we only have 1 (load)
    dummy_preds = np.zeros((len(preds_scaled), Config.INPUT_DIM))
    dummy_preds[:, 0] = preds_scaled.flatten()
    preds_mw = pipeline.scaler.inverse_transform(dummy_preds)[:, 0]
    
    actuals_mw = tensors['y_test_raw']

    # Check 1: Physics (No Negative Load)
    if preds_mw.min() < 0:
        logger.warning(f"⚠️ Physics Check Failed: Model predicted negative load ({preds_mw.min()})")
    else:
        logger.info("✅ Physics Check Passed: No negative predictions.")

    # Check 2: Naive Baseline Comparison
    naive_rmse = np.sqrt(mean_squared_error(actuals_mw[1:], actuals_mw[:-1]))
    model_rmse = np.sqrt(mean_squared_error(actuals_mw, preds_mw))
    
    logger.info(f"Baseline RMSE: {naive_rmse:.2f} MW")
    logger.info(f"Model RMSE:    {model_rmse:.2f} MW")
    
    if model_rmse < naive_rmse:
        logger.info("✅ SUCCESS: Model beat the baseline.")
        # E. Export to ONNX only if model is good
        sample_input = torch.randn(1, Config.SEQ_LEN, Config.INPUT_DIM)
        trainer.export_onnx(sample_input)
    else:
        logger.error("❌ FAILURE: Model is worse than baseline. Aborting export.")

if __name__ == "__main__":
    run_pipeline()
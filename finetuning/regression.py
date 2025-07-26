import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd

# 모델 정의
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

def main():
    parser = argparse.ArgumentParser(description="Run MLP regression with k-fold evaluation")
    parser.add_argument("--embedding_dataset", required=True, help="Path to original file")
    parser.add_argument("--embedding_pkl", required=True, help="Path to pickle file containing 'freesolv_embeddings'")
    parser.add_argument("--device", default="cuda:0", help="CUDA device (default: cuda:0)")
    
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--input_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=1e-3)

    args = parser.parse_args()
        
    # 설정
    batch_size = args.batch_size
    epochs = args.epochs
    k_folds = args.k_folds
    input_dim = args.input_dim
    lr = args.lr
    weight_decay = args.weight_decay
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    data_df = pd.read_csv(args.embedding_dataset)
    labels = data_df["labels"].values
    
    with open(args.embedding_pkl, "rb") as f:
        data = pickle.load(f)
        embeddings = data["final_embeddings"]

    # 정규화 및 Tensor 변환
    scaler = StandardScaler()
    freesolv_labels_scaled = scaler.fit_transform(np.array(labels).reshape(-1, 1))
    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(freesolv_labels_scaled, dtype=torch.float32)

    # K-Fold 시작
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_test_rmses = []

    for fold, (all_idx, _) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")

        train_idx, temp_idx = train_test_split(all_idx, test_size=0.4, random_state=fold)
        valid_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=fold)

        X_train, y_train = X[train_idx].to(device), y[train_idx].to(device)
        X_valid, y_valid = X[valid_idx].to(device), y[valid_idx].to(device)
        X_test, y_test = X[test_idx].to(device), y[test_idx].to(device)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

        model = MLPRegressor(input_dim=input_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr, weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

        train_losses, valid_losses = [], []
        best_valid_loss = float('inf')
        best_test_rmse = None

        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            scheduler.step()

            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_valid)
                val_loss = criterion(val_outputs, y_valid).item()
                valid_losses.append(val_loss)

                if val_loss < best_valid_loss:
                    best_valid_loss = val_loss
                    test_preds = model(X_test).cpu().numpy()
                    best_test_rmse = np.sqrt(mean_squared_error(y_test.cpu().numpy(), test_preds))

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")

        print(f"Best Validation Loss: {best_valid_loss:.4f}")
        print(f"Test RMSE at Best Validation: {best_test_rmse:.4f}")
        fold_test_rmses.append(best_test_rmse)

        # 학습 곡선 시각화
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(valid_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title(f'Training & Validation Loss - Fold {fold+1}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"loss_curve_fold{fold+1}.png")
        plt.close()

    mean_rmse = np.mean(fold_test_rmses)
    std_rmse = np.std(fold_test_rmses)
    print(f"\n✅ Final {k_folds}-Fold Test RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")

if __name__ == "__main__":
    main()
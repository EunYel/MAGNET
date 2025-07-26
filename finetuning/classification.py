import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd

# MLP 분류기 정의
class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# AUC 계산 함수
def compute_auc(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for embeddings, labels in data_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device).float()
            outputs = model(embeddings).squeeze(1)
            all_preds.extend(outputs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return roc_auc_score(all_labels, all_preds), all_preds, all_labels

def main():
    parser = argparse.ArgumentParser(description="Run MLP classification with k-fold evaluation")
    parser.add_argument("--embedding_pkl", required=True, help="Path to pickle file containing 'final_embeddings'")
    parser.add_argument("--embedding_dataset", required=True, help="Path to CSV file containing 'labels' column")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--input_dim", type=int, default=1024)
    args = parser.parse_args()

    # 디바이스 설정
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 임베딩 로드
    with open(args.embedding_pkl, "rb") as f:
        data = pickle.load(f)
        X = torch.tensor(data["final_embeddings"], dtype=torch.float32)

    # 라벨 로드 (CSV에서)
    data_df = pd.read_csv(args.embedding_dataset)
    y = torch.tensor(data_df["labels"].values, dtype=torch.float32)

    # K-Fold 시작
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    fold_test_aucs = []

    for fold, (train_val_idx, _) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold+1}/{args.k_folds} ---")

        train_idx, temp_idx = train_test_split(train_val_idx, test_size=0.4, random_state=fold)
        valid_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=fold)

        X_train, y_train = X[train_idx].to(device), y[train_idx].to(device)
        X_val, y_val = X[valid_idx].to(device), y[valid_idx].to(device)
        X_test, y_test = X[test_idx].to(device), y[test_idx].to(device)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

        model = MLPClassifier(input_dim=args.input_dim).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)

        best_val_auc = 0
        best_model_state = None
        train_losses, val_aucs = [], []

        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device).float()
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)

            val_auc, _, _ = compute_auc(model, val_loader, device)
            val_aucs.append(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Val AUC = {val_auc:.4f}")

        model.load_state_dict(best_model_state)
        test_auc, test_preds, test_labels = compute_auc(model, test_loader, device)
        fold_test_aucs.append(test_auc)

        print(f"✅ Best Validation AUC: {best_val_auc:.4f}")
        print(f"📈 Test AUC at Best Validation: {test_auc:.4f}")

        if fold == args.k_folds - 1:
            fpr, tpr, _ = roc_curve(test_labels, test_preds)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {test_auc:.4f})")
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve (Fold {fold+1})")
            plt.legend()
            plt.tight_layout()
            plt.savefig("roc_curve_fold_last.png")
            plt.close()

    mean_auc = np.mean(fold_test_aucs)
    std_auc = np.std(fold_test_aucs)
    print(f"\n✅ Final {args.k_folds}-Fold Test AUC: {mean_auc:.4f} ± {std_auc:.4f}")

if __name__ == "__main__":
    main()

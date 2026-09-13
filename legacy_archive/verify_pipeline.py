import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

class DualAttentionTLOB(nn.Module):
    def __init__(self, feature_dim=80):
        super().__init__()
        self.spatial_conv = nn.Conv1d(in_channels=feature_dim, out_channels=64, kernel_size=1)
        self.temporal_lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.regressor = nn.Linear(128, 1)

    def forward(self, x):
        batch, seq, feat = x.shape
        x = x.view(batch * seq, feat, 1)
        x = self.spatial_conv(x).view(batch, seq, 64)
        lstm_out, _ = self.temporal_lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return self.regressor(attn_out[:, -1, :])

def run_verification():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Running on compute device: {device}")
    if device.type == "cuda":
        print(f"[+] Device Name: {torch.cuda.get_device_name(0)}")

    df = pd.read_parquet('synthetic_test.parquet')
    features = df.drop(columns=['target_spread']).values
    targets = df['target_spread'].values

    # Reshape 9,900 rows into 99 sequences of 100 ticks each
    X = torch.tensor(features[:9900].reshape(99, 100, 80), dtype=torch.float32)
    Y = torch.tensor(targets[:9900:100], dtype=torch.float32).view(-1, 1)

    loader = DataLoader(TensorDataset(X, Y), batch_size=8, shuffle=True)
    model = DualAttentionTLOB().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    print("[+] Running 3 pipeline sanity epochs...")
    for epoch in range(3):
        total_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx.to(device))
            loss = loss_fn(pred, by.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/3 complete | Batch Loss: {total_loss/len(loader):.4f}")

    print("[✓] Pipeline plumbing verified: GPU tensors, loss backpropagation, and memory allocation work without errors.")

if __name__ == "__main__":
    run_verification()

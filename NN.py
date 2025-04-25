import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# ------------------------
# 1. Simulated Sequence Pose Data (e.g. 30 frames of 34 keypoints each)
# ------------------------
# Shape: (samples, sequence_length, features)
np.random.seed(42)
X = np.random.rand(500, 30, 34).astype(np.float32)  # 500 samples, 30 time steps, 34 features
y = np.random.randint(0, 2, size=(500,))

# ------------------------
# 2. Preprocessing
# ------------------------
scaler = StandardScaler()
X = X.reshape(-1, 34)
X = scaler.fit_transform(X).reshape(500, 30, 34)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

# ------------------------
# 3. High-End Research-Style Model: Temporal 1D CNN + Attention
# ------------------------
class TemporalPoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(34, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.transpose(1, 2)  # shape: (batch, features, time)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.transpose(1, 2)  # shape: (batch, time, features)
        x, _ = self.attn(x, x, x)  # self-attention
        x = x.mean(dim=1)  # global average pooling
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

model = TemporalPoseNet()

# ------------------------
# 4. Training Setup
# ------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 20

# ------------------------
# 5. Training Loop
# ------------------------
for epoch in range(epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        _, predicted = torch.max(val_outputs, 1)
        accuracy = (predicted == y_val).float().mean()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Val Acc: {accuracy:.4f}")

# ------------------------
# 6. Save the Model
# ------------------------
torch.save(model.state_dict(), "temporal_pose_net.pth")
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "../models")
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, "pytorch_point_surrogate.pth")
scaler_path = os.path.join(models_dir, "pytorch_point_scaler.pkl")

# ==========================================
# 1. Load Data
# ==========================================
print("Loading real materials dataset (v2)...")
data_path = os.path.join(script_dir, "../data/processed/mie_materials_v2.npz")
data = np.load(data_path, allow_pickle=False)

wl = data["wavelengths_nm"]
W = wl.size
radii = data["radius_nm"]
n_med = data["n_medium"]
mat_ids = data["material_id"]

sigma_ext = data["sigma_ext"]
geo_area = np.pi * (radii ** 2)
Y_qext = sigma_ext / geo_area[:, None]

M_names = data["material_names"]
M_n = data["materials_n"]
M_k = data["materials_k"]

name_to_idx = {name: i for i, name in enumerate(M_names)}
indices = np.array([name_to_idx[name] for name in mat_ids])

N_n = M_n[indices]
N_k = M_k[indices]

N_total = len(radii)

# ==========================================
# 2. Train / Test Split (No Leakage)
# ==========================================
# Split the particle indices FIRST, so test spectra are completely isolated
idx_train, idx_test = train_test_split(
    np.arange(N_total), test_size=0.2, random_state=42
)

def build_point_dataset(particle_indices):
    """Flattens full spectra into individual (r, n_med, wl, n, k) -> Qext points"""
    N_sub = len(particle_indices)
    
    R_flat = np.repeat(radii[particle_indices], W)
    M_flat = np.repeat(n_med[particle_indices], W)
    L_flat = np.tile(wl, N_sub)
    n_flat = N_n[particle_indices].flatten()
    k_flat = N_k[particle_indices].flatten()
    
    X = np.column_stack((R_flat, M_flat, L_flat, n_flat, k_flat))
    Y = Y_qext[particle_indices].flatten().reshape(-1, 1)
    
    return X, Y

X_train, Y_train = build_point_dataset(idx_train)
X_test, Y_test = build_point_dataset(idx_test)

input_dim = X_train.shape[1]
output_dim = Y_train.shape[1]

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# ==========================================
# 3. PyTorch Architecture (Point-to-Point)
# ==========================================
class SurrogateNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SurrogateNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 4. Training Loop
# ==========================================
if os.path.exists(model_path) and os.path.exists(scaler_path):
    print("Found pre-trained model! Loading from disk...")
    scaler_X = joblib.load(scaler_path)
    model = SurrogateNN(input_dim, output_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
else:
    print(f"Training Point-to-Point Map on {len(X_train)} individual points...")
    
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    
    # Massive batch size for 2 million points
    dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)

    model = SurrogateNN(input_dim, output_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    epochs = 50
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_Y in dataloader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
            
        epoch_loss /= len(dataset)
        scheduler.step(epoch_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | MSE Loss (Q_ext): {epoch_loss:.6f}")

    print("Saving model...")
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler_X, scaler_path)

# ==========================================
# 5. Validation on True Physics
# ==========================================
print("\nValidating on un-seen physical material...")
model.eval()

# Select one random particle from the test set
test_particle_idx = np.random.choice(idx_test)

test_radius = radii[test_particle_idx]
test_material = mat_ids[test_particle_idx]

# Build the 61-point input batch for this specific particle
X_sample_unscaled, _ = build_point_dataset([test_particle_idx])
X_sample_scaled = scaler_X.transform(X_sample_unscaled)

X_tensor = torch.tensor(X_sample_scaled, dtype=torch.float32).to(device)
with torch.no_grad():
    pred_qext = model(X_tensor).cpu().numpy().flatten()

true_qext = Y_qext[test_particle_idx]

# Convert dimensionless Q_ext to cross-section Area
area = np.pi * (test_radius ** 2)
predicted_sigma_ext = pred_qext * area
true_sigma_ext = true_qext * area

plt.figure(figsize=(8, 5))
plt.plot(wl, true_sigma_ext, color='black', linewidth=3, label="True Mie Theory")
plt.plot(wl, predicted_sigma_ext, color='red', linewidth=2, linestyle='dashed', label="Predicted NN Surrogate")

plt.title(f"Holdout Test: Material={test_material}, Radius={test_radius:.1f}nm")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Extinction Cross Section (nm²)")
plt.grid(True)
plt.legend()
plt.show()
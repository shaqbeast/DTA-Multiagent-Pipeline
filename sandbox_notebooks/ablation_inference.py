"""
Ablation Study: Inference Strategy vs. Val RMSE
================================================
Compares four inference strategies using already-trained ensemble models:

  Strategy                            Passes
  ─────────────────────────────────── ──────
  Single model (deterministic)           1
  3-model ensemble                       3
  5-model ensemble                       5
  5-model ensemble + MC Dropout (T=10)  50

Run from the DTA-Multiagent-Pipeline directory:
  python ablation_inference.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.loader import DataLoader
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR      = 'data/'
MODEL_DIR     = 'saved_models_v2/'
ENSEMBLE_SIZE = 5
HIDDEN_DIM    = 256
ESM_DIM       = 1280
NODE_FEAT     = 29
EDGE_FEAT     = 7
N_GIN_LAYERS  = 4
DROPOUT       = 0.2
BATCH_SIZE    = 128
N_MC          = 10   # MC Dropout passes per model

device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps'  if torch.backends.mps.is_available() else
    'cpu'
)
print(f'Device: {device}\n')


# ── Model (must match Phase 3 exactly) ────────────────────────────────────────
class DTA_Model(nn.Module):
    # Architecture matches saved checkpoints in saved_models_v2/:
    #   - GINEConv(mlp, edge_dim=edge_feat) — raw 7-dim edge features passed directly
    #   - No separate edge_proj layer
    def __init__(self, node_feat=29, edge_feat=7, esm_dim=1280,
                 hidden_dim=256, n_layers=4, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for i in range(n_layers):
            in_dim = node_feat if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # edge_dim=edge_feat (7): GINEConv internally projects raw edge features
            self.convs.append(GINEConv(mlp, edge_dim=edge_feat))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.fp_proj = nn.Sequential(
            nn.Linear(2048, hidden_dim), nn.ReLU(), nn.Dropout(dropout))

        self.prot_proj = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)  # raw edge_attr, GINEConv projects internally
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        drug_graph = global_mean_pool(x, batch)
        drug_fp    = self.fp_proj(data.fp.squeeze(1))
        prot       = self.prot_proj(data.target_emb.squeeze(1))
        return self.head(torch.cat([drug_graph, drug_fp, prot], dim=-1)).squeeze(-1)


# ── Load data & models ────────────────────────────────────────────────────────
print('Loading val data...')
val_data   = torch.load(os.path.join(DATA_DIR, 'val_data.pt'))
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
val_targets = np.array([d.y.item() for d in val_data])
print(f'Val samples: {len(val_data)}\n')

print('Loading models...')
models = []
for i in range(1, ENSEMBLE_SIZE + 1):
    m = DTA_Model(node_feat=NODE_FEAT, edge_feat=EDGE_FEAT, esm_dim=ESM_DIM,
                  hidden_dim=HIDDEN_DIM, n_layers=N_GIN_LAYERS, dropout=DROPOUT).to(device)
    m.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, f'model_{i}.pt'), map_location=device))
    m.eval()
    models.append(m)
print(f'Loaded {len(models)} models.\n')


# ── Inference helpers ─────────────────────────────────────────────────────────
def predict_deterministic(model):
    """Single deterministic forward pass (dropout off)."""
    preds = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            preds.extend(model(batch).cpu().numpy())
    return np.array(preds)

def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def rmse(preds, targets):
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


# ── Run ablation ──────────────────────────────────────────────────────────────
results = []

# Strategy 1: Single model, deterministic (1 pass)
print('Strategy 1: Single model (deterministic)...')
preds_1 = predict_deterministic(models[0])
r1 = rmse(preds_1, val_targets)
results.append(('Single model (deterministic)', 1, r1))
print(f'  Val RMSE: {r1:.4f}')

# Strategy 2: 3-model ensemble, deterministic (3 passes)
print('Strategy 2: 3-model ensemble (deterministic)...')
preds_3 = np.mean([predict_deterministic(m) for m in models[:3]], axis=0)
r2 = rmse(preds_3, val_targets)
results.append(('3-model ensemble', 3, r2))
print(f'  Val RMSE: {r2:.4f}')

# Strategy 3: 5-model ensemble, deterministic (5 passes)
print('Strategy 3: 5-model ensemble (deterministic)...')
preds_5 = np.mean([predict_deterministic(m) for m in models], axis=0)
r3 = rmse(preds_5, val_targets)
results.append(('5-model ensemble', 5, r3))
print(f'  Val RMSE: {r3:.4f}')

# Strategy 4: 5-model ensemble + MC Dropout (5 models × 10 passes = 50 total)
print(f'Strategy 4: 5-model ensemble + MC Dropout (T={N_MC})...')
all_passes = []
for idx, model in enumerate(models):
    model.eval()
    enable_mc_dropout(model)
    for _ in range(N_MC):
        preds = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                preds.extend(model(batch).cpu().numpy())
        all_passes.append(preds)
    print(f'  Model {idx+1}/{ENSEMBLE_SIZE} done ({N_MC} passes)')
preds_mc = np.mean(all_passes, axis=0)
r4 = rmse(preds_mc, val_targets)
results.append((f'5-model ensemble + MC Dropout (T={N_MC})', ENSEMBLE_SIZE * N_MC, r4))
print(f'  Val RMSE: {r4:.4f}')


# ── Print table ───────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('ABLATION: Inference Strategy vs. Val RMSE')
print('='*60)
print(f'  {"Strategy":<42} {"Passes":>6}  {"Val RMSE":>8}')
print(f'  {"-"*42} {"------":>6}  {"--------":>8}')
for name, passes, r in results:
    print(f'  {name:<42} {passes:>6}  {r:>8.4f}')
print('='*60)

# Compute gains over single model baseline
baseline = results[0][2]
print('\nGain over single-model baseline:')
for name, passes, r in results[1:]:
    gain = baseline - r
    print(f'  {name}: {gain:+.4f} RMSE improvement')

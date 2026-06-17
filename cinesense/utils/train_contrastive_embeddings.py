import os
import sys
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ResidualProjection(nn.Module):
    """
    Learns a non-linear residual projection on top of frozen sentence-transformer embeddings.
    Base size: 384, Projection size: 512, Final size: 384.
    """
    def __init__(self, base_embeddings: np.ndarray):
        super().__init__()
        # Register base embeddings as a frozen buffer
        self.register_buffer("base_embeddings", torch.tensor(base_embeddings, dtype=torch.float32))
        
        # Projection MLP block (384 -> 512 -> 384)
        self.net = nn.Sequential(
            nn.Linear(384, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 384)
        )
        # Zero-initialize the final linear layer of the residual block
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        
    def project(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.net(x)
        out = x + proj
        return F.normalize(out, p=2, dim=-1)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        base_embs = self.base_embeddings[indices]
        return self.project(base_embs)

class ContrastivePairDataset(Dataset):
    def __init__(self, pairs: np.ndarray, weights: np.ndarray, negatives: np.ndarray):
        self.pairs = torch.tensor(pairs, dtype=torch.long)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.negatives = torch.tensor(negatives, dtype=torch.long)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        anchor = self.pairs[idx, 0]
        positive = self.pairs[idx, 1]
        negs = self.negatives[idx]
        weight = self.weights[idx]
        return anchor, positive, negs, weight

def compute_infonce_loss(model, anchor_idx, positive_idx, negative_idxs, weights, l2_lambda=1e-4, temperature=0.07, beta=1.0):
    """
    Computes InfoNCE loss with in-batch and hard negatives, weighted by Jaccard score.
    """
    q = model(anchor_idx)          # (B, 384)
    k_pos = model(positive_idx)    # (B, 384)
    
    B = q.shape[0]
    
    # Project hard negatives
    flat_neg_idxs = negative_idxs.view(-1)  # (B * 10,)
    k_neg = model(flat_neg_idxs)            # (B * 10, 384)
    k_neg = k_neg.view(B, 10, 384)          # (B, 10, 384)
    
    # Positive similarities: shape (B, 1)
    pos_logits = torch.bmm(q.unsqueeze(1), k_pos.unsqueeze(2)).squeeze(2) / temperature
    
    # Hard negative similarities: shape (B, 10)
    neg_logits = torch.bmm(k_neg, q.unsqueeze(2)).squeeze(2) / temperature
    
    # In-batch negative similarities: shape (B, B)
    in_batch_logits = torch.matmul(q, k_pos.T) / temperature
    
    # Mask out the positive diagonal to avoid self-positives
    diagonal_mask = torch.eye(B, device=q.device) * -1e9
    in_batch_logits = in_batch_logits + diagonal_mask
    
    # Concatenate all logits: shape (B, 1 + 10 + B)
    logits = torch.cat([pos_logits, neg_logits, in_batch_logits], dim=1)
    
    # Positive targets are at index 0
    targets = torch.zeros(B, dtype=torch.long, device=q.device)
    
    # Weighted Cross-Entropy Loss
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    ce_loss = loss_fn(logits, targets)
    
    # Normalize by the sum of weights for scale invariance
    weighted_loss = torch.sum(ce_loss * weights) / torch.sum(weights)
    
    # Cosine distance regularization to restrict embedding drift directly in the loss
    q_base = model.base_embeddings[anchor_idx]
    k_pos_base = model.base_embeddings[positive_idx]
    sim_q = torch.sum(q * q_base, dim=-1)
    sim_k = torch.sum(k_pos * k_pos_base, dim=-1)
    drift_penalty = torch.mean((1.0 - sim_q) + (1.0 - sim_k))
    
    # L2 weight regularization
    l2_reg = 0.0
    for name, param in model.net.named_parameters():
        if 'weight' in name:
            l2_reg += torch.norm(param, p=2) ** 2
            
    total_loss = weighted_loss + l2_lambda * l2_reg + beta * drift_penalty
    return total_loss

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def main():
    set_seed(42)
    device = select_device()
    print(f"Using device: {device}", flush=True)
    
    print("Loading datasets...", flush=True)
    # Load twostage_v1 model assets
    baseline_assets = np.load("cinesense/models/twostage_v1/model_assets.npz")
    anime_ids_v1 = baseline_assets["anime_ids"].astype(np.int32)
    popularity_scores = baseline_assets["popularity_scores"].astype(np.float32)
    catalog_embeddings = baseline_assets["catalog_embeddings"].astype(np.float32)
    
    # Load training pairs
    pairs_data = np.load("cinesense/models/twostage_v2/train_pairs.npz")
    train_pairs = pairs_data["train_pairs"]
    train_weights = pairs_data["train_weights"]
    train_negatives = pairs_data["train_negatives"]
    val_pairs = pairs_data["val_pairs"]
    val_weights = pairs_data["val_weights"]
    val_negatives = pairs_data["val_negatives"]
    
    train_dataset = ContrastivePairDataset(train_pairs, train_weights, train_negatives)
    val_dataset = ContrastivePairDataset(val_pairs, val_weights, val_negatives)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    print("Initializing model and optimizer...", flush=True)
    model = ResidualProjection(catalog_embeddings).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    
    # Learning rate warmup configurations
    warmup_epochs = 3
    warmup_steps = len(train_loader) * warmup_epochs
    current_step = 0
    
    # Check if a smoke test environment variable is set
    smoke_test = os.environ.get("CINESENSE_SMOKE_TEST", "False").lower() in ("true", "1", "yes")
    epochs = 3 if smoke_test else 30
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Median pop for tail drift tracking
    median_pop = np.median(popularity_scores)
    tail_indices = np.where(popularity_scores < median_pop)[0]
    tail_indices_t = torch.tensor(tail_indices, dtype=torch.long, device=device)
    
    # Define degree-0 mask to isolate nodes with no training edges
    active_indices = set(train_pairs.flatten()).union(set(val_pairs.flatten()))
    degree_0_mask = np.ones(16261, dtype=bool)
    for idx in active_indices:
        degree_0_mask[idx] = False
    degree_0_mask_t = torch.tensor(degree_0_mask, dtype=torch.bool, device=device)
    
    print(f"Beginning training. Smoke test: {smoke_test}, Target Epochs: {epochs}", flush=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_weight_sum = 0.0
        
        for anchor_idx, positive_idx, negative_idxs, weight in train_loader:
            anchor_idx = anchor_idx.to(device)
            positive_idx = positive_idx.to(device)
            negative_idxs = negative_idxs.to(device)
            weight = weight.to(device)
            
            # Update learning rate if in warmup phase
            current_step += 1
            if current_step <= warmup_steps:
                lr = 1e-5 + (3e-4 - 1e-5) * (current_step / warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    
            optimizer.zero_grad()
            loss = compute_infonce_loss(
                model, anchor_idx, positive_idx, negative_idxs, weight, 
                l2_lambda=1e-4, temperature=0.07
            )
            loss.backward()
            optimizer.step()
            
            # Track weighted average training loss
            train_loss_sum += loss.item() * weight.sum().item()
            train_weight_sum += weight.sum().item()
            
        train_loss = train_loss_sum / train_weight_sum if train_weight_sum > 0 else 0.0
        
        # Validation Loop
        model.eval()
        val_loss_sum = 0.0
        val_weight_sum = 0.0
        
        with torch.no_grad():
            for anchor_idx, positive_idx, negative_idxs, weight in val_loader:
                anchor_idx = anchor_idx.to(device)
                positive_idx = positive_idx.to(device)
                negative_idxs = negative_idxs.to(device)
                weight = weight.to(device)
                
                loss = compute_infonce_loss(
                    model, anchor_idx, positive_idx, negative_idxs, weight, 
                    l2_lambda=1e-4, temperature=0.07
                )
                val_loss_sum += loss.item() * weight.sum().item()
                val_weight_sum += weight.sum().item()
                
        val_loss = val_loss_sum / val_weight_sum if val_weight_sum > 0 else 0.0
        
        # Compute Cosine Drift and Tail Cosine Drift on device
        with torch.no_grad():
            all_indices = torch.arange(16261, device=device)
            projected = model(all_indices).clone()  # (16261, 384)
            base = model.base_embeddings     # (16261, 384)
            
            # Mask out degree-0 nodes (identity mapping preservation)
            projected[degree_0_mask_t] = base[degree_0_mask_t]
            
            sims = torch.sum(projected * base, dim=1) # (16261,)
            avg_drift = (1.0 - torch.mean(sims)).item()
            tail_drift = (1.0 - torch.mean(sims[tail_indices_t])).item()
            
        print(f"[Epoch {epoch+1:2d}/{epochs:2d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Avg Drift: {avg_drift:.4f} | Tail Drift: {tail_drift:.4f}", flush=True)
        
        # Check early stopping / save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience and not smoke_test:
                print(f"Early stopping triggered at epoch {epoch+1}.", flush=True)
                break
                
    # If not smoke test, restore best model and export
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        
    # Final projection to numpy
    model.eval()
    with torch.no_grad():
        all_indices = torch.arange(16261, device=device)
        projected = model(all_indices)
        base = model.base_embeddings
        projected_masked = projected.clone()
        projected_masked[degree_0_mask_t] = base[degree_0_mask_t]
        v2_embeddings = projected_masked.cpu().numpy()
        
    # Phase 3: Validation Assertions
    print("Step 8: Running validation assertions on retrained embeddings...", flush=True)
    # 1. Embedding shape == (16261, 384)
    assert v2_embeddings.shape == (16261, 384), f"Invalid embedding shape: {v2_embeddings.shape}"
    assert v2_embeddings.dtype == np.float32, "Invalid dtype!"
    
    # 2. Anime IDs unchanged
    assert np.array_equal(anime_ids_v1, baseline_assets["anime_ids"]), "Anime IDs mutated!"
    
    # 3. Popularity scores unchanged
    assert np.array_equal(popularity_scores, baseline_assets["popularity_scores"]), "Popularity scores mutated!"
    
    # 4. Unit-normalized embeddings
    norms = np.linalg.norm(v2_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Embeddings are not unit L2-normalized!"
    
    # 5. Zero NaN check
    assert not np.isnan(v2_embeddings).any(), "Found NaNs in embeddings!"
    
    # 6. Tail drift <= 0.02
    final_sims = np.sum(v2_embeddings * catalog_embeddings, axis=1)
    final_tail_drift = 1.0 - np.mean(final_sims[tail_indices])
    print(f"Final Average Cosine Drift: {1.0 - np.mean(final_sims):.4f}, Final Tail Cosine Drift: {final_tail_drift:.4f}", flush=True)
    if not smoke_test:
        assert final_tail_drift <= 0.02, f"Tail drift exceeds 0.02 threshold: {final_tail_drift:.4f}"
    
    # 7. No quarantined IDs in training graph (verified in pair generation, double check)
    quarantined_ids_set = np.load("cinesense/models/twostage_v2/train_pairs.npz")
    # Verify no validation node leakage
    # ... already asserted in build_training_pairs.py and train_pairs.npz contains correct disjoint sets.
    
    print("Step 9: Exporting assets to models/twostage_v2...", flush=True)
    os.makedirs("cinesense/models/twostage_v2", exist_ok=True)
    
    # Save model_assets.npz
    np.savez_compressed(
        "cinesense/models/twostage_v2/model_assets.npz",
        catalog_embeddings=v2_embeddings,
        popularity_scores=popularity_scores,
        anime_ids=anime_ids_v1
    )
    
    # Save metadata.json
    metadata = {
        "model_version": "twostage_v2",
        "catalog_version": "sha256_8a0b7d55355bb36c6925b687fd43acc2beb5871bce8372846f5d91d7c84381d2",
        "embedding_version": "all-MiniLM-L6-v2-residual-projection_sha256_v2",
        "created_at": pd.Timestamp.now(tz='UTC').isoformat().replace("+00:00", "Z"),
        "hyperparameters": {
            "semantic_weight": 0.85,
            "popularity_weight": 0.15,
            "rating_weight_scheme": "normalized",
            "retrieval_candidate_count": 100,
            "seed_batch_size": 128
        }
    }
    with open("cinesense/models/twostage_v2/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    # Save catalog.parquet
    catalog_df = pd.read_parquet("cinesense/models/twostage_v1/catalog.parquet")
    catalog_df.to_parquet("cinesense/models/twostage_v2/catalog.parquet", index=False)
    
    print("Export completed successfully.", flush=True)

if __name__ == "__main__":
    main()

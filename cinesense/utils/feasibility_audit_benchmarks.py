import os
import sys
import time
import numpy as np
import pandas as pd
import psutil

# Set PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cinesense.utils.model_storage import load_model

def get_current_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0 * 1024.0)

def main():
    print("="*80)
    # 1. Baseline Model Load
    t_start = time.perf_counter()
    mem_before_load = get_current_memory_usage_mb()
    model, catalog_df, metadata = load_model(os.path.join(PROJECT_ROOT, "cinesense/models/twostage_v1"))
    t_end = time.perf_counter()
    mem_after_load = get_current_memory_usage_mb()
    
    baseline_load_time = t_end - t_start
    baseline_mem_increase = mem_after_load - mem_before_load
    
    # Catalog Stats
    N = len(model.catalog_embeddings)
    D = model.catalog_embeddings.shape[1]
    dtype = model.catalog_embeddings.dtype
    
    print("1. CATALOG STATISTICS")
    print(f"  Total Anime Count (N):    {N}")
    print(f"  Embedding Dimension (D):  {D}")
    print(f"  Embedding dtype:          {dtype}")
    print(f"  Baseline Model Load Time: {baseline_load_time:.4f} seconds")
    print(f"  Baseline Memory Increase: {baseline_mem_increase:.2f} MB (RSS after load: {mem_after_load:.2f} MB)")
    print("="*80)

    # 2. Memory Estimations
    # Float32: N * N * 4 bytes
    # Float64: N * N * 8 bytes
    mem_f32_gb = (N * N * 4) / (1024**3)
    mem_f64_gb = (N * N * 8) / (1024**3)
    
    print("2. THEORETICAL MEMORY ESTIMATES FOR N x N MATRIX")
    print(f"  For N = {N}:")
    print(f"    - float32 matrix: {mem_f32_gb:.4f} GB ({mem_f32_gb * 1024:.2f} MB)")
    print(f"    - float64 matrix: {mem_f64_gb:.4f} GB ({mem_f64_gb * 1024:.2f} MB)")
    print("="*80)

    # 3. Benchmark Strategies
    # We copy the embeddings to ensure we run cleanly
    embeddings = model.catalog_embeddings.copy()
    
    # --- Strategy A: Full Matrix (N x N in memory) ---
    print("3. BENCHMARKING STRATEGIES")
    print("Running Strategy A (Full N x N Matrix)...", flush=True)
    t_a_start = time.perf_counter()
    mem_a_start = get_current_memory_usage_mb()
    
    sims_a = embeddings @ embeddings.T
    np.fill_diagonal(sims_a, -np.inf)
    top50_a = np.partition(sims_a, -50, axis=1)[:, -50:]
    avg_top50_a = top50_a.mean(axis=1)
    
    t_a_end = time.perf_counter()
    mem_a_end = get_current_memory_usage_mb()
    time_a = t_a_end - t_a_start
    mem_a_delta = mem_a_end - mem_a_start
    print(f"  Strategy A Done in {time_a:.4f}s | Peak Mem Increase: {mem_a_delta:.2f} MB")
    
    # Clean up to free memory
    del sims_a
    del top50_a
    import gc
    gc.collect()

    # --- Strategy B: Batched Matrix (B x N in memory) ---
    # We evaluate two batch sizes: 1000 and 2000
    for B in [1000, 2000, 4000]:
        print(f"Running Strategy B (Batched with B = {B})...", flush=True)
        t_b_start = time.perf_counter()
        mem_b_start = get_current_memory_usage_mb()
        
        avg_top50_b = np.zeros(N, dtype=np.float32)
        for start in range(0, N, B):
            end = min(N, start + B)
            # Compute similarity for this batch
            batch_sims = embeddings[start:end] @ embeddings.T # shape: (end-start, N)
            
            # Exclude self-similarity. For each item in batch, its self-similarity is at index start + i
            for i in range(end - start):
                batch_sims[i, start + i] = -np.inf
                
            # Partition to find top 50
            top50_b = np.partition(batch_sims, -50, axis=1)[:, -50:]
            avg_top50_b[start:end] = top50_b.mean(axis=1)
            
        t_b_end = time.perf_counter()
        mem_b_end = get_current_memory_usage_mb()
        time_b = t_b_end - t_b_start
        mem_b_delta = mem_b_end - mem_b_start
        print(f"  Strategy B (B={B}) Done in {time_b:.4f}s | Peak Mem Increase: {mem_b_delta:.2f} MB")
        
        # Verify correctness against Strategy A
        diff = np.abs(avg_top50_a - avg_top50_b).max()
        print(f"  Correctness Check: Max absolute diff vs Strategy A: {diff:.2e}")
        
        del batch_sims
        del top50_b
        gc.collect()

    # --- Strategy C: Scikit-learn NearestNeighbors ---
    try:
        from sklearn.neighbors import NearestNeighbors
        print("Running Strategy C (Scikit-learn NearestNeighbors - brute force)...", flush=True)
        t_c_start = time.perf_counter()
        mem_c_start = get_current_memory_usage_mb()
        
        # Note: NearestNeighbors expects distance metric. For cosine similarity, cosine distance is 1 - similarity.
        # We need the closest 51 neighbors (including self, which distance is 0)
        nn = NearestNeighbors(n_neighbors=51, metric='cosine', algorithm='brute', n_jobs=-1)
        nn.fit(embeddings)
        distances, indices = nn.kneighbors(embeddings)
        
        # cosine similarity = 1 - cosine distance
        # Exclude the first neighbor (which is self with distance 0)
        similarities = 1.0 - distances[:, 1:]
        avg_top50_c = similarities.mean(axis=1)
        
        t_c_end = time.perf_counter()
        mem_c_end = get_current_memory_usage_mb()
        time_c = t_c_end - t_c_start
        mem_c_delta = mem_c_end - mem_c_start
        print(f"  Strategy C Done in {time_c:.4f}s | Peak Mem Increase: {mem_c_delta:.2f} MB")
        
        # Verify correctness against Strategy A
        # NearestNeighbors uses cosine distance which might have slight float differences or different order for ties
        diff = np.abs(avg_top50_a - avg_top50_c).max()
        print(f"  Correctness Check: Max absolute diff vs Strategy A: {diff:.2e}")
        
    except Exception as e:
        print(f"  Strategy C Failed: {e}")

if __name__ == "__main__":
    main()

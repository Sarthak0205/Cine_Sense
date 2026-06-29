# CineSense Distance Matrix Scalability Audit

This audit evaluates the memory scalability and representation options for the collaborative watch distance lookup matrix used in the CineSense Graph-Reranking pipeline.

---

## 1. Current Metrics (Baseline)

* **Catalog Size ($N$):** 7,533 items
* **Matrix Structure:** Dense $N \times N$ numpy array
* **Data Type:** `int8` (1 byte per element)
* **Total Elements:** 56,746,089
* **In-Memory Footprint:** **54.12 MB**
* **Serialized NPZ Size (Compressed):** 7.85 MB
* **Non-Zero Entries (Density):** 683,626 (1.20% density)
* **Sparsity:** **98.80%** (98.80% of the elements are zero/disconnected)

---

## 2. Growth Curve Projections

Because the current representation is a **dense** $N \times N$ matrix, memory consumption scales quadratically ($O(N^2)$). As the catalog grows, the dense representation hits a "scalability wall":

| Catalog Size ($N$) | Total Elements ($N^2$) | Dense Footprint (`int8`) | Density (Est. 90 neighbors/item) | Sparse CSR Footprint (Est.) |
| :--- | :--- | :--- | :--- | :--- |
| **7,533 (Current)** | 56.7 Million | 54.12 MB | 1.20% | **3.29 MB** (93.9% reduction) |
| **25,000** | 625 Million | 596.05 MB (0.58 GB) | 0.36% | **10.82 MB** (98.2% reduction) |
| **50,000** | 2.5 Billion | 2,384.19 MB (2.33 GB) | 0.18% | **21.65 MB** (99.1% reduction) |
| **100,000** | 10.0 Billion | 9,536.74 MB (9.31 GB) | 0.09% | **43.30 MB** (99.5% reduction) |

```
Memory Footprint (MB)
  10,000 |                                                 Dense (9.31 GB)
   9,000 |                                                /
   8,000 |                                               /
   7,000 |                                              /
   6,000 |                                             /
   5,000 |                                            /
   4,000 |                                           /
   3,000 |                                  Dense   /
   2,000 |                                 / (2.33 GB)
   1,000 |                        Dense   /
       0 |_______________________/_______/________/
         Current (7.5k)        25k     50k     100k    (Catalog Size)
         Sparse CSR: [---------- Flat line ~43MB ----------]
```

---

## 3. Sparse CSR Alternative

A Compressed Sparse Row (CSR) representation stores only the non-zero elements using three arrays:
1. `data`: The values of non-zero elements (`int8`, 1 byte each).
2. `indices`: The column index of each non-zero element (`int32`, 4 bytes each).
3. `indptr`: Pointers to the start of each row (`int32`, 4 bytes each, length $N + 1$).

### CSR Footprint Formula:
$$\text{CSR Memory (bytes)} = 5 \times E + 4 \times N + 4$$
Where $E$ is the number of non-zero edges, and $N$ is the catalog size.

Assuming a constant average degree of $d \approx 90$ neighbors per anime, the number of non-zero entries scales linearly: $E = d \times N$.
* **CSR Memory (bytes):** $(5 \times 90 + 4) \times N = 454 \times N$ bytes.
* **At 100,000 items:** $45.4$ MB (linear scaling $O(N)$) vs **9.31 GB** (dense quadratic scaling $O(N^2)$).

---

## 4. Migration Recommendations

To safeguard production memory and allow CineSense to scale to 100k+ catalog items without memory bloat, we recommend the following migration path:

1. **Lightweight Array Neighbors Approach (Zero-Dependency):**
   Instead of using `scipy.sparse` (which introduces a heavy runtime SciPy dependency and complicates minimal container packaging), store the collaborative distances in the exact same format as `neighbor_ids` and `neighbor_jaccards`.
   * Save `neighbor_distances` as a sparse 2D array of shape $(N, \text{Max Neighbors})$ where only neighbors within distance 1 or 2 are recorded.
   * Access distance with $O(\log \text{Neighbors})$ binary search (already implemented for Jaccard lookups), which is extremely fast and has zero overhead.
   * This retains the pure numpy layout and cuts serialized/in-memory overhead by $>90\%$.

2. **SciPy CSR Option (If Complex Graph Queries are Needed):**
   * If the pipeline is expanded to require graph algorithms (shortest path, clustering, page rank), migrate to `scipy.sparse.csr_matrix`.
   * The matrix can be saved in NPZ using `scipy.sparse.save_npz` and loaded on server startup in $<0.05$ seconds.

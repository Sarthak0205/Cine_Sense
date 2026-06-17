# CineSense RC1 Release Checklist

This document acts as the final production validation sign-off checklist for CineSense Release Candidate 1 (RC1).

---

## 1. Release Checklist

| Check Item | Status | Verification Detail |
| :--- | :--- | :--- |
| **All Unit Tests Pass** | **PASSED** | 32 tests in `cinesense/tests` executed with 0 failures, 0 errors. |
| **All Integration Tests Pass** | **PASSED** | 16 tests in `api/tests` executed with 0 failures, 0 errors. |
| **All Benchmarks Executed** | **PASSED** | `compare_rerank_methods.py` successfully completed. |
| **Monitoring Scripts Run** | **PASSED** | `production_monitor.py` and `multi_seed_validation.py` execute and pass. |
| **Path-Independent Execution** | **PASSED** | Verified dynamic `Path(__file__)` parent directory resolution in all scripts. |
| **Graph Validation Fallback** | **PASSED** | structural check failure on `graph_assets.npz` prints warning and continues startup with `graph_available = False`. |
| **Control Rollback** | **PASSED** | Rollback by disabling `rerank_enabled` acts as a full semantic-only fallback. |

---

## 2. Validation & Monitoring Details

### A. Benchmarks Comparison (`compare_rerank_methods.py`)
* **Model A (Semantic Only):** NDCG@10 = `0.1301`, MRR = `0.3905`, P@10 = `9.71%`
* **Model B (Semantic + Jaccard):** NDCG@10 = `0.2243`, MRR = `0.5414`, P@10 = `18.00%`
* **Model C (Semantic + Jac + Dist):** NDCG@10 = `0.2249`, MRR = `0.5438`, P@10 = `18.00%` (Locked Baseline)
* **Model D (Full Production):** NDCG@10 = `0.1734`, MRR = `0.4551`, P@10 = `14.00%`

* **Audit Verdict:** **SOME GATES FAILED for Model D**.
  * The full production configuration (Model D, featuring `cosine_power = 2` and `popularity_penalty = 0.05`) experiences recommended quality degradation and fails target promotion gates (NDCG $\ge 0.22$).
  * The locked baseline configuration (Model C, featuring `cosine_power = 0` and `popularity_penalty = 0.0`) comfortably exceeds all target gates.

### B. Production Monitor (`production_monitor.py`)
Evaluating against `gold_standard_v2.json` with the locked baseline configuration (Model C, `cosine_power=0`):
* **Mean NDCG@10:** `0.2305` (Target: $\ge 0.2024$) — **PASSED**
* **Mean MRR:** `0.5438`
* **Mean Precision@10:** `19.71%`
* **Mean Discovery Rate:** `100.0%`
* **Mean Franchise Diversity:** `10.00`
* **Mean Rerank Delta:** `+0.2997` (average score improvement)

### C. Multi-Seed Validation (`multi_seed_validation.py`)
* **Mean Dominant Seed Share:** `59.3%` (Target: $\le 60\%$) — **PASSED**
* **Scenarios evaluated:** 55
* **Scenarios exceeding 60% individually:** 15 (e.g. `Death Note + One Piece` at 90.0%)

### D. A/B Routing Split Simulation (100,000 users)
Deterministic routing split using CRC32 modulo hash distribution:
* **Mean Bucket:** `49.5783` (Expected: `49.50`)
* **Std Deviation:** `28.8775` (Expected: `28.87`)
* **Routing Deviation Error:**
  * 25% Rollout: Actual `25.0250%` (Error = `0.0250%`) — **PASSED**
  * 50% Rollout: Actual `49.8320%` (Error = `0.1680%`) — **PASSED**
  * 75% Rollout: Actual `74.8760%` (Error = `0.1240%`) — **PASSED**
  *(All errors strictly $< 1.0\%$)*

---

## 3. Final Release Decision

**B. Release RC1 with known limitations**

### Known Limitations:
1. **Model D Quality Degradation:**
   Using `cosine_power = 2` and `popularity_penalty = 0.05` causes recommendations quality to drop from NDCG@10 of `0.2249` (Model C) to `0.1734` (Model D), which is below target gates. 
   * **Mitigation:** Deploy with production environment variables set to the Model C configuration parameters:
     - `CINESENSE_COSINE_POWER=0`
     - `CINESENSE_POPULARITY_PENALTY=0.0`
     - `CINESENSE_JACCARD_WEIGHT=1.0`
     - `CINESENSE_DISTANCE_WEIGHT=0.05`
     This setup guarantees full promotion gates are met (NDCG@10 = `0.2305`).

2. **Distance Lookup Matrix Scalability:**
   The collaborative distance matrix is stored as a dense $7533 \times 7533$ array. Memory footprint is currently manageable ($54$ MB), but scales quadratically $O(N^2)$ to $9.3$ GB at 100k catalog items.
   * **Mitigation:** Prioritize the sparse neighbor-based indexing migration (as detailed in `distance_matrix_audit.md`) before expanding the catalog size beyond 20k items.

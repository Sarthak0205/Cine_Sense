# CineSense Final Validation Audit Summary (Pre-v1.0.0 Release Gate)

This document presents the final validation summary and release recommendation for CineSense Release Candidate 1 (RC1) based on the execution of the full validation audit suite comparing Model C (Locked Baseline) and Model D (Production Candidate).

---

## 1. Executive Summary

| Audit Area | Model C Status | Model D Status | Critical Metric |
| :--- | :---: | :---: | :--- |
| **Franchise Leakage Stress Test** | **PASS** | **PASS** | Leakage@10 = 0% / Leakage@20 = 0% |
| **Explanation Truthfulness Audit** | **PASS** | **PASS** | Explanation Precision = 99.50% |
| **Holdout Benchmark Evaluation** | **PASS** | **FAIL** | Statistically significant quality regressions on Model D |
| **Diversity Audit** | **FAIL** | **FAIL** | High Gini coefficient (> 0.99) & high Top 1% concentration |
| **Stability Audit** | **PASS** | **FAIL** | Model D failed Seed Expansion stability threshold (42.9%) |

---

## 2. Release Gates Check

### Gate 1: Franchise Leakage@10 = 0%
* **Model C**: 0.0% — **PASSED**
* **Model D**: 0.0% — **PASSED**

### Gate 2: Explanation Precision $\ge$ 95%
* **Model C**: 99.50% — **PASSED**
* **Model D**: 99.50% — **PASSED**

### Gate 3: Generic Explanation Rate < 30%
* **Model C**: 0.00% — **PASSED**
* **Model D**: 0.00% — **PASSED**

### Gate 4: No Holdout Regression
* **Model C**: Baseline — **PASSED**
* **Model D**: **FAILED**
  - **NDCG@10 Regression**: -15.61% (95% CI: `[-0.0172, -0.0094]`, statistically significant)
  - **Recall@20 Regression**: -3.74% (95% CI: `[-0.0058, -0.0004]`, statistically significant)

### Gate 5: Catalog Coverage $\ge$ Model C
* **Model C**: 1.79% — Baseline
* **Model D**: 2.26% — **PASSED**

### Gate 6: Seed Order Stability $\ge$ 90%
* **Model C**: 100.0% — **PASSED**
* **Model D**: 100.0% — **PASSED**

### Gate 7: Discovery Rate $\ge$ 95%
* **Model C**: 100.0% — **PASSED**
* **Model D**: 100.0% — **PASSED**

---

## 3. Comparative Analysis

| Metric / Dimension | Model C | Model D | Comparison & Analysis |
| :--- | :---: | :---: | :--- |
| **NDCG@10 (Holdout)** | **0.0833** | 0.0703 | Model D regressed by **-15.61%** (highly significant drop) |
| **Recall@10 (Holdout)** | **0.0629** | 0.0604 | Model D regressed by **-3.92%** |
| **Recall@20 (Holdout)** | **0.0836** | 0.0804 | Model D regressed by **-3.74%** (statistically significant drop) |
| **Attribution Accuracy** | **94.20%** | 70.50% | Model D regressed by **-23.70%** (explanation mismatch) |
| **Catalog Coverage** | 1.79% | **2.26%** | Model D marginally increased coverage due to popularity penalty |
| **Gini Coefficient** | 0.9955 | **0.9943** | Both models exhibit severe concentration inequality (>0.99) |
| **Seed Expansion Stability** | **53.8%** | 42.9% | Model D failed to meet the moderate stability threshold ($\ge 50\%$) |
| **Top 1% Concentration** | 72.36% | **61.84%** | Both models suffer from high concentration towards elite items |

---

## 4. Root Cause Analysis

### Failure 1: Quality Regression in Model D (Gate 4)
* **Description**: Statistically significant drop in NDCG@10 (-15.61%) and Recall@20 (-3.74%) on holdout users.
* **Root Cause**:
  - The combination of **popularity penalty** (`popularity_penalty = 0.05`) and **cosine scaling** (`cosine_power = 2`) in Model D over-penalizes popular relevant items.
  - Cosine power of 2 heavily discounts Jaccard similarity for seeds that are not semantically close, causing the model to lose high-quality collaborative recommendation candidates.
* **Severity**: **CRITICAL**

### Failure 2: Explanation Attribution Mismatch in Model D
* **Description**: Explanation Attribution Accuracy dropped from 94.20% to 70.50% under Model D.
* **Root Cause**:
  - The scoring algorithm in Model D uses `cosine_sim ** 2` to scale the Jaccard similarity component and subtracts a popularity penalty.
  - However, the explanation generator (`RecommendationService.generate_explanations`) determines the `matched_seed` using a hardcoded linear relevance formula:
    `relevance = sim + 1.0 * jac + 0.3 * dist_score`
  - This mismatch causes the explanation to attribute the recommendation to a seed that did not actually contribute the highest score in Model D's formula.
* **Severity**: **HIGH**

### Failure 3: Stability Threshold Violation in Model D
* **Description**: Seed expansion overlap for Model D (42.9%) fell below the 50.0% moderate stability threshold.
* **Root Cause**:
  - Incorporating a popularity penalty and cosine scaling makes the ranking formula highly sensitive to the presence of multiple seeds. Minor changes in seed composition can cause the ranking order to shift dramatically.
* **Severity**: **MEDIUM**

### Failure 4: Diversity Collapse & Concentration (Both Models)
* **Description**: Gini coefficient exceeds 0.99 and Top 1% items occupy >60% of all recommended slots.
* **Root Cause**:
  - The core two-stage retrieval retrieves from a limited pool, and the ranking formula is dominated by a few highly connected hub nodes in the neighbor graph. Popularity penalty only slightly mitigates this.
* **Severity**: **HIGH** (Shared systemic limitation)

---

## 5. Final Release Recommendation

Based on the empirical audit evidence, the final release recommendation is:

```text
Release Model C as CineSense v1.0.0
Move Model D to RC2 experimentation.
```

### Justification

1. **Recommendation Quality**: Model C outperforms Model D significantly on the unseen holdout test split. Model D's popularity penalty and power scaling cause a **15.61% degradation** in NDCG@10, which fails the release gate.
2. **Attribution Accuracy**: Model C maintains a high attribution accuracy of **94.20%**, ensuring that users are presented with truthful explanations. Model D's attribution accuracy drops to **70.50%**, presenting a high risk of user confusion due to incorrect seed attribution.
3. **Robustness and Stability**: Model C meets the stability threshold for seed expansion (53.8% overlap), whereas Model D is highly sensitive and fails the threshold (42.9% overlap).
4. **Conclusion**: Model C is stable, accurate, and provides high-quality recommendations. Model D should be rolled back to experimentation under RC2 to remediate its popularity penalty scaling and explanation attribution mismatch.

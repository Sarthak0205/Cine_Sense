# CineSense Explanation Truthfulness Audit Report

## Audit Summary
* **Model C Status**: **PASS**
* **Model D Status**: **PASS**

## Performance Metrics
| Metric | Model C | Model D |
| --- | --- | --- |
| Explanation Precision | 99.50% | 99.50% |
| Attribution Accuracy | 94.20% | 70.50% |
| Consistency (Shares Sum) | 100.00% | 100.00% |
| Generic Explanation Rate | 0.00% | 0.00% |
| Explanation Coverage >= 1 | 100.00% | 100.00% |
| Explanation Coverage >= 2 | 100.00% | 100.00% |
| Explanation Coverage >= 3 | 100.00% | 100.00% |

## Key Findings

### 1. Attribution Accuracy Regression
* Model C Attribution Accuracy: 94.20%
* Model D Attribution Accuracy: 70.50%

> [!WARNING]
> Model D shows a regression in Attribution Accuracy. 
> This is due to a mismatch between Model D's scoring formula (which weights Jaccard by `cosine_sim ** 2` and subtracts a popularity penalty) 
> and the hardcoded explanation relevance formula: `relevance = sim + 1.0 * jac + 0.3 * dist_score`. 
> As a result, the explanation selected a `matched_seed` that was not the seed that actually drove the recommendation score.
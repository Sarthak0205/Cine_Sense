# CineSense Recommendation Stability Audit Report

## Audit Summary
* **Model C Stability Status**: **PASS**
* **Model D Stability Status**: **FAIL**

## Model C (Locked Baseline) Results
| Test Scenario | Expected Class | Top-10 Overlap | Top-20 Overlap | Spearman Corr | Attribution Drift |
| --- | --- | --- | --- | --- | --- |
| Seed Ordering: Naruto + One Piece | Expected Stable | 100.0% | 100.0% | 1.0000 | 0.0000 |
| Rating Perturbation: DN(5)+MNS(10) vs DN(10)+MNS(10) | Expected Stable | 100.0% | 100.0% | 0.7697 | 0.0000 |
| Seed Expansion: DN vs DN + Monster | Expected Moderate | 53.8% | 53.8% | 0.3778 | 0.4433 |
| Multi-Seed Expansion: NAR+OP vs NAR+OP+BL | Expected Larger | 33.3% | 48.1% | -0.1222 | 0.3177 |

## Model D (Production Candidate) Results
| Test Scenario | Expected Class | Top-10 Overlap | Top-20 Overlap | Spearman Corr | Attribution Drift |
| --- | --- | --- | --- | --- | --- |
| Seed Ordering: Naruto + One Piece | Expected Stable | 100.0% | 100.0% | 1.0000 | 0.0000 |
| Rating Perturbation: DN(5)+MNS(10) vs DN(10)+MNS(10) | Expected Stable | 100.0% | 90.5% | 0.9152 | 0.0061 |
| Seed Expansion: DN vs DN + Monster | Expected Moderate | 42.9% | 42.9% | 0.2899 | 0.4498 |
| Multi-Seed Expansion: NAR+OP vs NAR+OP+BL | Expected Larger | 25.0% | 48.1% | -0.1767 | 0.3171 |

## Sensitivity Analysis & Findings

> [!WARNING]
> Model D violated stability thresholds. 
> Popularity penalties and cosine scaling increase susceptibility to input noise, causing rank drift on minor perturbations.
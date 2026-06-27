# CineSense Holdout Benchmark Evaluation Report

## Audit Summary
* **Holdout Validation Status**: **FAIL**
* **Regressions Flagged**: 6

## Statistical Significance and Bootstrap Results (1000 Samples)
| Metric | Model C | Model D | Abs Delta | Rel Delta | 95% CI of Delta |
| --- | --- | --- | --- | --- | --- |
| ndcg@10 | 0.0833 | 0.0703 | -0.0130 | -15.61% | [-0.0172, -0.0094] |
| recall@10 | 0.0629 | 0.0604 | -0.0025 | -3.92% | [-0.0053, +0.0004] |
| recall@20 | 0.0836 | 0.0804 | -0.0031 | -3.74% | [-0.0058, -0.0004] |
| precision@10 | 0.0774 | 0.0755 | -0.0019 | -2.45% | [-0.0046, +0.0009] |

## Segmented Performance Analysis
| Segment & Metric | Model C | Model D | Abs Delta | Rel Delta |
| --- | --- | --- | --- | --- |
| Popular - ndcg@10 | 0.1043 | 0.0809 | -0.0234 | -22.42% |
| Popular - recall@10 | 0.1024 | 0.0937 | -0.0087 | -8.47% |
| Mid-Tail - ndcg@10 | 0.0753 | 0.0644 | -0.0109 | -14.47% |
| Mid-Tail - recall@10 | 0.0549 | 0.0546 | -0.0003 | -0.60% |
| Long-Tail - ndcg@10 | 0.0782 | 0.0714 | -0.0068 | -8.74% |
| Long-Tail - recall@10 | 0.0393 | 0.0388 | -0.0005 | -1.34% |
| Cold-Start | N/A | N/A | N/A | N/A |

## Overfitting and Recommendation Quality Analysis

> [!IMPORTANT]
> Model D shows statistically significant regressions compared to Model C on holdout validation. 
> The popularity penalty and cosine scaling parameters in Model D appear to degrade overall recommendations quality on unseen holdout users.
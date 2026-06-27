# CineSense Diversity Audit Report

## Audit Summary
* **Model C Diversity Status**: **FAIL**
* **Model D Diversity Status**: **FAIL**

## Diversity Metrics
| Diversity Metric | Model C | Model D |
| --- | --- | --- |
| Catalog Coverage | 1.79% | 2.26% |
| Gini Coefficient | 0.9955 | 0.9943 |
| Novelty | 1.6470 | 1.9875 |
| Franchise Diversity | 10.00 | 10.00 |
| Genre Entropy | 3.4313 | 3.4450 |
| Studio Entropy | 3.7710 | 3.8120 |
| Year Entropy | 4.5886 | 4.6580 |
| Popularity Bias (Mean Pct) | 98.48% | 97.90% |

## Popularity Bucket Distribution
| Popularity Bucket | Model C Frequency | Model D Frequency |
| --- | --- | --- |
| Top 1% | 72.36% | 61.84% |
| Top 10% | 97.65% | 96.26% |
| Top 25% | 99.72% | 99.62% |
| Top 50% | 100.00% | 99.99% |
| Bottom 50% | 0.00% | 0.01% |

## Risk & Concentration Analysis

> [!CAUTION]
> Model D has flagged a high concentration of popular items in recommendations. 
> The Top 1% items dominate more than 30% of the recommendations list, 
> which indicates severe popularity bias and a risk of diversity collapse.
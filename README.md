# Time-aware coarse costs for hierarchical private route planning

## Motivation
"Hprop: Hierarchical privacy-preserving route planning for smart cities" (2023) proposes planning routes hierarchically so that a server learns only coarse location information while exact local legs stay on the device. In fast-changing traffic, static coarse costs may be stale. This experiment tests whether a learned, time-aware coarse cost model helps at the same privacy level.

## Question
In a simulated city with fast-changing traffic, can a hierarchical privacy-preserving route planner with a learned time-aware coarse-level cost model produce routes closer to the non-private optimum than one with static coarse costs, at the same privacy level?

## Dataset
Synthetic 30x30 grid city generated in the script (networkx, numpy): travel times depend on time of day, moving congestion hotspots and noise. Training days and held-out test days use different hotspot paths; random origin-destination trips are drawn on test days. This is not the paper's dataset and nothing is downloaded.

## Baseline vs improved
- **Baseline:** standard two-level hierarchical planner. The server sees only the 5x5-block regions of origin and destination and plans a region route using static historical-average coarse costs; the device plans the exact legs. (This is a generic reconstruction, not the paper's exact method.)
- **Improved:** identical hierarchy and disclosed information, but coarse costs come from a LightGBM model predicting current edge times from time of day, location and a few sparse recent observations. Hyper-parameters are chosen by day-grouped cross-validation on training days only; the test days are used once. LightGBM is run with seeds 42, 43, 44.

Main metric: mean % extra travel time vs a non-private Dijkstra route on true current costs (lower is better). Secondary: share of trips within 10% of optimal, 95th-percentile excess.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Results (latest run)

On a synthetic grid-city traffic simulation (840 test trips), a standard hierarchical region-level private route planner with static average costs reached 6.61% mean extra travel time versus the non-private Dijkstra optimum, while my version with LightGBM time-aware coarse costs reached 1.74% (std 0.02) on the same test trips.

| Metric | Value |
|---|---|
| baseline_score | 6.613 |
| improved_score | 1.745 |
| baseline_score_std | 0 |
| improved_score_std | 0.02104 |
| improvement | 4.869 |
| baseline_within10pct | 80.6 |
| improved_within10pct | 95.36 |
| improved_within10pct_std | 0.0972 |
| baseline_p95_excess | 35.99 |
| improved_p95_excess | 8.832 |
| improved_p95_excess_std | 0.08703 |
| baseline_edge_rmse | 0.7366 |
| improved_edge_rmse | 0.3523 |
| n_test_trips | 840 |
| best_num_leaves | 15 |
| cv_rmse_ratio | 0.3309 |

Run on 2026-10-11. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Rows count edge-time observations (days x slots x edges); LightGBM trains on a 250-edge sample per snapshot. Device is assumed to know true current costs inside the corridor for its local legs (same for both planners). Probes (60 random edges per snapshot) are only used by the improved model. The baseline is deterministic (std 0); LightGBM seeds 42-44 vary. Hyperparameters chosen by day-grouped CV on training days only. Results depend on the synthetic simulator's design; not the paper's dataset or method.

## Reference

This project is motivated by the following paper:

> F Tiausas, K Yasumoto, JP Talusan (2023). *Hprop: Hierarchical privacy-preserving route planning for smart cities*. ACM Transactions on. https://dl.acm.org/doi/abs/10.1145/3616874

Paper author: Prof. Abhishek Dubey, Vanderbilt University (abhishek.dubey@vanderbilt.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

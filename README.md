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

## Reference

This project is motivated by the following paper:

> F Tiausas, K Yasumoto, JP Talusan (2023). *Hprop: Hierarchical privacy-preserving route planning for smart cities*. ACM Transactions on. https://dl.acm.org/doi/abs/10.1145/3616874

Paper author: Prof. Abhishek Dubey, Vanderbilt University (abhishek.dubey@vanderbilt.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

"""Hierarchical private route planning: static vs LightGBM time-aware coarse costs (synthetic grid city)."""
import json, os, sys
import numpy as np, networkx as nx, lightgbm as lgb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

SEED, N, BLK, T, K, H = 42, 30, 5, 48, 60, 3   # grid, block size, slots/day, probes, hotspots
N_TR, N_TE, SLOTS_TE, TRIPS = 18, 7, 6, 20       # train days, test days, test slots/day, trips/snapshot
NB = N // BLK
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

# ---- city: fine grid graph, edge geometry, base times ----
G = nx.grid_2d_graph(N, N)
edges = list(G.edges()); E = len(edges)
mid = np.array([[(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] for a, b in edges])
horiz = np.array([a[0] == b[0] for a, b in edges], float)
base = 1 + 0.3 * rng.random(E)
DM = np.linalg.norm(mid[:, None] - mid[None], axis=2)
central = np.exp(-np.linalg.norm(mid - (N - 1) / 2, axis=1) / 12)
hour = np.arange(T) / 2
rush = np.exp(-(hour - 8) ** 2 / 4.5) + np.exp(-(hour - 17) ** 2 / 4.5)
tod = 1 + 0.5 * rush[:, None] * central[None]     # (T, E) time-of-day factor

def simulate(n_days, r):
    """True edge times (days, T, E): base * rush hour * moving hotspots * noise."""
    out = np.empty((n_days, T, E))
    for d in range(n_days):
        s, e = r.uniform(0, N - 1, (H, 2)), r.uniform(0, N - 1, (H, 2))
        amp, laps = r.uniform(1.0, 3.0, H), r.uniform(1.5, 3.5, H)
        for t in range(T):
            frac = 1 - np.abs((t / T * laps) % 2 - 1)           # back-and-forth movement
            c = s + (e - s) * frac[:, None]
            d2 = ((mid[:, None] - c[None]) ** 2).sum(2)
            hot = 1 + (amp * np.exp(-d2 / 18)).sum(1)
            out[d, t] = base * tod[t] * hot * np.exp(0.1 * r.standard_normal(E))
    return out

def features(t, y, r):
    """Edge features: time, location, base time, and K sparse current probes (kernel-smoothed ratios)."""
    o = r.choice(E, K, replace=False); ratio = y[o] / base[o]; D = DM[:, o]
    f = [np.full(E, t), mid[:, 0], mid[:, 1], base, horiz, np.full(E, ratio.mean())]
    for s in (3, 6):
        w = np.exp(-D ** 2 / (2 * s * s))
        f += [(w * ratio).sum(1) / (w.sum(1) + 1e-6), w.sum(1)]
    f += [ratio[D.argmin(1)], D.min(1)]
    return np.column_stack(f)

print("simulating city...", flush=True)
Ytr = simulate(N_TR, np.random.default_rng(SEED))
Yte = simulate(N_TE, np.random.default_rng(SEED + 1000))   # different hotspot paths
static = Ytr.mean((0, 1))                                    # baseline: static historical edge averages

# ---- training set for the coarse cost model (sampled edges per snapshot) ----
r = np.random.default_rng(SEED)
X, y, grp = [], [], []
for d in range(N_TR):
    for t in range(T):
        idx = r.choice(E, 250, replace=False)
        X.append(features(t, Ytr[d, t], r)[idx]); y.append((Ytr[d, t] / base)[idx]); grp += [d] * 250
X, y, grp = np.vstack(X), np.concatenate(y), np.array(grp)
print(f"train rows for LightGBM: {len(y)}", flush=True)

def make(seed, **p):
    return lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, min_child_samples=50, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.8, random_state=seed, n_jobs=4, verbose=-1, **p)

# ---- small grid search, CV on training days only ----
best, best_rmse = None, 1e9
for nl in (15, 63):
    errs = []
    for a, b in GroupKFold(3).split(X, y, grp):
        m = make(SEED, num_leaves=nl).fit(X[a], y[a])
        errs.append(np.mean((m.predict(X[b]) - y[b]) ** 2))
    print(f"CV num_leaves={nl}: rmse={np.sqrt(np.mean(errs)):.4f}", flush=True)
    if np.mean(errs) < best_rmse: best, best_rmse = nl, np.mean(errs)
models = {s: make(s, num_leaves=best).fit(X, y) for s in (42, 43, 44)}

# ---- hierarchical private planner ----
R = nx.grid_2d_graph(NB, NB)
reg = lambda c: (c[0] // BLK, c[1] // BLK)
cells = {k: [(i, j) for i in range(k[0] * BLK, (k[0] + 1) * BLK) for j in range(k[1] * BLK, (k[1] + 1) * BLK)] for k in R}
ctr = lambda k: (k[0] * BLK + BLK // 2, k[1] * BLK + BLK // 2)

def set_w(name, arr):
    for (u, v), w in zip(edges, arr): G[u][v][name] = w

def server_regions(arr, a, b):
    """Server side: coarse region route using only region ids and coarse edge costs `arr`."""
    set_w("c", arr)
    for A, B in R.edges:
        R[A][B]["w"] = nx.dijkstra_path_length(G.subgraph(cells[A] + cells[B]), ctr(A), ctr(B), weight="c")
    return nx.dijkstra_path(R, a, b, weight="w")

def device_leg(path, o, d):
    """Device side: exact route on current true costs inside the corridor of the chosen regions."""
    return nx.dijkstra_path_length(G.subgraph([c for k in path for c in cells[k]]), o, d, weight="true")

# ---- test: sparse-probe snapshots, random OD trips ----
print("evaluating on held-out days...", flush=True)
rt = np.random.default_rng(SEED + 7)
names = ["base"] + [f"ml{s}" for s in models]
cost = {n: [] for n in names}; opt = []; se = {n: 0.0 for n in names}; cnt = 0
for d in range(N_TE):
    for t in rt.choice(T, SLOTS_TE, replace=False):
        yt = Yte[d, t]; set_w("true", yt)
        Xt = features(t, yt, rt)
        coarse = {"base": static}
        for s, m in models.items(): coarse[f"ml{s}"] = np.clip(m.predict(Xt), 0.3, None) * base
        for n in names: se[n] += np.sum((coarse[n] - yt) ** 2)
        cnt += E
        for _ in range(TRIPS):
            while True:
                o, g = (tuple(rt.integers(0, N, 2)) for _ in range(2))
                if reg(o) != reg(g) and abs(o[0] - g[0]) + abs(o[1] - g[1]) >= 10: break
            opt.append(nx.dijkstra_path_length(G, o, g, weight="true"))  # non-private optimum
            cost.setdefault("_o", []).append((o, g))
        trips = cost["_o"][-TRIPS:]
        for n in names:
            routes = {}
            for o, g in trips:
                routes[(o, g)] = server_regions(coarse[n], reg(o), reg(g)) if False else None
            cost[n] += [device_leg(server_regions(coarse[n], reg(o), reg(g)), o, g) for o, g in trips]
opt = np.array(opt)

def summarize(n):
    ex = 100 * (np.array(cost[n]) / opt - 1)
    return ex.mean(), 100 * (ex <= 10).mean(), np.percentile(ex, 95), np.sqrt(se[n] / cnt)

res = {n: summarize(n) for n in names}
ml = np.array([res[f"ml{s}"] for s in models])
b = res["base"]
metrics = {
    "baseline_score": b[0], "improved_score": ml[:, 0].mean(), "baseline_score_std": 0.0, "improved_score_std": ml[:, 0].std(),
    "improvement": b[0] - ml[:, 0].mean(),
    "baseline_within10pct": b[1], "improved_within10pct": ml[:, 1].mean(), "improved_within10pct_std": ml[:, 1].std(),
    "baseline_p95_excess": b[2], "improved_p95_excess": ml[:, 2].mean(), "improved_p95_excess_std": ml[:, 2].std(),
    "baseline_edge_rmse": b[3], "improved_edge_rmse": ml[:, 3].mean(),
    "n_test_trips": len(opt), "best_num_leaves": best, "cv_rmse_ratio": float(np.sqrt(best_rmse)),
}
metrics = {k: float(v) for k, v in metrics.items()}
print(json.dumps(metrics, indent=2))
if not all(np.isfinite(list(metrics.values()))):
    sys.exit("non-finite metrics")

os.makedirs("figures", exist_ok=True)
plt.figure(figsize=(5, 3.5))
plt.bar(["Static coarse", "LightGBM coarse"], [metrics["baseline_score"], metrics["improved_score"]], color=["gray", "tab:blue"])
plt.ylabel("Mean extra travel time vs optimal (%)"); plt.tight_layout(); plt.savefig("figures/excess.png", dpi=120)

out = {
    "question": "In a simulated fast-changing city, does a hierarchical privacy-preserving route planner with a learned time-aware coarse cost model give routes closer to the non-private optimum than one with static coarse costs, at the same privacy level?",
    "method": "30x30 grid city with time-of-day, moving hotspots and noise; train/test days with different hotspot paths; server sees only 5x5-block regions of origin/destination, device plans exact legs in the chosen region corridor; coarse region costs from static averages (baseline) vs LightGBM using time, location and sparse probes (improved).",
    "dataset": {"name": "Synthetic grid-city traffic simulation", "source_url": "generated in script (networkx, numpy); no download",
                "n_rows": (N_TR + N_TE) * T * E, "n_train": N_TR * T * E, "n_test": N_TE * T * E,
                "same_as_paper": False, "baseline_is_papers_method": False},
    "metric_name": "Mean percentage extra travel time vs non-private Dijkstra on true current costs", "higher_is_better": False,
    "baseline": "Two-level hierarchical private planner; coarse region-to-region costs from static per-edge historical averages.",
    "improved": "Same hierarchy and disclosed info; coarse costs from a LightGBM model predicting current edge times from time of day, location and sparse recent probes.",
    "metrics": metrics,
    "notes": "Rows count edge-time observations (days x slots x edges); LightGBM trains on a 250-edge sample per snapshot. Device is assumed to know true current costs inside the corridor for its local legs (same for both planners). Probes (60 random edges per snapshot) are only used by the improved model. The baseline is deterministic (std 0); LightGBM seeds 42-44 vary. Hyperparameters chosen by day-grouped CV on training days only. Results depend on the synthetic simulator's design; not the paper's dataset or method.",
}
with open("results.json", "w", encoding="utf-8") as f: json.dump(out, f, indent=2)
print("wrote results.json")

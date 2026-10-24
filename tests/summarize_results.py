import glob
import json
import numpy as np

print("=== EXP-0 ===")
for f in glob.glob("results/raw/exp0/*.json"):
    d = json.load(open(f))
    for r, v in d["results_by_rank"].items():
        print(f"Rank {r}: Frob={v['mean_relative_frobenius']}, Ratio={v['core_to_dense_byte_ratio']}, KL={v['kl_divergence']}, Agr={v['argmax_agreement']}, Jac={v['window_jaccard_mean']}")

print("\n=== EXP-2 SYSTEMS ===")
for f in glob.glob("results/raw/exp2/*.json")[:1]:
    d = json.load(open(f))
    for m, v in d["modes"].items():
        c = v["cold"]
        w = v["warm"]
        print(f"Mode {m}: Cold Wall={c['wall_time_sec']*1000:.1f}ms, Pred={c['predicted_time_sec']*1000:.1f}ms, DRAM={c['dram_bytes']/(1024*1024):.2f}MB, SSD={c['ssd_bytes']/(1024*1024):.2f}MB, Misses={c['ssd_misses']}")

print("\n=== EXP-3 ABLATION ===")
var_data = {}
for f in sorted(glob.glob("results/raw/exp3/*.json")):
    d = json.load(open(f))
    for vname, vinfo in d["variants"].items():
        s = vinfo["systems"]
        if vname not in var_data:
            var_data[vname] = {"tpot": [], "ttft": [], "rss": [], "dram": [], "ssd": [], "tok_s": []}
        var_data[vname]["tpot"].append(s["tpot_sec"] * 1000)
        var_data[vname]["ttft"].append(s["ttft_sec"] * 1000)
        var_data[vname]["rss"].append(s["peak_rss_mb"])
        var_data[vname]["dram"].append(s["dram_bytes"] / (1024 * 1024))
        var_data[vname]["ssd"].append(s["ssd_bytes"] / (1024 * 1024))
        var_data[vname]["tok_s"].append(s["tok_per_sec"])

for k, v in var_data.items():
    print(f"{k:15s} | TTFT: {np.mean(v['ttft']):.1f}+-{np.std(v['ttft']):.1f}ms | TPOT: {np.mean(v['tpot']):.1f}+-{np.std(v['tpot']):.1f}ms | RSS: {np.mean(v['rss']):.1f}MB | DRAM: {np.mean(v['dram']):.2f}MB | SSD: {np.mean(v['ssd']):.2f}MB | Tok/s: {np.mean(v['tok_s']):.1f}")

print("\n=== EXP-4 SENSITIVITY ===")
for f in glob.glob("results/raw/exp4/*.json"):
    d = json.load(open(f))
    print("Rank sweep:", d["sensitivity"]["rank"])
    print("Budget sweep:", d["sensitivity"]["budget_mb"])
    print("Tau sweep:", d["sensitivity"]["tau"])

"""
Tables from mechanics.py: how do tied and symmetric gaits compare?

The unit is the body on fixed bodies (seeds averaged, as in the paper) and the
seed in evolution. Exploratory: many measures, tests uncorrected.

Writes to --out:
  gaps.csv            per brain and measure: set means, and tied - tree, symmetric - tree,
                      symmetric - tied with bootstrap 95% CI and permutation p
  decomposition.csv   log displacement gap split into cycles, path per cycle, straightness
  locking.csv         locked share by set and training stage, by each measure
  knockout.csv        per set: how the score depends on single hinges
  knockout_at_zero.csv  the same with hinges held at 0, which also changes the pose
  partner_knockout.csv, partner_tests.csv  how alike a hinge's and its partner's effects are
  detector_conditional.csv  the paper's locked share, split into paired stroke and locked of paired
  tails.csv           each set's top and bottom quartile of bodies
  lags.csv            joint lag of steady pairs by relation
  evolved.csv         E1: hinges, modules, gap adjusted for hinge count (cluster bootstrap)
  evolved_by_hinges.csv  E1: best score within each hinge-count bin

Usage:
    python analyse_mechanics.py --data __data__/ariel_paper_v2 --out __data__/ariel_paper_v2/paper/mechanics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

import runs
from mechanics import inverse_participation
from sim import joint_count
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

SETS = ("tree", "leg", "shape", "tied", "sym")
BRAINS = ("bang_bang", "dann")
COMPARE = (("tied", "tree"), ("sym", "tree"), ("sym", "tied"))
N_BOOT, N_PERM = 5000, 5000
KO_MIN_BASE = 0.25          # m; below this a relative knockout effect is mostly noise
MEASURES = ("score", "cycles", "path_per_cycle", "straightness", "wobble", "lateral_m", "period_s", "regularity",
            "n_active", "hinge_range_deg", "participation", "participation_limb", "limb_locked_all", "limb_locked_unnested",
            "joint_locked_all", "bodies_in_contact", "n_stepping", "n_always", "stepping_duty", "contact_changes_per_s",
            "foot_speed_rel", "foot_fwd_frac", "tilt_mean_deg", "tilt_rms_deg", "core_z_mean", "core_z_rms",
            "yaw_total_deg", "work_j", "log_work_per_m", "log_cost_of_transport", "work_hinges")


def set_label(genome: str, source: str) -> str:
    if genome == "tree_symmetric":
        return "sym"
    if genome == "tree_tied":
        return "tied"
    return {"random": "tree", "shape_matched": "shape", "leg_matched": "leg"}[source]


def load(root: Path, name: str) -> pd.DataFrame:
    # A body with fewer than two active hinges has an empty pairs.csv.
    return pd.concat([pd.read_csv(p) for p in sorted(root.glob(f"*/{name}")) if p.stat().st_size > 1], ignore_index=True)


def load_episodes(root: Path) -> pd.DataFrame:
    df = load(root, "mechanics.csv")
    df["set"] = [set_label(g, s) for g, s in zip(df["genome"], df["body_source"])]
    df["log_work_per_m"] = np.log(df["work_per_m"])
    df["log_cost_of_transport"] = np.log(df["cost_of_transport"])
    return df


def gap_stats(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    boot = rng.choice(a, (N_BOOT, len(a))).mean(1) - rng.choice(b, (N_BOOT, len(b))).mean(1)
    gap = a.mean() - b.mean()
    both = np.r_[a, b]
    perm = np.array([(lambda s: s[:len(a)].mean() - s[len(a):].mean())(rng.permutation(both)) for _ in range(N_PERM)])
    return {"gap": gap, "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5),
            "p": (1 + (np.abs(perm) >= abs(gap)).sum()) / (N_PERM + 1)}


def gaps(e: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for brain in BRAINS:
        body = e[e["brain"] == brain].groupby(["set", "body_id"])[list(MEASURES)].mean().reset_index()
        for m in MEASURES:
            row = {"brain": brain, "measure": m, **{s: body.loc[body["set"] == s, m].mean() for s in SETS}}
            for x, y in COMPARE:
                st = gap_stats(body.loc[body["set"] == x, m].to_numpy(), body.loc[body["set"] == y, m].to_numpy(), rng)
                row |= {f"{x}-{y}": st["gap"], f"{x}-{y}_lo": st["lo"], f"{x}-{y}_hi": st["hi"], f"{x}-{y}_p": st["p"]}
            rows.append(row)
    return pd.DataFrame(rows)


def decomposition(e: pd.DataFrame) -> pd.DataFrame:
    """Each set's mean log term, over trainings that moved forward, and each term's
    share of the log displacement gap to tree. The log removes the skew, and only
    moving trainings have a log, so the total differs from the gap in metres."""
    ok = e[(e["displacement"] >= 0.05) & (e["straightness"] > 0) & e["cycles"].notna()].copy()
    terms = {"log_cycles": "cycles", "log_path": "path_per_cycle", "log_straight": "straightness"}
    for t, c in terms.items():
        ok[t] = np.log(ok[c])
    ok["log_disp"] = np.log(ok["displacement"])
    rows = []
    for brain in BRAINS:
        b = ok[ok["brain"] == brain].groupby(["set", "body_id"])[["log_disp", *terms]].mean().groupby("set").mean()
        for s in SETS:
            gap = b.loc[s, "log_disp"] - b.loc["tree", "log_disp"]
            row = {"brain": brain, "set": s, "n_trainings": int(((ok["brain"] == brain) & (ok["set"] == s)).sum()),
                   "log_disp": b.loc[s, "log_disp"], "log_gap_to_tree": gap}
            for t in terms:
                row[t] = b.loc[s, t]
                row[f"{t}_gap"] = b.loc[s, t] - b.loc["tree", t]
            rows.append(row)
    return pd.DataFrame(rows)


def detector(data: Path) -> pd.DataFrame:
    """The paper's detector on the E2 replays: phase_locked_frac per checkpoint."""
    frames = []
    for sub in ("e2_random", "e2_tied", "e2_shape_matched", "e2_leg_matched"):
        root = data / "e3" / sub
        if not root.exists():
            continue
        for p in sorted(root.glob("*__*__s*/episodes.csv")):
            if "__bang_bang__" in p.parent.name or "__dann__" in p.parent.name:
                frames.append(pd.read_csv(p, usecols=["run", "brain", "genome", "body_id", "body_source", "ckpt", "evals", "score",
                                                      "phase_locked_frac", "paired_frac", "stored_fitness", "replayed_fitness"]))
    df = pd.concat(frames, ignore_index=True)
    df["set"] = [set_label(g, s) for g, s in zip(df["genome"], df["body_source"])]
    return df


def locking(last: pd.DataFrame, first: pd.DataFrame, det: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fin = det[det["evals"] == det.groupby("run")["evals"].transform("max")]
    early = det[det["evals"] == det.groupby("run")["evals"].transform("min")]
    for brain in BRAINS:
        for s in SETS:
            pick = lambda d: d[(d["brain"] == brain) & (d["set"] == s)]  # noqa: E731
            f, l, df, de = pick(first), pick(last), pick(fin), pick(early)
            rows.append({
                "brain": brain, "set": s,
                "limb_unnested_early": f["limb_locked_unnested"].mean(), "limb_unnested_final": l["limb_locked_unnested"].mean(),
                "limb_all_final": l["limb_locked_all"].mean(), "joint_final": l["joint_locked_all"].mean(),
                "regularity_final": l["regularity"].mean(), "pairs_unnested": l["n_pairs_unnested"].mean(),
                "paper_paired_frac": df["paired_frac"].mean(), "paper_locked_frac_early": de["phase_locked_frac"].mean(),
                "paper_locked_frac_final": df["phase_locked_frac"].mean(),
                "paper_locked_gaits_final": (df["phase_locked_frac"] >= 0.5).mean(),
                "rho_limb_paper": spearmanr(l.merge(df, on="run")["limb_locked_unnested"], l.merge(df, on="run")["phase_locked_frac"],
                                            nan_policy="omit").statistic,
            })
    return pd.DataFrame(rows)


def knockout(ko: pd.DataFrame, hinges: pd.DataFrame, score_col: str = "score_mean") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ko = ko.merge(hinges[["run", "ckpt", "hinge", "active", "range_deg", "limb_amp_m", "work_j", "distal_mass", "axis", "twin", "mirror"]],
                  on=["run", "ckpt", "hinge"])
    ko["delta"] = ko["baseline"] - ko[score_col]
    ko["rel"] = ko["delta"] / ko["baseline"]
    rows, part = [], []
    for (run, ckpt), g in ko.groupby(["run", "ckpt"]):
        base = g["baseline"].iloc[0]
        if base < KO_MIN_BASE:
            continue
        pos = g["delta"].clip(lower=0).to_numpy()
        n = len(g)
        rows.append({"set": g["set"].iloc[0], "brain": g["brain"].iloc[0], "body_id": g["body_id"].iloc[0], "n": n, "baseline": base,
                     "all_frozen": g["all_frozen"].iloc[0],
                     "effective_hinges": inverse_participation(pos), "effective_share": inverse_participation(pos) / n,
                     "top_share": pos.max() / pos.sum() if pos.sum() > 0 else np.nan,
                     "mean_rel_loss": g["rel"].mean(), "sum_rel_loss": g["rel"].clip(lower=0).sum(),
                     "share_score_up": (g["rel"] < -0.05).mean(), "share_negligible": (g["rel"].abs() < 0.05).mean(),
                     "inactive_rel": g.loc[~g["active"], "rel"].mean(),
                     "rho_rel_work": spearmanr(g["rel"], g["work_j"]).statistic if n >= 4 else np.nan})
        rel = dict(zip(g["node"], g["rel"]))
        nodes = list(rel)
        for col in ("twin", "mirror"):
            done = set()
            for n1, partner in zip(g["node"], g[col]):
                if partner >= 0 and partner in rel and frozenset((n1, partner)) not in done:
                    done.add(frozenset((n1, partner)))
                    others = [abs(rel[n1] - rel[x]) for x in nodes if x not in (n1, partner)]
                    part.append({"set": g["set"].iloc[0], "brain": g["brain"].iloc[0], "body_id": g["body_id"].iloc[0], "kind": col,
                                 "partner_diff": abs(rel[n1] - rel[partner]), "other_diff": np.mean(others) if others else np.nan})
    per = pd.DataFrame(rows)
    body = per.groupby(["brain", "set", "body_id"]).mean(numeric_only=True).reset_index()
    table = body.groupby(["brain", "set"]).mean(numeric_only=True).drop(columns=[]).reset_index()
    table["n_bodies"] = body.groupby(["brain", "set"]).size().to_numpy()
    p = pd.DataFrame(part)
    pt = p.groupby(["brain", "set", "kind"]).agg(pairs=("partner_diff", "size"), partner_diff=("partner_diff", "mean"),
                                                 other_diff=("other_diff", "mean")).reset_index()
    # Body level: is a partner's effect closer than other hinges', within a body?
    tests = []
    rng = np.random.default_rng(0)
    for (brain, s, kind), g in p.dropna(subset=["other_diff"]).groupby(["brain", "set", "kind"]):
        d = (g["partner_diff"] - g["other_diff"]).groupby(g["body_id"]).mean().to_numpy()
        if len(d) < 5:
            continue
        boot = rng.choice(d, (N_BOOT, len(d))).mean(1)
        tests.append({"brain": brain, "set": s, "kind": kind, "bodies": len(d), "partner_minus_other": d.mean(),
                      "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5), "p": wilcoxon(d).pvalue})
    return table, pt, pd.DataFrame(tests)


def detector_conditional(det: pd.DataFrame) -> pd.DataFrame:
    """The paper's locked share has all stroke as its denominator. Split it into the
    share of stroke in left-right pairs and the locked share of that paired stroke."""
    fin = det[det["evals"] == det.groupby("run")["evals"].transform("max")].copy()
    rows = []
    for (brain, s), g in fin.groupby(["brain", "set"]):
        body = g.groupby("body_id")[["phase_locked_frac", "paired_frac"]].mean()
        pooled = body["phase_locked_frac"].sum() / body["paired_frac"].sum()
        some = g[g["paired_frac"] >= 0.25]
        rows.append({"brain": brain, "set": s, "locked_share_all_stroke": body["phase_locked_frac"].mean(),
                     "paired_share": body["paired_frac"].mean(), "locked_of_paired_pooled": pooled,
                     "n_paired_25": len(some), "trainings": len(g),
                     "locked_of_paired_25": (some["phase_locked_frac"] / some["paired_frac"]).clip(upper=1).mean()})
    return pd.DataFrame(rows)


def tails(e: pd.DataFrame) -> pd.DataFrame:
    """Measures of each set's bottom and top quartile of bodies, by score."""
    cols = ["score", "n_hinges", "path_per_cycle", "straightness", "regularity", "limb_locked_unnested", "joint_locked_all",
            "tilt_mean_deg", "tilt_rms_deg", "core_z_rms", "bodies_in_contact", "stepping_duty", "foot_speed_rel", "log_work_per_m"]
    rows = []
    for (brain, s), g in e.groupby(["brain", "set"]):
        body = g.groupby("body_id")[cols].mean()
        lo, hi = body["score"].quantile(0.25), body["score"].quantile(0.75)
        for name, sub in (("bottom", body[body["score"] <= lo]), ("top", body[body["score"] >= hi])):
            rows.append({"brain": brain, "set": s, "quartile": name, "bodies": len(sub), **sub.mean().to_dict()})
    return pd.DataFrame(rows)


def lags(root: Path, e: pd.DataFrame) -> pd.DataFrame:
    """Folded joint lag of steady pairs, by relation: a twin's identical gene and
    command could keep it in phase. Only steady pairs have a meaningful lag."""
    p = load(root, "pairs.csv").merge(e[["run", "set", "brain", "body_id"]], on="run")
    p = p[p["joint_quality"] >= 0.9].copy()
    p["fold"] = np.minimum(p["joint_lag"], 1 - p["joint_lag"])
    p["in_phase"], p["anti_phase"] = p["fold"] < 0.1, p["fold"] > 0.4
    body = p.groupby(["brain", "set", "kind", "body_id"])[["fold", "in_phase", "anti_phase"]].mean().reset_index()
    out = body.groupby(["brain", "set", "kind"]).agg(bodies=("fold", "size"), mean_fold=("fold", "mean"), in_phase=("in_phase", "mean"),
                                                     anti_phase=("anti_phase", "mean")).reset_index()
    return out[out["bodies"] >= 5]


def evolved(data: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """E1 bang-bang final populations: hinge counts and the gap adjusted for them."""
    rows = []
    for run_dir in sorted((data / "e1").glob("e1_bang_bang_tree*")):
        cfg = runs.run_config(run_dir)
        for ckpt in runs.final_checkpoints(run_dir):
            g = json.loads((ckpt / "best_genome.json").read_text())
            t = TreeGenome.from_dict(g)
            rows.append({"genome": cfg["genome_type"], "seed": int(cfg["seed"]), "hinges": joint_count(t), "modules": len(t.nodes),
                         "score": -json.loads((ckpt / "meta.json").read_text())["fitness"]})
    pop = pd.DataFrame(rows)
    rng = np.random.default_rng(0)

    def fit(d: pd.DataFrame) -> tuple[float, float, float]:
        x = np.c_[np.ones(len(d)), d["genome"] == "tree_tied", d["genome"] == "tree_symmetric", d["hinges"]].astype(float)
        b = np.linalg.lstsq(x, d["score"].to_numpy(), rcond=None)[0]
        return b[1], b[2], b[3]

    seeds = pop["seed"].unique()
    by_seed = {s: g for s, g in pop.groupby("seed")}
    boot = np.array([fit(pd.concat([by_seed[s] for s in rng.choice(seeds, len(seeds))], ignore_index=True)) for _ in range(2000)])
    est = fit(pop)
    champ = pop.loc[pop.groupby(["genome", "seed"])["score"].idxmax()]
    means = pop.groupby("genome")[["hinges", "modules", "score"]].mean()
    table = pd.DataFrame([
        {"quantity": "final-pop hinges", **means["hinges"].to_dict()},
        {"quantity": "final-pop modules", **means["modules"].to_dict()},
        {"quantity": "final-pop score", **means["score"].to_dict()},
        {"quantity": "champion hinges", **champ.groupby("genome")["hinges"].mean().to_dict()},
        {"quantity": "champion score", **champ.groupby("genome")["score"].mean().to_dict()},
    ])
    adj = pd.DataFrame([{"term": n, "estimate": est[i], "lo": np.percentile(boot[:, i], 2.5), "hi": np.percentile(boot[:, i], 97.5)}
                        for i, n in enumerate(("tied-tree | hinges", "sym-tree | hinges", "per hinge"))])
    raw = pop.groupby(["genome", "seed"])["score"].mean().unstack(0)
    adj = pd.concat([adj, pd.DataFrame([{"term": "tied-tree raw", "estimate": (raw["tree_tied"] - raw["tree"]).mean()},
                                        {"term": "sym-tree raw", "estimate": (raw["tree_symmetric"] - raw["tree"]).mean()}])])
    pop["bin"] = pd.cut(pop["hinges"], [0, 5, 7, 9, 11, 99], labels=["4-5", "6-7", "8-9", "10-11", "12+"])
    best = pop.groupby(["genome", "seed", "bin"], observed=True)["score"].max().reset_index()
    binned = best.groupby(["genome", "bin"], observed=True).agg(best_mean=("score", "mean"), seeds=("score", "size")).reset_index()
    return pd.concat([table, adj], ignore_index=True), binned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250, "display.max_columns", 50)

    e3 = args.data / "e3"
    last, first = load_episodes(e3 / "mechanics_e2"), load_episodes(e3 / "mechanics_e2_first")
    print("replays not reproducing stored fitness:", int(((last["fitness"] - last["stored_fitness"]).abs() > 1e-6).sum()), "of", len(last))

    g = gaps(last, rng)
    g.to_csv(args.out / "gaps.csv", index=False)
    d = decomposition(last)
    d.to_csv(args.out / "decomposition.csv", index=False)
    print("\n== Decomposition: mean log term, gap to tree per term ==")
    print(d[["brain", "set", "n_trainings", "log_gap_to_tree", "log_cycles_gap", "log_path_gap", "log_straight_gap"]].round(3).to_string(index=False))
    show = ["brain", "measure", *SETS, "tied-tree", "tied-tree_p", "sym-tree", "sym-tree_p", "sym-tied", "sym-tied_lo", "sym-tied_hi", "sym-tied_p"]
    print("\n== Gaps (body level) ==")
    print(g[show].round(3).to_string(index=False))

    det = detector(args.data)
    bad = int(((det["replayed_fitness"] - det["stored_fitness"]).abs() > 1e-6).sum())
    print(f"\ndetector replays: {len(det)} checkpoints, {bad} not reproduced")
    lock = locking(last, first, det)
    lock.to_csv(args.out / "locking.csv", index=False)
    print("\n== Locking ==")
    print(lock.round(3).to_string(index=False))

    ko = load(e3 / "mechanics_e2", "knockout.csv")
    ko["set"] = [set_label(a, b) for a, b in zip(ko["genome"], ko["body_source"])]
    hinges = load(e3 / "mechanics_e2", "hinges.csv")
    print("\nknockout baselines not reproduced:", int(((ko["baseline"] - ko["stored_score"]).abs() > 1e-6).sum()), "of", len(ko))
    kt, pt, ptest = knockout(ko, hinges)
    kz, _, _ = knockout(ko, hinges, "score")
    kt.to_csv(args.out / "knockout.csv", index=False)
    kz.to_csv(args.out / "knockout_at_zero.csv", index=False)
    pt.to_csv(args.out / "partner_knockout.csv", index=False)
    ptest.to_csv(args.out / "partner_tests.csv", index=False)
    print("\n== Knockout, hinge held at its mean angle (trainings with baseline >= 0.25 m) ==")
    print(kt.drop(columns=["n"]).round(3).to_string(index=False))
    print("\n== Knockout, hinge held at 0 ==")
    print(kz[["brain", "set", "effective_share", "mean_rel_loss", "inactive_rel"]].round(3).to_string(index=False))
    print("\n== Partner vs other hinge: mean |difference in relative knockout effect| ==")
    print(pt.round(3).to_string(index=False))
    print(ptest.round(3).to_string(index=False))

    cond = detector_conditional(det)
    cond.to_csv(args.out / "detector_conditional.csv", index=False)
    print("\n== The paper's detector: paired share x locked of paired ==")
    print(cond.round(3).to_string(index=False))
    tl = tails(last)
    tl.to_csv(args.out / "tails.csv", index=False)
    print("\n== Top and bottom quartile of bodies ==")
    print(tl[tl["set"].isin(["tree", "tied", "sym"])].round(3).to_string(index=False))
    lg = lags(e3 / "mechanics_e2", last)
    lg.to_csv(args.out / "lags.csv", index=False)
    print("\n== Joint lag of steady pairs ==")
    print(lg.round(3).to_string(index=False))

    ev, binned = evolved(args.data)
    ev.to_csv(args.out / "evolved.csv", index=False)
    binned.to_csv(args.out / "evolved_by_hinges.csv", index=False)
    print("\n== E1 bang-bang ==")
    print(ev.round(3).to_string(index=False))
    print(binned.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

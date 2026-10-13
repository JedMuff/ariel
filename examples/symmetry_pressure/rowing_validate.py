"""
Score rowing_detect.py against hand labels from rowing_review_page.py.

Input: rowing_labels.json, exported from the review page as
{"labels": {"<run>/<gen>/<body_idx>": {"label": ..., "note": ...}}}. Labels
are rowing / alternating / mixed / other / unsure; unsure is left out.

Reports:
  - the confusion matrix of your labels vs the detector's;
  - precision / recall / F1 for rowing vs the rest;
  - ROC AUC of the continuous rowing_score;
  - a sweep of the main thresholds (LAG_ROW, LAG_ALT, LAT_MAX, FORE_MIN, PERIODIC_MIN,
    AMP_RATIO_MIN, FREQ_TOL, SHARE_MIN), recomputed from the stored pair metrics, best first
    by balanced accuracy, with the current setting marked;
  - every disagreement, with the detector's reason.

A disagreement means you and the detector differ on "rowing or not" or on
"alternating or not". Your mixed / other vs the detector's mixed / aperiodic /
unpaired / static is not a disagreement.

Writes <rowing-dir>/validation/disagreements.csv and threshold_sweep.csv.
Rebuild the review page with only the disagreements:
    python rowing_review_page.py --only-disagreements rowing_labels.json

Usage:
    python rowing_validate.py --labels ~/Downloads/rowing_labels.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import rowing_detect as rd

REPO_ROOT = Path(__file__).resolve().parents[2]
USER_LABELS = ("rowing", "alternating", "mixed", "other", "unsure")
KEYS = ["run", "gen", "body_idx"]


def load_labels(path: Path) -> pd.DataFrame:
    raw = json.loads(Path(path).read_text())
    raw = raw.get("labels", raw)
    rows = []
    for key, v in raw.items():
        run, gen, body = key.rsplit("/", 2)
        lab = v["label"] if isinstance(v, dict) else v
        if lab not in USER_LABELS:
            raise ValueError(f"{key}: unknown label {lab!r}")
        rows.append({"run": run, "gen": int(float(gen)), "body_idx": int(float(body)), "user_label": lab,
                     "note": v.get("note", "") if isinstance(v, dict) else ""})
    return pd.DataFrame(rows, columns=[*KEYS, "user_label", "note"])


def joined(labels: pd.DataFrame, individuals: pd.DataFrame) -> pd.DataFrame:
    df = labels.merge(individuals, on=KEYS, how="left", validate="one_to_one")
    missing = df["rowing_label"].isna()
    if missing.any():
        print(f"[warn] {missing.sum()} labelled individuals are not in individuals.csv, ignored")
    return df[~missing & (df["user_label"] != "unsure")].reset_index(drop=True)


def _disagree(user: pd.Series, det: pd.Series) -> pd.Series:
    return ((user == "rowing") != (det == "rowing")) | ((user == "alternating") != (det == "alternating"))


def reason(row: pd.Series, pairs: pd.DataFrame) -> str:
    """The detector's side of a disagreement, in words."""
    lab = row["rowing_label"]
    if lab == "static":
        return "static: no module moves more than the threshold"
    if lab == "unpaired":
        return f"unpaired: only {row['paired_frac']:.0%} of the stroke is in left/right pairs"
    if lab == "aperiodic":
        return f"aperiodic: {row['aperiodic_frac']:.0%} of the stroke has no clear cycle"
    p = pairs[(pairs["run"] == row["run"]) & (pairs["gen"] == row["gen"]) & (pairs["body_idx"] == row["body_idx"])]
    if p.empty:
        return lab
    kinds = p.groupby("pair_kind")["weight"].sum() / max(row["stroke_total_m"], 1e-12)
    parts = [f"{k} {v:.0%}" for k, v in kinds.sort_values(ascending=False).items()]
    sync = np.average(p["mirror_sync"], weights=np.maximum(p["weight"], 1e-12))
    lag = np.average(p["phase_lag"].fillna(0), weights=np.maximum(p["weight"], 1e-12))
    return f"{lab}: stroke by pair kind {', '.join(parts)}; mean sync {sync:.2f}, phase lag {lag:.2f} cycles"


def disagreements(labels: pd.DataFrame, individuals: pd.DataFrame, pairs: pd.DataFrame | None = None) -> pd.DataFrame:
    df = joined(labels, individuals)
    dis = df[_disagree(df["user_label"], df["rowing_label"])].copy()
    if pairs is not None:
        dis["reason"] = [reason(r, pairs) for _, r in dis.iterrows()]
    return dis


def auc(score: np.ndarray, positive: np.ndarray) -> float:
    """ROC AUC as the Mann-Whitney probability that a positive outscores a negative."""
    pos, neg = score[positive], score[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def relabel(df: pd.DataFrame, pairs: pd.DataFrame, lag: float, alt: float, lat: float, fore: float, per: float,
            ratio: float, ftol: float, share: float) -> pd.Series:
    """Detector labels under other thresholds, from the stored pair metrics.
    aperiodic_frac is per module and stays as computed (at PERIODIC_MIN)."""
    p = pairs.copy()
    fmean = p[["freq_l", "freq_r"]].mean(axis=1)
    ok = ((p[["periodicity_l", "periodicity_r"]].min(axis=1) >= per)
          & ((p["freq_l"] - p["freq_r"]).abs() <= ftol * fmean) & (p["amp_ratio"] >= ratio)
          & (p["stroke_lat"] < lat))
    p["w_row"] = p["weight"] * (ok & (p["phase_lag"] < lag) & (p["stroke_fore"] >= fore))
    p["w_alt"] = p["weight"] * (ok & (p["phase_lag"] >= alt))
    w = p.groupby(KEYS)[["w_row", "w_alt"]].sum()
    d = df.join(w, on=KEYS).fillna({"w_row": 0.0, "w_alt": 0.0})
    tot = d["stroke_total_m"].where(d["stroke_total_m"] > 0, np.nan)
    row_frac, alt_frac = d["w_row"] / tot, d["w_alt"] / tot
    lab = np.select(
        [tot.isna(), d["paired_frac"] < rd.UNPAIRED_MAX, d["aperiodic_frac"] > rd.APERIODIC_MIN,
         (row_frac > share) & (row_frac > alt_frac), (alt_frac > share) & (alt_frac > row_frac)],
        ["static", "unpaired", "aperiodic", "rowing", "alternating"], default="mixed")
    return pd.Series(lab, index=df.index)


def scores(user: pd.Series, det: pd.Series) -> dict:
    y, p = (user == "rowing").to_numpy(), (det == "rowing").to_numpy()
    tp, fp, fn, tn = (y & p).sum(), (~y & p).sum(), (y & ~p).sum(), (~y & ~p).sum()
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else float("nan"),
            "balanced_acc": np.nanmean([rec, spec]), "agreement": float((~_disagree(user, det)).mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, nargs="+", required=True,
                        help="one or more exports; an individual labelled in several takes the last file's label")
    parser.add_argument("--stride-root", type=Path, default=REPO_ROOT / "__data__" / "ariel_control_stride_sweep")
    parser.add_argument("--rowing-dir", type=Path, default=None, help="default: <stride-root>/rowing")
    args = parser.parse_args()
    rdir = args.rowing_dir or args.stride_root / "rowing"
    individuals = pd.read_csv(rdir / "individuals.csv")
    pairs = pd.concat([pd.read_csv(p).assign(run=p.stem) for p in sorted((rdir / "pairs").glob("*.csv"))
                       if p.stat().st_size > 1], ignore_index=True)
    labels = pd.concat([load_labels(p) for p in args.labels]).drop_duplicates(KEYS, keep="last")
    df = joined(labels, individuals)
    print(f"{len(labels)} labels, {len(df)} usable (unsure and unknown left out)\n")
    pd.set_option("display.width", 200)

    print("Your label (rows) vs detector (columns):")
    print(pd.crosstab(df["user_label"], df["rowing_label"], margins=True), "\n")
    st = scores(df["user_label"], df["rowing_label"])
    print("Rowing vs rest: " + ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in st.items()))
    print(f"ROC AUC of rowing_score for your 'rowing': {auc(df['rowing_score'].to_numpy(), (df['user_label'] == 'rowing').to_numpy()):.3f}")
    print("By genome x stride:")
    for (g, s), d in df.groupby(["genome", "stride"]):
        sc = scores(d["user_label"], d["rowing_label"])
        print(f"  {g:15s} s{s:<4d} n={len(d):3d} agreement={sc['agreement']:.2f} "
              f"precision={sc['precision']:.2f} recall={sc['recall']:.2f}")

    # Threshold sweep on the labelled individuals only.
    lp = pairs.merge(df[KEYS], on=KEYS)
    grid = {"lag": [0.2, 0.25, 0.3], "alt": [0.35, 0.4, 0.45], "lat": [0.5, 0.7, 1.01], "fore": [0.0, 0.3, 0.4],
            "per": [0.3, 0.5, 0.7], "ratio": [0.1, 0.2, 0.3, 0.5], "ftol": [0.1, 0.2],
            "share": [0.2, 0.25, 0.3, 0.4]}
    current = {"lag": rd.LAG_ROW, "alt": rd.LAG_ALT, "lat": rd.LAT_MAX, "fore": rd.FORE_MIN, "per": rd.PERIODIC_MIN,
               "ratio": rd.AMP_RATIO_MIN, "ftol": rd.FREQ_TOL, "share": rd.SHARE_MIN}
    rows = []
    for combo in itertools.product(*grid.values()):
        th = dict(zip(grid, combo))
        rows.append({**th, **scores(df["user_label"], relabel(df, lp, **th)),
                     "current": all(np.isclose(th[k], current[k]) for k in th)})
    sweep = pd.DataFrame(rows).sort_values(["balanced_acc", "agreement", "f1"], ascending=False)
    out = rdir / "validation"
    out.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(out / "threshold_sweep.csv", index=False)
    cols = [*grid, "precision", "recall", "f1", "balanced_acc", "agreement"]
    print(f"\nThreshold sweep (lag=LAG_ROW, alt=LAG_ALT, lat=LAT_MAX, fore=FORE_MIN, per=PERIODIC_MIN, ratio=AMP_RATIO_MIN, "
          f"ftol=FREQ_TOL, share=SHARE_MIN), "
          f"best 10 of {len(sweep)}:")
    print(sweep.head(10)[cols].to_string(index=False, float_format="%.2f"))
    print("Current setting:")
    print(sweep[sweep["current"]][cols].to_string(index=False, float_format="%.2f"))
    print("A few labels can't pin down five thresholds: prefer settings near the current one that are "
          "consistently near the top, and recheck on a fresh sample (--seed).")

    dis = disagreements(labels, individuals, pairs)
    dis_cols = [*KEYS, "genome", "stride", "user_label", "rowing_label", "rowing_score", "reason", "note"]
    dis[dis_cols].to_csv(out / "disagreements.csv", index=False)
    print(f"\n{len(dis)} disagreements:")
    for _, r in dis.iterrows():
        print(f"  {r['run']} g{r['gen']} b{r['body_idx']}: you={r['user_label']} detector={r['reason']}"
              + (f"  [note: {r['note']}]" if r["note"] else ""))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

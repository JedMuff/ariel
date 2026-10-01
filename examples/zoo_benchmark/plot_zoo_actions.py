"""Plot the action outputs of the best champion of each brain on one body.

Replays, for each brain, the rep with the highest best_xspeed in
<root>/summary.csv on --body, with the training episode loop, and plots one
row per hinge and one column per brain: the commanded action (step line, held
for each control period) over the resulting joint angle (thin grey line).

Usage:
  python plot_zoo_actions.py __data__/ariel_zoo_benchmark --body babyb [--t-min 0 --t-max 5]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from plot_zoo_fitness_curves import BRAIN_COLORS, BRAINS, INK, MUTED, style_axis  # noqa: E402
from zoo_benchmark import build_world, get_state_from_data, make_brain_for  # noqa: E402

HALF_PI = np.pi / 2


def replay(run_dir: Path) -> dict:
    """Run the champion's episode; return action (per control step) and joint angle (per physics step)."""
    cfg = json.loads((run_dir / "config.json").read_text())
    model, data = build_world(cfg["body"])
    brain = make_brain_for(cfg["brain"], model, data, cfg["duration"])
    brain.set_params(np.load(run_dir / "champion.npy"))
    steps_per_ctrl = round(1.0 / (cfg["control_freq"] * model.opt.timestep))
    qadr = model.jnt_qposadr[model.actuator_trnid[:, 0]]

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    brain.reset()
    t_act, acts, t_q, qs = [], [], [float(data.time)], [data.qpos[qadr].copy()]
    for _ in range(round(cfg["duration"] * cfg["control_freq"])):
        t_act.append(float(data.time))
        acts.append(np.asarray(brain.act(float(data.time), get_state_from_data(data)), dtype=float))
        data.ctrl[:] = acts[-1]
        for _ in range(steps_per_ctrl):
            mujoco.mj_step(model, data)
            t_q.append(float(data.time))
            qs.append(data.qpos[qadr].copy())
    return {"t_act": np.array(t_act), "act": np.array(acts), "t_q": np.array(t_q), "q": np.array(qs),
            "duration": cfg["duration"], "seed": cfg["seed"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--body", default="babyb")
    parser.add_argument("--t-min", type=float, default=0.0)
    parser.add_argument("--t-max", type=float, default=None, help="default: episode end")
    parser.add_argument("-o", "--out", type=Path, default=None, help="default: <root>/zoo_actions_<body>.png")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    torch.set_num_threads(1)

    df = pd.read_csv(args.root / "summary.csv")
    df = df[df["body"] == args.body]
    if df.empty:
        raise SystemExit(f"No runs for body {args.body!r} in {args.root / 'summary.csv'}")
    best = df.loc[df.groupby("brain")["best_xspeed"].idxmax()].set_index("brain")
    brains = [b for b in BRAINS if b in best.index]
    traces = {b: replay(args.root / best.loc[b, "run_dir"]) for b in brains}

    n_hinges = traces[brains[0]]["act"].shape[1]
    t_max = args.t_max if args.t_max is not None else traces[brains[0]]["duration"]
    fig, axes = plt.subplots(n_hinges, len(brains), figsize=(3.4 * len(brains), 0.85 * n_hinges + 1.2),
                             sharex=True, sharey=True, squeeze=False)

    for col, brain in enumerate(brains):
        tr = traces[brain]
        m_a = (tr["t_act"] >= args.t_min) & (tr["t_act"] <= t_max)
        m_q = (tr["t_q"] >= args.t_min) & (tr["t_q"] <= t_max)
        for h in range(n_hinges):
            ax = axes[h, col]
            ax.axhspan(-HALF_PI, HALF_PI, color=MUTED, alpha=0.06, lw=0)
            ax.plot(tr["t_q"][m_q], tr["q"][m_q, h], color=MUTED, lw=0.8, alpha=0.9)
            ax.step(tr["t_act"][m_a], tr["act"][m_a, h], where="post", color=BRAIN_COLORS[brain], lw=1.1)
            style_axis(ax)
            ax.grid(False)
            ax.axhline(0, color=MUTED, lw=0.4, alpha=0.6)
            if col == 0:
                ax.set_ylabel(f"hinge {h}", fontsize=8, color=INK, rotation=0, ha="right", va="center")
        axes[0, col].set_title(f"{brain}\n{best.loc[brain, 'best_xspeed']:.3f} m/s (seed {tr['seed']})",
                               fontsize=10, color=INK)
        axes[-1, col].set_xlabel("Time (s)", fontsize=9, color=INK)

    axes[0, 0].set_ylim(-HALF_PI * 1.15, HALF_PI * 1.15)
    axes[0, 0].set_yticks([-HALF_PI, 0, HALF_PI], ["-π/2", "0", "π/2"])
    axes[0, 0].set_xlim(args.t_min, t_max)
    fig.suptitle(f"{args.body}: best champion per brain. Coloured = commanded action "
                 "(held per 50 ms control step), grey = joint angle",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=0.3)
    window = "" if (args.t_min == 0 and args.t_max is None) else f"_{args.t_min:g}-{t_max:g}s"
    out = args.out or args.root / f"zoo_actions_{args.body}{window}.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

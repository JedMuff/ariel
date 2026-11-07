"""
Videos of tree, symmetric and tied individuals from the v2 runs.

Each setup is one experiment and controller: E1 (evolved) with the three
bang-bang controllers, E2 (fixed random bodies) with bang-bang and DANN. Per
setup and genome three individuals are picked by final score:
  E1  the champions of the best, median and worst seed (20 seeds)
  E2  the trainings at the 90th, 50th and 10th percentile (240 trainings)

An individual is replayed with the training episode itself (sim.run_skill_episode)
and its poses are stored; rendering happens afterwards from the stored poses, so
it cannot affect the simulation. The replayed score is compared with the stored
fitness and both are written to manifest.csv.

Writes to <out>/:
  <setup>/<genome>_<rank>.mp4    one individual, real time, camera follows the core
  <setup>_montage.mp4            rows = genomes, columns = ranks
  <setup>_bodies.png             settled pose of each individual: 3/4 view and from above
  manifest.csv

Usage:
    python render_videos.py --data __data__/ariel_paper_v2 --out __data__/ariel_paper_v2/paper/videos
    python render_videos.py ... --only e1_bang_bang --limit 1
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import math
import subprocess
from multiprocessing import Pool
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pandas as pd

import runs
from brains import SETTLE_DURATION, make_brain_for_model
from replay import body_descriptors, ckpt_config
from sim import FORWARD_AXIS, PROFILE, build_world, profile_for, robot_bodies, run_skill_episode, zoo_spec_fn

W, H = 640, 480
FPS = 25
STEPS_PER_FRAME = 20    # physics runs at 500 Hz, so 25 frames per simulated second
GENOMES = ("tree", "tree_symmetric", "tree_tied")
E1_RANKS = {"best": 1.0, "median": 0.5, "worst": 0.0}
E2_RANKS = {"p90": 0.9, "p50": 0.5, "p10": 0.1}
SETUPS = {
    "e1_bang_bang": ("e1", "bang_bang"), "e1_drift": ("e1", "bang_bang_drift"), "e1_free": ("e1", "bang_bang_free"),
    "e2_bang_bang": ("e2", "bang_bang"), "e2_dann": ("e2", "dann"),
}
COL = {"tree": (0xeb, 0x68, 0x34), "tree_symmetric": (0x2a, 0x78, 0xd6), "tree_tied": (0x1b, 0xaf, 0x7a)}


# ── Choosing individuals ──────────────────────────────────────────────────────


def at_quantile(scores: pd.DataFrame, q: float) -> pd.Series:
    """The row at quantile q of score (an actual row, not an interpolation)."""
    s = scores.sort_values("score", kind="stable").reset_index(drop=True)
    return s.iloc[round(q * (len(s) - 1))]


def pick_e1(data: Path, brain: str, genome: str) -> list[dict]:
    df = runs.load_runs(data / "e1")
    df = df[(df["brain"] == brain) & (df["genome"] == genome)]
    out = []
    for rank, q in E1_RANKS.items():
        r = at_quantile(df, q)
        out.append({"rank": rank, "ckpt": str(runs.best_checkpoint(Path(r["path"]))), "id": f"seed {r['seed']}"})
    return out


def pick_e2(data: Path, brain: str, genome: str) -> list[dict]:
    root = data / "e2" / ("tied" if genome == "tree_tied" else "random")
    rows = []
    for d in sorted(root.glob(f"rand_{genome}_[0-9]*__{brain}__s*")):
        if (d / "final.json").exists():
            rows.append({"score": json.loads((d / "final.json").read_text())["score"], "path": str(d), "name": d.name})
    df = pd.DataFrame(rows)
    out = []
    for rank, q in E2_RANKS.items():
        r = at_quantile(df, q)
        last = max((Path(r["path"]) / "checkpoints").iterdir(), key=lambda c: int(c.name.removeprefix("evals")))
        body, _, seed = r["name"].split("__")
        out.append({"rank": rank, "ckpt": str(last), "id": f"body {body.rsplit('_', 1)[1]} {seed}"})
    return out


# ── Replay and render ─────────────────────────────────────────────────────────


def replay_poses(genome: dict, weights: np.ndarray, brain_kind: str, cfg: dict):
    """Run the training episode; (model, data, fitness, qpos per frame, core index)."""
    model, data = build_world(genome, zoo_spec_fn(runs.hinge_actuator(cfg)))
    brain = make_brain_for_model(brain_kind, model, data, np.asarray(weights, dtype=np.float32), None,
                                 duration=PROFILE.loco_duration)
    qpos, n = [], [0]

    def record(d: mujoco.MjData) -> None:
        if n[0] % STEPS_PER_FRAME == 0:
            qpos.append(d.qpos.copy())
        n[0] += 1

    fitness, _ = run_skill_episode(model, data, brain, profile_for(cfg), recorder=record, on_settle_step=record)
    return model, data, fitness, np.asarray(qpos)


def smooth(x: np.ndarray, width: int) -> np.ndarray:
    """Centred moving average per column, edge-padded (steadies the camera)."""
    pad = np.pad(x, ((width // 2, width - width // 2 - 1), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, i], np.ones(width) / width, mode="valid") for i in range(x.shape[1])], axis=1)


def camera(lookat: np.ndarray, distance: float, azimuth: float = 30.0, elevation: float = -25.0) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def label(frame: np.ndarray, lines: list[str], color: tuple[int, int, int]) -> np.ndarray:
    frame = frame.copy()
    for i, text in enumerate(lines):
        y = 24 + 22 * i
        scale, thick = (0.6, 2) if i == 0 else (0.5, 1)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick + 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color if i == 0 else (30, 30, 30), thick, cv2.LINE_AA)
    return frame


def encoder(path: Path) -> subprocess.Popen:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "fast", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render_job(job: dict) -> dict:
    ckpt = Path(job["ckpt"])
    cfg = ckpt_config(ckpt)
    genome = json.loads((ckpt / "best_genome.json").read_text())
    weights = np.load(ckpt / "fwd_weights.npy")
    stored = float(json.loads((ckpt / "meta.json").read_text())["fitness"])
    desc = body_descriptors(genome)

    model, data, fitness, qpos = replay_poses(genome, weights, cfg["brain"], cfg)
    score = -fitness
    bodies = robot_bodies(model)

    # First pass over the poses: the camera path and the body's reach.
    core, reach = [], 0.0
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        core.append(data.qpos[:3].copy())
        reach = max(reach, float(np.linalg.norm(data.xpos[bodies][:, :2] - data.qpos[:2], axis=1).max()))
    core = np.asarray(core)
    lookat = smooth(core, FPS)
    lookat[:, 2] = 0.08
    distance = max(0.8, 2.0 * reach + 0.2)
    settled = int(SETTLE_DURATION * FPS)
    xy0 = core[settled, :2]

    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    model.vis.headlight.ambient[:] = 0.25   # the default washes out surfaces facing the camera
    model.vis.headlight.diffuse[:] = 0.35
    model.vis.headlight.specular[:] = 0.0
    out = Path(job["out"]) / job["setup"] / f"{job['genome']}_{job['rank']}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = encoder(out)
    name = runs.GENOME_LABEL[job["genome"]]
    title = f"{name} - {job['rank']} ({job['id']})"
    size = f"{desc['n_modules']} modules, {desc['n_hinges']} hinges"
    color = tuple(int(c * 0.8) for c in COL[job["genome"]])
    with mujoco.Renderer(model, width=W, height=H) as r:
        for i, q in enumerate(qpos):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            r.update_scene(data, camera=camera(lookat[i], distance))
            moved = float(np.dot(core[i, :2] - xy0, FORWARD_AXIS)) if i >= settled else 0.0
            state = "settling" if i < settled else f"forward {moved:+.2f} m"
            enc.stdin.write(label(r.render(), [title, f"{size}; {state}"], color).tobytes())
    enc.stdin.close()
    enc.wait()

    # Settled pose, for the body sheet.
    data.qpos[:] = qpos[settled]
    mujoco.mj_forward(model, data)
    # The 45 degree field of view fits a radius r at a distance of about 2.4 r.
    rest = float(np.linalg.norm(data.xpos[bodies][:, :2] - data.qpos[:2], axis=1).max())
    views = []
    with mujoco.Renderer(model, width=W, height=H) as r:
        for az, el in ((30.0, -25.0), (90.0, -80.0)):
            r.update_scene(data, camera=camera(np.array([*core[settled, :2], 0.08]), max(0.7, 2.5 * rest + 0.2), az, el))
            views.append(r.render().copy())
    still = label(np.hstack(views), [title, size], color)
    np.save(out.with_suffix(".still.npy"), still)

    return {**{k: job[k] for k in ("setup", "genome", "rank", "id")}, **desc, "stored_score": -stored, "replayed_score": score,
            "diff": score + stored, "reach_m": reach, "video": str(out), "ckpt": str(ckpt)}


# ── Montage and sheet ─────────────────────────────────────────────────────────


def montage(setup_dir: Path, ranks: list[str], out: Path) -> None:
    inputs, layout = [], []
    for row, genome in enumerate(GENOMES):
        for col, rank in enumerate(ranks):
            inputs += ["-i", str(setup_dir / f"{genome}_{rank}.mp4")]
            layout.append(f"{col * W}_{row * H}")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex",
           f"xstack=inputs={len(layout)}:layout={'|'.join(layout)}",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-preset", "fast", str(out)]
    subprocess.run(cmd, check=True)


def body_sheet(setup_dir: Path, ranks: list[str], out: Path) -> None:
    rows = []
    for genome in GENOMES:
        tiles = [np.load(setup_dir / f"{genome}_{rank}.still.npy") for rank in ranks]
        rows.append(np.hstack(tiles))
    sheet = np.vstack(rows)
    cv2.imwrite(str(out), cv2.cvtColor(cv2.resize(sheet, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
    for p in setup_dir.glob("*.still.npy"):
        p.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="+", choices=list(SETUPS), default=list(SETUPS))
    parser.add_argument("--limit", type=int, default=0, help="test: render only this many individuals, no montage")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    jobs = []
    for setup in args.only:
        exp, brain = SETUPS[setup]
        for genome in GENOMES:
            picks = pick_e1(args.data, brain, genome) if exp == "e1" else pick_e2(args.data, brain, genome)
            jobs += [{**p, "setup": setup, "genome": genome, "out": str(args.out)} for p in picks]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"{len(jobs)} individuals", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool:
        rows = []
        for row in pool.imap_unordered(render_job, jobs):
            rows.append(row)
            print(f"{row['setup']} {row['genome']} {row['rank']}: stored {row['stored_score']:.3f}, "
                  f"replayed {row['replayed_score']:.3f}, {row['n_modules']} modules", flush=True)
    df = pd.DataFrame(rows).sort_values(["setup", "genome", "rank"])
    df.to_csv(args.out / "manifest.csv", index=False)
    if args.limit:
        return
    for setup in args.only:
        ranks = list(E1_RANKS if SETUPS[setup][0] == "e1" else E2_RANKS)
        montage(args.out / setup, ranks, args.out / f"{setup}_montage.mp4")
        body_sheet(args.out / setup, ranks, args.out / f"{setup}_bodies.png")
    print("largest |stored - replayed|:", f"{df['diff'].abs().max():.2e}", flush=True)


if __name__ == "__main__":
    main()

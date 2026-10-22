"""Interactive gait-similarity map from gait_similarity.py output.

Writes a self-contained HTML page with the 2D MDS maps of plot_gait_similarity.py
(same embeddings, sized by hinge count, coloured by champion x-speed or by
cycle amplitude relative to body size), one per comparison in --frames with a
switch between them (the first is shown on load; later maps are rotated onto
the first so the switch does not flip the layout). When gait_similarity.py wrote
the _ignore variants, a second switch picks whether the larger robot's
unmatched modules are charged their best match or ignored. Hovering a gait shows its
champion video, its behavioural descriptors and the other gaits ranked by
distance; clicking pins it.

The descriptors are symmetry_pressure/behavior_common.compute_behavior_descriptors
on a full-rate replay of each champion with the zoo episode loop (15 s from
rest, no settle, 20 Hz control straight to ctrl). The reward axis is +x, the
zoo fitness direction. Replays are cached in gait_descriptors_<tag>.csv.

Videos come from render_zoo_best.py (<root>/best_videos/<body>_<brain>.mp4) and
are transcoded to H.264 next to the page (videos/), which references them by
relative path: keep the page and videos/ together.

Usage:
  python gait_similarity_page.py __data__/ariel_zoo_benchmark [--tag ann] [--frames cycle_shape cycle]
Writes <root>/gait_similarity/gait_similarity_<tag>.html.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
import torch
from scipy.linalg import orthogonal_procrustes

from gait_similarity import FRAMES, LEFTOVER_SUFFIX, LEFTOVERS, load_recordings, select_runs
from plot_gait_similarity import FRAME_TITLE, FRAME_UNIT, mds_embedding
from zoo_benchmark import CORE_BODY, build_world, get_state_from_data, make_brain_for
from behavior_common import DESCRIPTOR_NAMES, _robot_bodies, compute_behavior_descriptors  # via brains' sys.path
from shared import floor_id, rotor_geom_ids, signed_vertical_yaw_delta

ZOO_AXIS_ANGLE_DEG = 90.0  # rotates shared.FORWARD_AXIS (0, -1) onto +x
SIMILAR_TOP = 5            # nearest gaits joined by lines on the map
PAGE_TEMPLATE = Path(__file__).with_name("gait_similarity_page.html")

DESCRIPTOR_GROUPS = {
    "Locomotion": DESCRIPTOR_NAMES[0:8],
    "Posture": DESCRIPTOR_NAMES[8:12],
    "Contact": DESCRIPTOR_NAMES[12:17],
    "Gait": DESCRIPTOR_NAMES[17:20],
    "Left/right symmetry": DESCRIPTOR_NAMES[20:25],
    "Effort": DESCRIPTOR_NAMES[25:27],
}
CYCLE_STATS = {"period_s": "gait cycle period (s)", "autocorr": "cycle autocorrelation",
               "cycle_rms_m": "cycle amplitude, RMS (m)", "body_size_m": "body size, RMS (m)",
               "rel_amplitude": "amplitude / body size"}
FRAME_NAME = {"world": "World frame", "core": "Core frame", "cycle": "Gait cycle (m)",
              "cycle_shape": "Gait cycle shape"}


# ── Behavioural descriptors ───────────────────────────────────────────────────


def record_behavior(run_dir: Path) -> dict[str, float]:
    """Replay run_dir's champion physics step by physics step; descriptors + x-speed."""
    torch.set_num_threads(1)
    cfg = json.loads((run_dir / "config.json").read_text())
    model, data = build_world(cfg["body"])
    brain = make_brain_for(cfg["brain"], model, data, cfg["duration"], cfg["body"])
    brain.set_params(np.load(run_dir / "champion.npy"))
    steps_per_ctrl = round(1.0 / (cfg["control_freq"] * model.opt.timestep))
    core_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CORE_BODY)
    floor, rotors = floor_id(model), rotor_geom_ids(model)
    bodies = _robot_bodies(model)
    body_col = {b: i for i, b in enumerate(bodies)}
    hinges = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    qpos_adr = model.jnt_qposadr[hinges]

    mujoco.mj_resetData(model, data)
    mujoco.mj_kinematics(model, data)
    core_R = data.xmat[core_id].reshape(3, 3)
    anchors = (data.xanchor[hinges] - data.xpos[core_id]) @ core_R
    mujoco.mj_forward(model, data)
    brain.reset()
    x0 = float(data.xpos[core_id, 0])

    core_xy, core_z, tilt, yaw, angles, contact, ctrl = [], [], [], [], [], [], []
    r_prev = data.xmat[core_id].reshape(3, 3).copy()
    prev_rotors: set[int] = set()
    c_hinge = 0
    for _ in range(round(cfg["duration"] * cfg["control_freq"])):
        data.ctrl[:] = brain.act(float(data.time), get_state_from_data(data))
        ctrl.append(data.ctrl.copy())
        for _ in range(steps_per_ctrl):
            mujoco.mj_step(model, data)
            mask, rotors_now = np.zeros(len(bodies), dtype=bool), set()
            for c in data.contact[:data.ncon]:
                g1, g2 = int(c.geom1), int(c.geom2)
                other = g2 if g1 == floor else g1 if g2 == floor else -1
                if other < 0:
                    continue
                col = body_col.get(int(model.geom_bodyid[other]))
                if col is not None:
                    mask[col] = True
                if other in rotors:
                    rotors_now.add(other)
            c_hinge += len(rotors_now - prev_rotors)
            prev_rotors = rotors_now
            r_curr = data.xmat[core_id].reshape(3, 3).copy()
            yaw.append(signed_vertical_yaw_delta(r_prev, r_curr))
            r_prev = r_curr
            core_xy.append(data.xpos[core_id, :2].copy())
            core_z.append(data.xpos[core_id, 2])
            tilt.append(math.acos(max(-1.0, min(1.0, float(r_curr[2, 2])))))
            angles.append(data.qpos[qpos_adr].copy())
            contact.append(mask)

    desc = compute_behavior_descriptors(
        core_xy=np.array(core_xy), core_z=np.array(core_z), core_tilt=np.array(tilt),
        yaw_delta=np.array(yaw), joint_angles=np.array(angles).reshape(-1, len(hinges)),
        body_contact=np.array(contact).reshape(-1, len(bodies)), ctrl_applied=np.array(ctrl),
        joint_anchors_core=anchors, dt=float(model.opt.timestep),
        axis_angle_deg=ZOO_AXIS_ANGLE_DEG, c_hinge=c_hinge,
    )
    return {**desc, "xspeed": (float(data.xpos[core_id, 0]) - x0) / float(data.time)}


def load_descriptors(root: Path, runs: pd.DataFrame, cache: Path, workers: int) -> pd.DataFrame:
    labels = list(runs.index)
    if cache.exists():
        df = pd.read_csv(cache, index_col=0)
        if list(df.index) == labels:
            print(f"Loaded descriptors from {cache}")
            return df
    run_dirs = [root / runs.loc[label, "run_dir"] for label in labels]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        df = pd.DataFrame(list(pool.map(record_behavior, run_dirs)), index=labels)
    for label in labels:
        if not np.isclose(df.loc[label, "xspeed"], runs.loc[label, "best_xspeed"], atol=1e-6):
            print(f"WARNING: {label} replay x-speed {df.loc[label, 'xspeed']:+.4f} "
                  f"vs stored {runs.loc[label, 'best_xspeed']:+.4f}")
    df.to_csv(cache)
    print(f"Replayed {len(df)} champions -> {cache}")
    return df


# ── Page data ─────────────────────────────────────────────────────────────────


def _num(x: float) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)


def rank_similar(dist: pd.DataFrame, align: pd.DataFrame | None) -> dict[str, list[dict]]:
    """Per gait, every other gait sorted by distance, with its best alignment if known."""
    lookup = {}
    if align is not None:
        for row in align.itertuples(index=False):
            info = {"mirrored": bool(row.mirrored), "yaw_deg": float(row.yaw_deg),
                    "phase_shift": float(row.phase_shift)}
            lookup[(row.a, row.b)] = lookup[(row.b, row.a)] = info
    ranked = {}
    for a in dist.index:
        others = dist.loc[a].drop(a).sort_values()
        ranked[a] = [{"label": b, "distance": float(d), **lookup.get((a, b), {})} for b, d in others.items()]
    return ranked


def frame_data(frame: str, dist: pd.DataFrame, align: pd.DataFrame | None, ref_xy: np.ndarray | None) -> dict:
    """MDS map and similarity ranking of one comparison; the map is rotated
    (and possibly reflected) onto ref_xy, scaled to its own units."""
    xy, stress = mds_embedding(dist.to_numpy())
    xy -= xy.mean(axis=0)
    if ref_xy is not None:
        xy = xy @ orthogonal_procrustes(xy, ref_xy)[0]
    return {
        "name": FRAME_NAME[frame], "title": FRAME_TITLE[frame], "unit": FRAME_UNIT[frame], "stress": stress,
        "xy": {label: [float(x), float(y)] for label, (x, y) in zip(dist.index, xy)},
        "similar": rank_similar(dist, align),
    }


def page_data(
    frames: dict[str, dict[str, tuple[pd.DataFrame, pd.DataFrame | None]]], meta: pd.DataFrame,
    desc: pd.DataFrame, cycles: pd.DataFrame, videos: dict[str, str], title: str,
) -> dict:
    """frames maps each comparison to {leftover: (distance matrix, alignment or None)},
    every comparison with the same leftover variants; the first of each is shown on load."""
    first = next(iter(frames.values()))
    labels = list(next(iter(first.values()))[0].index)
    out_frames, ref_xy = {}, None
    for frame, variants in frames.items():
        out_frames[frame] = {}
        for leftover, (dist, align) in variants.items():
            out_frames[frame][leftover] = frame_data(frame, dist.loc[labels, labels], align, ref_xy)
            if ref_xy is None:
                ref_xy = np.array([out_frames[frame][leftover]["xy"][label] for label in labels])
    gaits = []
    for label in labels:
        gaits.append({
            "label": label,
            "xspeed": float(meta.loc[label, "xspeed"]), "n_hinges": int(meta.loc[label, "n_hinges"]),
            "seed": int(meta.loc[label, "seed"]), "video": videos.get(label),
            "descriptors": {k: _num(desc.loc[label, k]) for k in DESCRIPTOR_NAMES},
            "cycle": {k: _num(cycles.loc[label, k]) for k in CYCLE_STATS},
        })
    return {
        "title": title, "frames": out_frames, "leftovers": list(first), "gaits": gaits,
        "groups": {**DESCRIPTOR_GROUPS, "Gait cycle": list(CYCLE_STATS)},
        "cycle_names": CYCLE_STATS, "top": SIMILAR_TOP,
    }


def browser_video(src: Path, out_dir: Path) -> Path:
    """H.264 copy of src in out_dir (render_zoo_run writes MPEG-4 Part 2, which
    browsers do not play); reused if newer than src."""
    out = out_dir / src.name
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "26", "-movflags", "+faststart", "-an", str(out)],
                       check=True)
    return out


def video_paths(root: Path, runs: pd.DataFrame, brain: str, page_dir: Path) -> dict[str, str]:
    """Video path per label relative to page_dir (transcoded into page_dir/videos);
    warns when a video is missing or from another seed."""
    video_dir = root / "best_videos"
    best = pd.read_csv(video_dir / "best_runs.csv") if (video_dir / "best_runs.csv").exists() else None
    out = {}
    for label in runs.index:
        path = video_dir / f"{label}_{brain}.mp4"
        if not path.exists():
            print(f"WARNING: no video {path}")
            continue
        if best is not None:
            row = best[(best["body"] == label) & (best["brain"] == brain)]
            if not row.empty and row["run_dir"].iloc[0] != runs.loc[label, "run_dir"]:
                print(f"WARNING: {path.name} is {row['run_dir'].iloc[0]}, not {runs.loc[label, 'run_dir']}")
        out[label] = browser_video(path, page_dir / "videos").relative_to(page_dir).as_posix()
    return out


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="zoo benchmark output dir with summary.csv")
    parser.add_argument("--tag", default="ann", help="brain whose best gait per body is shown")
    parser.add_argument("--frames", nargs="+", default=["cycle_shape", "cycle"], choices=FRAMES,
                        help="comparisons to switch between; the first is shown on load")
    parser.add_argument("--data-dir", type=Path, default=None, help="gait_similarity.py output, default <root>/gait_similarity")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    data_dir = args.data_dir or args.root / "gait_similarity"

    runs = select_runs(args.root, args.tag, None)
    labels, recs = load_recordings(data_dir / f"gait_trajectories_{args.tag}.npz")
    if labels != list(runs.index):
        raise SystemExit(f"Recordings in {data_dir} do not match summary.csv; rerun gait_similarity.py")
    meta = pd.DataFrame(recs, index=labels)
    # Leftover variants offered: those gait_similarity.py wrote for every comparison.
    leftovers = [lo for lo in LEFTOVERS if all(
        (data_dir / f"gait_distance_{args.tag}_{frame}{LEFTOVER_SUFFIX[lo]}.csv").exists() for frame in args.frames)]
    if not leftovers:
        raise SystemExit(f"No gait_distance_{args.tag}_<frame>.csv for {args.frames} in {data_dir}; run gait_similarity.py")
    frames = {}
    for frame in args.frames:
        frames[frame] = {}
        for leftover in leftovers:
            lsuffix = LEFTOVER_SUFFIX[leftover]
            dist = pd.read_csv(data_dir / f"gait_distance_{args.tag}_{frame}{lsuffix}.csv", index_col=0)
            align_path = data_dir / f"gait_alignment_{args.tag}{'' if frame == 'cycle' else '_' + frame}{lsuffix}.csv"
            is_cycle = frame in ("cycle", "cycle_shape")
            frames[frame][leftover] = (dist, pd.read_csv(align_path) if is_cycle and align_path.exists() else None)
    cycles = pd.read_csv(data_dir / f"gait_cycles_{args.tag}.csv", index_col="label")
    missing = [k for k in CYCLE_STATS if k not in cycles]
    if missing:
        raise SystemExit(f"gait_cycles_{args.tag}.csv lacks {missing}; rerun gait_similarity.py")
    desc = load_descriptors(args.root, runs, data_dir / f"gait_descriptors_{args.tag}.csv", args.workers)

    data = page_data(frames, meta, desc, cycles, video_paths(args.root, runs, args.tag, data_dir),
                     f"Gait similarity ({args.tag})")
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out = data_dir / f"gait_similarity_{args.tag}.html"
    out.write_text(PAGE_TEMPLATE.read_text().replace("/*__DATA__*/null", payload))
    stress = ", ".join(f"{f}{LEFTOVER_SUFFIX[lo]} {d['stress']:.3f}"
                       for f, variants in data["frames"].items() for lo, d in variants.items())
    print(f"Saved {out}  (MDS stress-1: {stress})")


if __name__ == "__main__":
    main()

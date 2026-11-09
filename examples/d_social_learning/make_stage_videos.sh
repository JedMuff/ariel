#!/bin/bash
#
# Evolution-stage videos: for every scheme x {0, 0.5, 1} in ARIEL (ig10) and
# EvoGym, the run with the highest final displacement; at generations 0, 15,
# 30, 45 and 60, the best, median and worst survivor, with an info panel
# (displacement, novelty, selection score, head start, scheme, x) and a live
# displacement bar. 360 videos per domain, in __data__/videos/{ariel,evogym}/,
# listed in __data__/videos/manifest.csv.
#
# 1. select_stage_videos.py writes the manifest (main venv)
# 2. two detached tmux sessions render: vid-ariel (main venv, MuJoCo on the
#    GPU via EGL) and vid-evogym (evogym-venv, Python 3.10, under xvfb-run).
#    Logs: __data__/videos/{ariel,evogym}.log. Both skip finished videos, so
#    rerunning this script only fills in what is missing.
#
# Needs: evogym-venv (see run_social_evogym_budget.sh, plus pillow) and xvfb.
#
# Usage (from the repo root):
#   bash examples/d_social_learning/make_stage_videos.sh
#   tmux attach -t vid-ariel      # or vid-evogym; detach with Ctrl-b d
set -euo pipefail

WORKERS=${WORKERS:-16}
V=examples/d_social_learning/make_videos
mkdir -p __data__/videos

uv run "$V/select_stage_videos.py"

tmux new-session -d -s vid-ariel \
    "MUJOCO_GL=egl uv run $V/render_stage_videos_ariel.py --workers $WORKERS 2>&1 | tee -a __data__/videos/ariel.log"
tmux new-session -d -s vid-evogym \
    "PYTHONPATH=src xvfb-run -a -s '-screen 0 1280x1024x24' evogym-venv/bin/python $V/render_stage_videos_evogym.py --workers $WORKERS 2>&1 | grep --line-buffered -v -i warn | tee -a __data__/videos/evogym.log"

echo "Rendering in tmux sessions vid-ariel and vid-evogym; logs in __data__/videos/{ariel,evogym}.log"

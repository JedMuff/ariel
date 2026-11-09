"""One simulated episode of one stored individual, observed by recorders.

The experiment-specific part of replaying an individual — how its body is
built, which brain drives it and how the episode is stepped — lives behind
:class:`Rollout`, which an adapter's ``make_rollout`` returns. Everything
generic (video, trajectories, behavioural descriptors) is a *recorder*: a
callable ``recorder(sim_step, data)`` invoked after every recorded physics
step, the same hook ``run_episode`` functions in the examples already take.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import mujoco

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class Recorder(Protocol):
    def __call__(self, sim_step: int, data: mujoco.MjData) -> None: ...


@dataclass
class Rollout:
    """A ready-to-run episode.

    ``run(recorder)`` plays the whole episode, calling
    ``recorder(sim_step, data)`` after every physics step that should be
    recorded (typically everything after a settle phase), and returns the
    episode's results (e.g. ``{"fitness": ...}``) for ``--verify``.
    """

    model: mujoco.MjModel
    data: mujoco.MjData
    run: Callable[[Recorder], dict[str, Any]]
    #: Name of the robot's core body; default: the first body named ``*core``.
    core_body: str | None = None
    #: Free-form extras an adapter wants to hand to custom recorders.
    info: dict[str, Any] = field(default_factory=dict)


def stepped_rollout(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    control: Callable[[mujoco.MjModel, mujoco.MjData], None],
    duration: float,
    *,
    settle: float = 0.0,
    ctrl_every: int = 1,
    core_body: str | None = None,
) -> Rollout:
    """A :class:`Rollout` for the common case: reset, step uncontrolled for
    ``settle`` seconds, then call ``control(model, data)`` every
    ``ctrl_every`` physics steps for ``duration`` seconds (e.g. a
    :class:`ariel.simulation.controllers.controller.Controller`'s callback or
    a CPG's ``forward``). Only the controlled part is recorded.
    """

    def run(recorder: Recorder) -> dict[str, Any]:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        step = 0
        while data.time < settle:
            mujoco.mj_step(model, data)
            step += 1
        start = data.qpos[:3].copy()
        end_time = settle + duration
        while data.time < end_time:
            if step % ctrl_every == 0:
                control(model, data)
            mujoco.mj_step(model, data)
            recorder(step, data)
            step += 1
        return {"displacement": float(data.qpos[0] - start[0])}

    return Rollout(model=model, data=data, run=run, core_body=core_body)


def play(rollout: Rollout, recorders: Sequence[Recorder]) -> dict[str, Any]:
    """Run ``rollout`` once, fanning every step out to all ``recorders``.
    Afterwards each recorder's ``close()`` is called if it has one (e.g. to
    finish a video); if the episode raised, ``abort()`` is called instead so
    no partial output is left behind.
    """

    def fanout(sim_step: int, data: mujoco.MjData) -> None:
        for rec in recorders:
            rec(sim_step, data)

    try:
        result = rollout.run(fanout)
    except BaseException:
        for rec in recorders:
            abort = getattr(rec, "abort", None)
            if abort is not None:
                abort()
        raise
    for rec in recorders:
        close = getattr(rec, "close", None)
        if close is not None:
            close()
    return result

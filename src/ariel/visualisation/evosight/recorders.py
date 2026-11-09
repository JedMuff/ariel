"""Generic recorders for a :class:`~.rollout.Rollout`: a compressed,
browser-playable video, and the trajectory behind the behavioural
descriptors of :mod:`ariel.utils.behavioural_descriptor`.

Both work for any ARIEL robot: they find the robot through its core body
(``*core``, e.g. ``robot1_core``) and need nothing from the experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

if TYPE_CHECKING:
    import os
    from collections.abc import Callable

# Video defaults: small enough that a run's worth of clips loads quickly in
# the browser (~50-200 KB per 15 s clip), large enough to read a gait.
VIDEO_WIDTH = 320
VIDEO_HEIGHT = 240
VIDEO_FPS = 15
VIDEO_CRF = (
    30  # x264 constant rate factor: 23 = default quality, +6 ~ half size
)
POSTER_QUALITY = 70
TRAJECTORY_HZ = 50.0


def find_core_body(model: mujoco.MjModel, name: str | None = None) -> int:
    """Body id of the robot's core: ``name`` if given, else the first body
    whose name ends in ``core`` (ARIEL's ``<prefix>core`` convention).
    """
    if name is not None:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            msg = f"no body named {name!r}"
            raise ValueError(msg)
        return bid
    for i in range(model.nbody):
        if (model.body(i).name or "").endswith("core"):
            return i
    msg = "no core body (a body named '*core'); pass Rollout(core_body=...)"
    raise ValueError(msg)


def robot_bodies(model: mujoco.MjModel, core: int) -> list[int]:
    """Geom-owning bodies of the robot, core first: the bodies sharing the
    core's name prefix (``robot1_core`` -> ``robot1_*``).
    """
    core_name = model.body(core).name
    prefix = core_name.removesuffix("core")
    owners = sorted({int(b) for b in model.geom_bodyid})
    bodies = [
        b for b in owners if b != core and model.body(b).name.startswith(prefix)
    ]
    return [core, *bodies]


def robot_radius(model: mujoco.MjModel, core: int) -> float:
    """Radius (m) of the robot around its core in the model's rest pose."""
    rest = mujoco.MjData(model)
    mujoco.mj_kinematics(model, rest)
    bodies = set(robot_bodies(model, core))
    geoms = [
        g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies
    ]
    if not geoms:
        return 0.1
    offsets = np.linalg.norm(rest.geom_xpos[geoms] - rest.xpos[core], axis=1)
    return float(np.max(offsets + model.geom_rbound[geoms]))


def tracking_camera(
    model: mujoco.MjModel,
    core: int,
    *,
    distance: float | None = None,
    azimuth: float = 90.0,
    elevation: float = -30.0,
) -> mujoco.MjvCamera:
    """Camera following the core, side-on from -y so +x motion reads
    left-to-right (as ``make_videos/make_best_videos.py``). By default the
    distance scales with the robot so it fills a small frame whatever its
    size; pass ``distance`` for a fixed framing across robots.
    """
    if distance is None:
        distance = float(np.clip(4.0 * robot_radius(model, core), 0.6, 4.0))
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = core
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


class VideoRecorder:
    """Render every ``1/fps`` simulated seconds and stream the frames to an
    H.264 MP4 (``yuv420p`` + ``faststart``, so browsers play it while it
    downloads), plus a WebP poster frame.

    Output is written to temporary files and renamed into place on
    :meth:`close`, so an existing ``path`` always means a complete video —
    which is what makes batch rendering resumable.

    Do not use :class:`ariel.utils.video_recorder.VideoRecorder` for this:
    its ``mp4v`` codec does not play in browsers.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        path: str | os.PathLike,
        *,
        poster_path: str | os.PathLike | None = None,
        width: int = VIDEO_WIDTH,
        height: int = VIDEO_HEIGHT,
        fps: int = VIDEO_FPS,
        crf: int = VIDEO_CRF,
        poster_at: float = 2.0,
        core_body: str | None = None,
        camera: Callable[[mujoco.MjModel, int], mujoco.MjvCamera] | None = None,
        overlay: Callable[[np.ndarray, float, mujoco.MjData], np.ndarray]
        | None = None,
    ) -> None:
        self.model = model
        # overlay(frame, t, data) -> frame, e.g. to draw text on each frame
        self.overlay = overlay
        self.path = Path(path)
        self.poster_path = Path(poster_path) if poster_path else None
        self.fps = fps
        self.crf = crf
        self.poster_at = poster_at
        core = find_core_body(model, core_body)
        self.camera = (camera or tracking_camera)(model, core)
        # The offscreen framebuffer defaults to 640x480; grow it if needed.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self._writer: Any = None
        self._tmp = self.path.with_name(
            f".{self.path.stem}.tmp{self.path.suffix}"
        )
        self._t0: float | None = None
        self._next_t = 0.0
        self._poster: np.ndarray | None = None
        self._last: np.ndarray | None = None
        self.n_frames = 0

    def __call__(self, sim_step: int, data: mujoco.MjData) -> None:  # noqa: ARG002
        if self._t0 is None:
            self._t0 = data.time
        t = data.time - self._t0
        if t + 1e-9 < self._next_t:
            return
        self._next_t += 1.0 / self.fps
        self.renderer.update_scene(data, camera=self.camera)
        frame = self.renderer.render()
        if self.overlay is not None:
            frame = self.overlay(frame, t, data)
        if self._writer is None:
            import imageio.v2 as imageio

            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = imageio.get_writer(
                self._tmp,
                fps=self.fps,
                codec="libx264",
                format="FFMPEG",
                macro_block_size=16,
                ffmpeg_log_level="error",
                ffmpeg_params=[
                    "-crf",
                    str(self.crf),
                    "-preset",
                    "slow",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                ],
            )
        self._writer.append_data(frame)
        self.n_frames += 1
        if self._poster is None and t >= self.poster_at:
            self._poster = frame.copy()
        self._last = frame

    def close(self) -> None:
        self.renderer.close()
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None
        Path(self._tmp).replace(self.path)
        if self.poster_path is not None:
            from PIL import Image

            poster = self._poster if self._poster is not None else self._last
            tmp = self.poster_path.with_name(f".{self.poster_path.name}.tmp")
            Image.fromarray(poster).save(
                tmp, format="WEBP", quality=POSTER_QUALITY
            )
            Path(tmp).replace(self.poster_path)

    def abort(self) -> None:
        self.renderer.close()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self._tmp.unlink(missing_ok=True)


class TrajectoryRecorder:
    """Record, at ``hz``, the traces :func:`behavior_descriptors` and
    :func:`rowing` need: body positions/quaternions (core first), hinge
    angles, per-body floor contact and the applied control.

    The hinge anchors (in the core frame, at rest) are taken from the
    model's reference pose when the recorder is created.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        hz: float = TRAJECTORY_HZ,
        core_body: str | None = None,
        floor_geom: str = "floor",
    ) -> None:
        self.model = model
        self.every = max(1, round(1.0 / (hz * model.opt.timestep)))
        self.dt = self.every * model.opt.timestep
        core = find_core_body(model, core_body)
        self.bodies = robot_bodies(model, core)
        self._body_arr = np.array(self.bodies)
        self.hinges = [
            j
            for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        ]
        self._qadr = np.array(
            [model.jnt_qposadr[j] for j in self.hinges], dtype=int
        )
        self.floor = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, floor_geom
        )

        rest = mujoco.MjData(model)
        mujoco.mj_kinematics(model, rest)
        self.anchors = (
            rest.xanchor[self.hinges] - rest.xpos[core]
        ) @ rest.xmat[core].reshape(3, 3)

        self._P: list[np.ndarray] = []
        self._Q: list[np.ndarray] = []
        self._A: list[np.ndarray] = []
        self._C: list[np.ndarray] = []
        self._U: list[np.ndarray] = []
        self._n = 0

    def __call__(self, sim_step: int, data: mujoco.MjData) -> None:  # noqa: ARG002
        n = self._n
        self._n += 1
        if n % self.every:
            return
        self._P.append(data.xpos[self.bodies].copy())
        self._Q.append(data.xquat[self.bodies].copy())
        self._A.append(data.qpos[self._qadr].copy())
        self._U.append(data.ctrl.copy())
        g1 = data.contact.geom1[: data.ncon]
        g2 = data.contact.geom2[: data.ncon]
        other = np.concatenate([g2[g1 == self.floor], g1[g2 == self.floor]])
        self._C.append(np.isin(self._body_arr, self.model.geom_bodyid[other]))

    def arrays(self) -> dict[str, np.ndarray]:
        """The recorded traces: P (T,B,3), Q (T,B,4), joint_angles (T,J),
        contact (T,B), ctrl (T,nu), plus anchors (J,3) and dt.
        """
        return {
            "P": np.asarray(self._P),
            "Q": np.asarray(self._Q),
            "joint_angles": np.asarray(self._A).reshape(
                len(self._A), len(self.hinges)
            ),
            "contact": np.asarray(self._C),
            "ctrl": np.asarray(self._U).reshape(len(self._U), self.model.nu),
            "anchors": self.anchors,
            "dt": np.float64(self.dt),
        }

    def descriptors(self) -> tuple[dict[str, float], dict[str, Any]]:
        """``(numeric descriptors, non-numeric extras)`` of the recorded
        episode; the gait label from :func:`rowing` is the main extra.
        """
        from ariel.utils.behavioural_descriptor import (
            behavior_descriptors,
            rowing,
        )

        a = self.arrays()
        if len(a["P"]) < 4:
            return {}, {}
        values: dict[str, Any] = {
            **behavior_descriptors(
                a["P"],
                a["Q"],
                a["joint_angles"],
                a["contact"],
                a["ctrl"],
                a["anchors"],
                self.dt,
            ),
            **rowing(a["P"], a["Q"], self.dt),
        }
        numeric: dict[str, float] = {}
        extras: dict[str, Any] = {}
        for k, v in values.items():
            if isinstance(
                v, (bool, np.bool_, int, float, np.integer, np.floating)
            ):
                numeric[k] = float(v)
            else:
                extras[k] = v
        return numeric, extras

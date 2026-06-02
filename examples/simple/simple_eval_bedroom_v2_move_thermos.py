#!/usr/bin/env python3
"""SIMPLE closed-loop eval specialized for G1WholebodyBedroomMoveThermos-v0.

Differs from the generic simple_eval.py in two ways:
  1. Filters baked-in scene objects from the composed_bedroom_v2 GSplat
     scene (lamp/cup/trashbin/vitamin_box/clock) so eval visuals match the
     datagen 3DGS replays — only the thermos (target=103) is rendered.
  2. Rewrites the per-episode env_conf at the dataset-loader boundary to
     override robot/thermos spawn positions. Necessary because eval replays
     the dataset's frozen env_conf via Task.load_state_dict() and does NOT
     consult the task file's dr_cfgs / _ROBOT_SPAWN_X constants.

Both patches affect ONLY this eval process. SIMPLE source is not modified,
and the policy server process (.venv-psi) does not import simple at all.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

# When MUJOCO_GL=egl, ensure PyOpenGL selects the EGL platform BEFORE OpenGL
# is imported anywhere (default on Linux without DISPLAY is GLXPlatform, which
# makes `from OpenGL import EGL` fail with `'GLXPlatform' object has no
# attribute 'EGL'`).
# Then, PyOpenGL <3.1.something is missing EGLDeviceEXT; mujoco.egl.egl_ext
# imports it unconditionally. Inject as opaque pointer so the import chain
# doesn't fail.
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import ctypes
    from OpenGL import EGL as _egl
    if not hasattr(_egl, "EGLDeviceEXT"):
        _egl.EGLDeviceEXT = ctypes.c_void_p

from simple.evals.api import EvalConfig, EvalRunner

# ─────────────────────────────────────────────────────────────────────────
# Eval-only patches (no effect on datagen / training / shared SIMPLE code)
# ─────────────────────────────────────────────────────────────────────────

# composed_bedroom_v2/objects_info.json lists 6 baked-in scene objects.
# GsplatSimulator auto-loads every entry into the renderer, even though
# the task's DR (`number_of_distractors=0`) only spawns the thermos.
# Block these specific IDs at the renderer.load_object boundary so the
# room background / table / thermos / robot links all still render.
_EVAL_BLOCK_OBJECT_IDS = {
    "101",  # lamp
    "102",  # cup
    "104",  # trashbin
    "105",  # vitamin_box
    "106",  # clock
}

# Eval replays env_conf snapshots from the LeRobot dataset, bypassing the
# task's dr_cfgs. To change spawn for eval only, fill this dict with
# `{actor_id: {axis: value}}`. Empty = use dataset's frozen positions.
# Table front edge (toward robot) is at X = +0.37 (table size [0.74, 1.63, 0.1]
# centered at origin); _GRASP_X = 0.58 is the canonical grasp pose. Override
# only robot X; keep the dataset's per-episode randomized Y.
_EVAL_SPAWN_OVERRIDES: dict = {
    "g1_wholebody": {"x": 0.55},
}


def _install_eval_patches() -> None:
    # 1) GSplat distractor filter
    from simple.engines import gsplat_renderer as _gr
    _orig_load = _gr.GSplatRenderer.load_object

    def _filtered_load(self, obj_id, *args, **kwargs):
        if obj_id in _EVAL_BLOCK_OBJECT_IDS:
            print(f"  [eval-filter] suppressed GSplat object {obj_id}")
            return None
        return _orig_load(self, obj_id, *args, **kwargs)

    _gr.GSplatRenderer.load_object = _filtered_load

    # 2) Spawn override (mutates env_conf before env.reset() consumes it)
    from simple.datasets import lerobot as _lr
    _orig_get = _lr.get_episode_lerobot

    def _patched_get(dataset, eps_idx, data_format=None):
        env_conf, episode = _orig_get(dataset, eps_idx, data_format)
        spatial = env_conf.get("dr_state_dict", {}).get("spatial", {})
        for actor_id, axes in _EVAL_SPAWN_OVERRIDES.items():
            if actor_id not in spatial:
                continue
            pos = list(spatial[actor_id]["position"])
            for axis, val in axes.items():
                pos["xyz".index(axis)] = float(val)
            spatial[actor_id]["position"] = pos
            print(f"  [eval-spawn] {actor_id} → pos={pos}")
        return env_conf, episode

    _lr.get_episode_lerobot = _patched_get


def _install_video_camera_filter(allowed: set[str]) -> None:
    # 3) VideoRecorder filter: only save the cameras whose obs keys are in
    #    `allowed`. The policy still receives ALL trained cameras — this
    #    purely affects which per-camera mp4s the recorder writes.
    #
    # Implementation: skip writer construction for non-allowed keys (instead
    # of releasing them afterward). Releasing a fresh VideoWriter writes a
    # 0-second placeholder mp4 + the _failed/_succeeded rename, which is
    # exactly what we don't want.
    if not allowed:
        return
    import os as _os
    import shutil as _shutil
    from simple.envs.wrappers import video_recorder as _vr
    from simple.envs.video_writer import VideoWriter as _VW
    import gymnasium as _gym

    def _filtered_reset(self, **kwargs):
        observations, info = _gym.Wrapper.reset(self, **kwargs)

        if kwargs.get("options") is not None:
            if kwargs["options"].get("task_id") is not None:
                self.name_prefix = kwargs["options"]["task_id"]
                video_folder = f"{self.work_dir}/{self.name_prefix}"
                if _os.path.exists(video_folder):
                    _shutil.rmtree(video_folder, ignore_errors=True)
                    print(f"Overwriting existing videos at {video_folder} folder")
                _os.makedirs(video_folder, exist_ok=True)

        self.video_writers = {}
        kept, dropped = [], []
        for key, subspace in self.unwrapped.observation_space.items():
            if len(subspace.shape) != 3 or subspace.shape[-1] != 3:
                continue
            if key not in allowed:
                dropped.append(key)
                continue
            self.video_writers[key] = _VW(
                f"{self.work_dir}/{self.name_prefix}/{key}.mp4",
                self.framerate,
                subspace.shape[:2][::-1],
                write_png=self.write_png,
            )
            self.video_writers[key].write(observations[key])
            kept.append(key)
        if dropped:
            print(f"  [eval-video] kept={sorted(kept)} dropped={sorted(dropped)}")

        self._is_released = False
        return observations, info

    _vr.VideoRecorder.reset = _filtered_reset
    # step() already iterates self.video_writers, which we've filtered above,
    # so no patch is needed there.


_install_eval_patches()


REPO_ROOT = Path(__file__).resolve().parents[2]


def _pick_free_port(start: int = 22085, end: int = 22999) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free port found in range {start}-{end}")


def _wait_for_port(host: str, port: int, tries: int, sleep_s: float) -> None:
    for _ in range(tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(sleep_s)
    raise TimeoutError(f"Timed out waiting for {host}:{port}")


def _build_server_cmd(args: argparse.Namespace, port: int) -> list[str]:
    cmd = [
        args.server_python or sys.executable,
        "-m",
        "psi.deploy.psi0_serve_simple",
        "--host",
        args.server_host,
        "--port",
        str(port),
        "--policy",
        args.policy,
        "--run-dir",
        args.run_dir,
        "--ckpt-step",
        str(args.ckpt_step),
        "--device",
        args.device,
    ]
    if args.action_exec_horizon is not None:
        cmd.extend(["--action-exec-horizon", str(args.action_exec_horizon)])
    if args.rtc:
        cmd.append("--rtc")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SIMPLE closed-loop eval for bedroom_v2 move-thermos with eval-only scene/spawn patches."
    )
    parser.add_argument("--run-dir", required=True, help="PSI run directory.")
    parser.add_argument("--ckpt-step", required=True, help="Checkpoint step or 'latest'.")
    parser.add_argument("--env-id", default="simple/G1WholebodyBedroomMoveThermos-v0")
    parser.add_argument("--policy", default="psi0")
    parser.add_argument("--data-dir", required=True, help="LeRobot dataset root for SIMPLE eval.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--server-host", default="0.0.0.0")
    parser.add_argument("--host", default="localhost", help="Host used by SIMPLE eval client.")
    parser.add_argument("--port", type=int, help="Policy server port. Defaults to a free port.")
    parser.add_argument("--sim-mode", default="mujoco_gsplat")
    parser.add_argument("--data-format", default="lerobot")
    parser.add_argument("--eval-dir", default="data/evals")
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=1500)
    parser.add_argument("--success-criteria", type=float, default=0.9)
    # Must be < the server's action_chunk_size (30): RTC needs the executed
    # horizon to leave an overlap of unexecuted actions to carry forward as
    # prev_actions. If exec_horizon == chunk_size there is no overlap, so the
    # first continuation call hits the server assertion "prev_actions ... must
    # be provided" and the rollout crashes (~step 90). 24 is the deployment value.
    parser.add_argument("--action-exec-horizon", type=int, default=24)
    parser.add_argument("--wait-tries", type=int, default=6000)
    parser.add_argument("--wait-sleep-s", type=float, default=0.1)
    parser.add_argument("--server-log", help="Optional path for server stdout/stderr.")
    parser.add_argument(
        "--server-python",
        help="Python interpreter for the policy server subprocess (default: same as eval client).",
    )
    parser.add_argument("--rtc", action="store_true")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--save-video", action="store_true", default=True)
    parser.add_argument("--no-save-video", dest="save_video", action="store_false")
    # Comma-separated obs-keys to record as .mp4. Empty string = record all
    # cameras. Default: head_stereo_left only.
    parser.add_argument("--video-cameras", default="head_stereo_left")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    allowed_video_cams = {c.strip() for c in args.video_cameras.split(",") if c.strip()}
    _install_video_camera_filter(allowed_video_cams)

    port = args.port or _pick_free_port()
    server_cmd = _build_server_cmd(args, port)
    eval_config = EvalConfig(
        env_id=args.env_id,
        policy=args.policy,
        split=args.split,
        host=args.host,
        port=port,
        data_format=args.data_format,
        sim_mode=args.sim_mode,
        headless=args.headless,
        eval_dir=args.eval_dir,
        max_episode_steps=args.max_episode_steps,
        num_episodes=args.num_episodes,
        episode_start=args.episode_start,
        data_dir=args.data_dir,
        success_criteria=args.success_criteria,
        save_video=args.save_video,
        num_workers=args.num_workers,
    )

    print(f"server_cmd={' '.join(server_cmd)}")
    print(f"eval_config={eval_config}")
    print(f"eval patches: block_objects={_EVAL_BLOCK_OBJECT_IDS}, spawn_overrides={_EVAL_SPAWN_OVERRIDES or '(none)'}")
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}".rstrip(":")

    log_handle = None
    server = None
    try:
        if args.server_log:
            log_path = Path(args.server_log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("w")
            server = subprocess.Popen(
                server_cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        else:
            server = subprocess.Popen(server_cmd, cwd=REPO_ROOT, env=env)

        _wait_for_port(args.host, port, args.wait_tries, args.wait_sleep_s)
        result = EvalRunner(eval_config).run()
        print(f"success_rate={result.success_rate:.6f}")
        print(f"log_path={result.log_path}")
        print(f"eval_dir={result.eval_dir}")
        return 0
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())

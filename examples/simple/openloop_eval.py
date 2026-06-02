"""Open-loop evaluation for Psi0 on one or more episodes.

Mirrors examples/simple/openloop_eval.ipynb as a runnable script.

Splits: the LeRobot config only natively wires `train` and `val` repo-ids;
the dataset wrapper routes any non-"train" split through `val_repo_ids`.
So `--split test` is implemented by pointing `val_repo_ids` at the `test`
repo and loading it through the val branch. `--data-root` lets you swap the
on-disk dataset (e.g. a fast re-encoded mirror) without touching the run config.
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor

# ---------------------------------------------------------------------------
# Locate project root and chdir into it so relative paths (.runs/, etc.) work
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent
while project_root != project_root.parent and not (project_root / "pyproject.toml").exists():
    project_root = project_root.parent
os.chdir(project_root)

from psi.utils import parse_args_to_tyro_config, seed_everything
from psi.config.config import LaunchConfig

CKPT_STEP = 12960

# Action component labels and split boundaries (36-dim action)
LABELS = [
    "hand_joints",   # [0:14]
    "arm_joints",    # [14:28]
    "torsor_roll",   # [28:31]  (torso rpy, 3-dim)
    "torsor_pitch",  # [31:32]
    "torsor_yaw",    # [32:33]
    "height",        # [33:34]
    "vx",            # [34:35]
    "vy",            # [35:36]
    "torso_vyaw",    # unused in print split but used in plot split
    "target_yaw",
]
PRINT_SPLITS = [14, 28, 31, 32, 33, 34, 35]   # 8 groups
PLOT_SPLITS  = [14, 28, 29, 30, 31, 32, 33, 34, 35]  # 10 groups


def parse_args():
    p = argparse.ArgumentParser(description="Open-loop eval for Psi0")
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Path to the run directory (contains argv.txt and run_config.json)")
    p.add_argument("--ckpt-step", type=int, default=CKPT_STEP,
                   help="Checkpoint step to load (looks for checkpoints/ckpt_<step>)")
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"],
                   help="Which split to evaluate on (test routes through val_repo_ids -> test)")
    p.add_argument("--data-root", type=str, default=None,
                   help="Override dataset root_dir (e.g. a fast re-encoded mirror)")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU index to use (e.g. 0 → cuda:0)")
    p.add_argument("--eps-idx", type=int, default=None,
                   help="Single episode index to evaluate (random if not set)")
    p.add_argument("--episodes", type=str, default=None,
                   help="Comma-separated explicit episode indices, e.g. 0,1,2,3")
    p.add_argument("--num-episodes", type=int, default=None,
                   help="Evaluate the first N episodes (0..N-1)")
    p.add_argument("--stride", type=int, default=4,
                   help="Frame stride within the episode")
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--no-plot", action="store_true", help="Skip saving per-episode plots")
    p.add_argument("--output-dir", type=str, default=".",
                   help="Directory to save the output plot")
    return p.parse_args()


def load_config(run_dir: Path) -> LaunchConfig:
    config_: LaunchConfig = parse_args_to_tyro_config(run_dir / "argv.txt")  # type: ignore
    conf = (run_dir / "run_config.json").read_text()
    launch_config = config_.model_validate_json(conf)
    return launch_config


def eval_episode(dataset, psi0, maxmin, eps_idx, args, device):
    """Run open-loop eval over one episode; returns (per_frame_errors (T,Da), mean_error (Da,))."""
    start_frame = dataset.raw_dataset.base_dataset.episode_data_index["from"][eps_idx].item()
    end_frame   = dataset.raw_dataset.base_dataset.episode_data_index["to"][eps_idx].item()
    print(f"\nEpisode {eps_idx}: frames {start_frame}..{end_frame}  ({end_frame - start_frame} frames)")

    per_frame_errors = []
    for i in tqdm(range(start_frame, end_frame, args.stride),
                  desc=f"eps {eps_idx}", unit="frame", leave=False):
        frame = dataset[i]
        batch_images       = [frame["raw_images"]]
        batch_instructions = [frame["instruction"]]
        batch_states       = torch.from_numpy(frame["states"]).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_actions = psi0.predict_action(
                observations=batch_images,
                states=batch_states,
                instructions=batch_instructions,
                num_inference_steps=args.num_inference_steps,
                traj2ds=None,
            )

        gt_action   = torch.from_numpy(frame["raw_actions"]).unsqueeze(0).to(device)
        denorm_pred = maxmin.denormalize(pred_actions)
        error_l1    = (denorm_pred - gt_action).abs().detach().cpu().numpy()
        error_l1    = error_l1.reshape(-1, gt_action.shape[-1])   # (Tp, Da)
        per_frame_errors.append(error_l1.mean(0))                 # (Da,)

    per_frame_errors = np.stack(per_frame_errors, axis=0)   # (T, Da)
    return per_frame_errors, per_frame_errors.mean(axis=0)  # (Da,)


def print_summary(mean_error, header):
    print(f"\n--- {header} ---")
    for i, seg in enumerate(np.split(mean_error, PRINT_SPLITS)):
        print(f"denormed_err_l1_{LABELS[i]} {seg.shape}:  {np.linalg.norm(seg):.6f}")


def save_plot(per_frame_errors, eps_idx, args, out_dir):
    error_groups   = np.split(per_frame_errors, PLOT_SPLITS, axis=-1)
    per_label_norm = [np.linalg.norm(g, axis=-1) for g in error_groups]
    curve_map      = dict(zip(LABELS, per_label_norm))
    plot_groups = [
        ("hand_joints + arm_joints (rad)", ["hand_joints", "arm_joints"]),
        ("torso rpy (rad)",                ["torsor_roll", "torsor_pitch", "torsor_yaw"]),
        ("height (m)",                     ["height"]),
        ("vx + vy (m/s)",                  ["vx", "vy"]),
        ("target_yaw (rad)",               ["target_yaw"]),
    ]
    fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
    for ax, (title, keys) in zip(axes, plot_groups):
        for key in keys:
            if key in curve_map:
                ax.plot(curve_map[key], label=key)
        ax.set_title(title)
        ax.set_ylabel("Error norm")
        ax.grid(True, alpha=0.3)
        if len(keys) > 1:
            ax.legend()
    axes[-1].set_xlabel(f"Sample step in episode (stride={args.stride})")
    plt.suptitle(f"[{args.split}] Episode {eps_idx}  |  ckpt step {args.ckpt_step}", y=1.01)
    plt.tight_layout()
    out_path = out_dir / f"openloop_{args.split}_ckpt{args.ckpt_step}_eps{eps_idx}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {out_path}")


def main():
    args = parse_args()
    run_dir = args.run_dir

    launch_config = load_config(run_dir)
    seed_everything(launch_config.seed or 42)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    from psi.models.psi0 import Psi0Model

    device = f"cuda:{args.gpu}"
    psi0 = Psi0Model.from_pretrained(run_dir, args.ckpt_step, launch_config, device=device)
    psi0.to(device)
    psi0.eval()
    print(f"Model loaded (ckpt {args.ckpt_step}).")

    # ------------------------------------------------------------------
    # Load dataset (split + optional data-root override)
    # ------------------------------------------------------------------
    from psi.config.data_lerobot import LerobotDataConfig

    data_cfg: LerobotDataConfig = launch_config.data  # type: ignore
    maxmin = data_cfg.transform.field

    if args.data_root is not None:
        print(f"Overriding data root_dir -> {args.data_root}")
        data_cfg.root_dir = args.data_root

    # The wrapper maps split=='train' -> train_repo_ids, else -> val_repo_ids.
    # For test, repoint val_repo_ids at the test repo and load via the val branch.
    if args.split == "test":
        data_cfg.val_repo_ids = ["test"]
        load_split = "val"
    else:
        load_split = args.split

    transform_kwargs = dict(vlm_processor=psi0.vlm_processor)
    dataset = data_cfg(split=load_split, transform_kwargs=transform_kwargs)

    total_episodes = dataset.raw_dataset.meta.total_episodes
    print(f"split={args.split}  total_episodes={total_episodes}  repo_dir={data_cfg.root_dir}")

    # ------------------------------------------------------------------
    # Resolve episode list
    # ------------------------------------------------------------------
    if args.episodes is not None:
        eps_list = [int(x) for x in args.episodes.split(",") if x.strip() != ""]
    elif args.eps_idx is not None:
        eps_list = [args.eps_idx]
    elif args.num_episodes is not None:
        eps_list = list(range(min(args.num_episodes, total_episodes)))
    else:
        eps_list = [int(np.random.randint(0, total_episodes))]
    print(f"Episodes to eval: {eps_list}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Eval loop over episodes (model loaded once)
    # ------------------------------------------------------------------
    all_mean_errors = []
    ok_eps = []
    for eps_idx in eps_list:
        try:
            per_frame_errors, mean_error = eval_episode(dataset, psi0, maxmin, eps_idx, args, device)
        except Exception as e:  # tolerate corrupt episodes
            print(f"!! episode {eps_idx} failed ({type(e).__name__}: {e}); skipping")
            continue
        print_summary(mean_error, f"ckpt {args.ckpt_step}  split={args.split}  eps {eps_idx}")
        if not args.no_plot:
            save_plot(per_frame_errors, eps_idx, args, out_dir)
        all_mean_errors.append(mean_error)
        ok_eps.append(eps_idx)

    # ------------------------------------------------------------------
    # Aggregate across episodes
    # ------------------------------------------------------------------
    if len(all_mean_errors) > 1:
        overall = np.stack(all_mean_errors, axis=0).mean(axis=0)
        print_summary(overall, f"OVERALL ckpt {args.ckpt_step}  split={args.split}  "
                               f"episodes={ok_eps} ({len(ok_eps)} eps)")
    elif not all_mean_errors:
        print("No episodes succeeded.")


if __name__ == "__main__":
    main()

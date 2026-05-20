# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Ψ₀ (Psi0) is a vision-language-action (VLA) foundation model for humanoid loco-manipulation. The model has two end-to-end trained components: a Qwen3-VL-2B-Instruct backbone (System-2) and a ~500M-parameter flow-based multimodal diffusion transformer action expert (System-1). An RL-based tracking controller (System-0) sits outside this repo. Pre-training uses egocentric human video (EgoDex, Humanoid-Everyday); post-/fine-training uses teleoperated robot data.

## Environment setup

Python 3.10, managed by **uv**. The project requires `.env` (copy from `.env.sample`); training scripts call `dotenv.load_dotenv()` and assert it succeeds — runs fail if `.env` is missing. Key env vars:
- `PSI_HOME` — root for cache/checkpoints/data (by convention `/hfm`)
- `HF_TOKEN`, `WANDB_API_KEY`, `WANDB_ENTITY`
- `HF_LEROBOT_HOME`, `HF_HOME`, `TORCH_HOME`, `UV_CACHE_DIR`

The Psi0 env lives in `.venv-psi`:
```
uv venv .venv-psi --python 3.10
source .venv-psi/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync --group serve --group viz --group psi --index-strategy unsafe-best-match --active
uv pip install flash_attn==2.7.4.post1 --no-build-isolation
```
For SIMPLE-based simulation eval, `git submodule update --init --recursive` then `uv sync --all-groups ...` and run `scripts/install_curobo.sh`.

**CUDA toolkit gotcha (this machine).** The default `/usr/bin/nvcc` is **CUDA 11.5** — too old for `flash_attn 2.7.4.post1` (requires ≥11.7) and mismatched against `torch 2.7.0+cu126`. Any shell that builds CUDA extensions (flash_attn, custom ops, anything invoking `nvcc`) **must** override the toolkit before running:
```
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```
Put these in `.env` so they load automatically with `source .env`. GPUs here are RTX 6000 Ada (sm_89). Don't pipe install commands through `tee` — `tee` swallows the non-zero exit code and makes failures look successful.

**Baselines each get their own venv (`.venv-dp`, `.venv-act`, `.venv-gr00t`, …) but share `src/`.** See `baselines/README.md`. Each baseline keeps its own torch/transformers pin in `baselines/<name>/requirements-*.txt`. Baselines that consume the patched LeRobot must `cp src/lerobot_patch/common/datasets/lerobot_dataset.py` into the baseline venv's `site-packages/lerobot/common/datasets/`.

Smoke checks:
```
python -c "import psi; print(psi.__version__)"
python -c "from psi.data.lerobot.compat import LEROBOT_LAYOUT; print(LEROBOT_LAYOUT)"
```

## Training

All training enters through **`scripts/train.py`**. The first positional argument is a config module name under `psi.config.train.*` (e.g. `finetune_real_psi0_config`); remaining args are parsed by `tyro` against that module's `DynamicLaunchConfig`. So every shell wrapper looks like:
```
torchrun --nproc_per_node=$NPROC --master_port=29500 scripts/train.py \
    <config_module> --train.x=... --data.y=... --model.z=...
```

`Trainer.instantiate()` then dispatches to `psi/trainers/<cfg.train.name>.py` by snake-case → `PascalCaseTrainer` (e.g. `train.name=finetune` → `psi/trainers/finetune.py::FinetuneTrainer`; also `pretrain`, `posttrain`, `act_g1`, `diffusion_policy_g1`). Adding a new training mode = new config module under `src/psi/config/train/` + new trainer module under `src/psi/trainers/`.

Pre-canned wrappers in `scripts/train/psi0/`:
- `finetune-real-psi0.sh <task>` — fine-tune on real G1 LeRobot data
- `finetune-simple-psi0.sh <task>` — fine-tune on SIMPLE sim data
- `pretrain-egodex-psi0-fast.sh`, `pretrain-he-psi0-fast.sh`, `pretrain-mix-psi0-fast.sh`
- `posttrain-he-psi0.sh`, `posttrain-mix-psi0.sh`

GPU selection via `CUDA_VISIBLE_DEVICES`; nproc is inferred from the comma count. Global batch size = device_batch × N_gpus × grad_accum — keep it at 128 (the value used for all published runs). For low-VRAM GPUs add `--train.optimizer-foreach=false`. Parallelism is selected via `--train.data_parallel={ddp,fsdp,deepspeed}`; DeepSpeed configs live in `scripts/deepspeed/zero{2,3,3_offload}.json`.

Runs land in `.runs/<exp>/...` with `run_config.json`, `argv.txt`, `envs.txt`, and `dataset_statistics.json` snapshot. `cfg.auto_tag_run` will `git add . && git commit && git tag <run_name>` at startup — be aware when running locally.

## Data

LeRobot is the canonical on-disk format. To convert raw teleop into LeRobot:
```
python scripts/data/raw_to_lerobot.py --data-root=... --work-dir=... --repo-id=... --robot-type=g1 --task=$task
python scripts/data/calc_modality_stats.py --work-dir=$PSI_HOME/data/real --task=$task
cp $PSI_HOME/data/real/$task/meta/stats.json $PSI_HOME/data/real/$task/meta/stats_psi0.json
```
Task → human-readable instruction mapping lives in `scripts/data/task_description_dict.json`.

The Psi0 LeRobot stack diverges from upstream — `src/psi/data/lerobot/compat.py` (`LEROBOT_LAYOUT`) and `src/lerobot_patch/` are the shim layer. If a dataset throws `stack(): argument 'tensors' (position 1) must be tuple of Tensors, not Column`, the env is on the legacy LeRobot — rerun `uv sync --group psi --active`. Real-world released datasets need `python scripts/data/patch_lerobot_meta.py $PSI_HOME/data/real/$task` before training.

Dataset adapters: `psi/data/egodex/` (EgoDex), `psi/data/humanoid/` (Humanoid-Everyday raw), `psi/data/lerobot/` (LeRobot + ext).

## Deployment / serving

Server: `serve_psi0` console script → `psi.deploy.psi0_serve_simple:main`. RTC-mode wrappers:
- `scripts/deploy/serve_psi0-rtc.sh` (requires `CHECKPOINT_DIR`, `CHECKPOINT_STEP`)
- `scripts/deploy/serve_psi0_simple.sh` for SIMPLE eval
- Client side: `real/scripts/deploy_psi0-rtc.sh`

For SIMPLE eval, two entrypoints depending on data origin: tasks ending `Teleop` use `eval_decoupled_wbc.py` + `psi0_decoupled_wbc`; tasks ending `MP` (CuRobo motion planning) use `eval.py` + `psi0`. Eval rollout videos land in `third_party/SIMPLE/data/evals/psi0/`.

## Tests

There is no `pytest` config and no formal test runner — `tests/` and `scripts/test/` are sparse standalone scripts. Run them directly with `python tests/<file>.py` from an activated venv. `tests/test_simple_datagen_curobo_missing.py` is an import-path sanity check that requires the SIMPLE submodule initialized and `third_party/SIMPLE/src` on `PYTHONPATH`.

`scripts/test_regression.py` exists for ad-hoc regression checks; review its top-of-file constants before running.

## Repo layout (selective)

- `src/psi/` — the actual model code. `config/` (Pydantic + tyro), `config/train/` (per-run launch configs), `models/psi0.py` (the model), `trainers/` (per-mode trainers), `data/` (dataset adapters + LeRobot compat), `tokenizer/fast_action_tokenizer.py` (FAST action tokenizer), `deploy/` (FastAPI serve scripts).
- `src/{act,dp,gr00t,h_rdt,egovla,openpi,InternVLA-M1,fast}/` — per-baseline source trees; each is editable-installed into its own venv.
- `baselines/<name>/` — per-baseline requirements + train/deploy/openloop shell scripts + README.
- `scripts/train/psi0/*.sh` — canonical training launchers (read these before constructing new commands).
- `scripts/{data,deploy,viz,test,deepspeed}/` — supporting tooling.
- `real/` — physical-robot teleop + deployment code (separate `pyproject.toml`).
- `third_party/SIMPLE/` — git submodule, simulator used for benchmarking.
- `.runs/` — run outputs (gitignored).

## Conventions worth knowing

- Don't bypass the config-module dispatch. New training variants belong in `src/psi/config/train/<name>_config.py` plus a wrapper under `scripts/train/...`, not as ad-hoc CLI changes.
- The `psi` package and the baselines all share `src/`. Changes to `psi.data`, `psi.models`, etc. affect every baseline venv. When touching shared code, sanity-check at least one baseline that exercises it.
- LeRobot integration is forked. Patches go in `src/lerobot_patch/`, not directly into venvs (the baseline READMEs copy from there).
- `scripts/install_curobo.sh` requires `UV_PROJECT_ENVIRONMENT` to point at the active venv.

# Ψ₀ (Psi0) — Models, Shapes, and the SIMPLE Closed-Loop Eval

A reference for the **bedroom move-thermos** fine-tune + SIMPLE closed-loop eval. All concrete
numbers below are taken from
[`scripts/train/psi0/finetune-bedroom-grasp-thermos.sh`](scripts/train/psi0/finetune-bedroom-grasp-thermos.sh),
[`src/psi/models/psi0.py`](src/psi/models/psi0.py),
[`src/psi/deploy/psi0_serve_simple.py`](src/psi/deploy/psi0_serve_simple.py),
and the SIMPLE simulator in the sibling repo `simple-3dgs/`.

---

## 0. The two-system picture

| System | What it is | Where it lives |
|---|---|---|
| **System-2** | Qwen3-VL-2B-Instruct backbone (vision-language). Frozen in this fine-tune. | `self.vlm_model` in [psi0.py](src/psi/models/psi0.py) |
| **System-1** | ~500M flow-matching DiT **action expert**. Trained. | `DiTActionTransformerModel` in [psi0.py:1211](src/psi/models/psi0.py#L1211) |
| **System-0** | External RL tracking / AMO whole-body controller. Out of this repo. | SIMPLE engine + real robot |

The deployed/fine-tune path is **flow matching with a DiT action expert**. The VLM is a *feature
extractor* here — it does **not** emit action tokens. (Autoregressive FAST action tokens are a
*separate* pre-training path, `trainers/pretrain.py`, not used at serve time.)

---

## 1. Training-time model

### 1.1 Dataflow

```
image(s) + instruction ──► VLM (Qwen3-VL-2B, frozen) ──► hidden_states[-1]  (B, seq, 2048)
                                                                │  (views = condition)
state (proprio) ───────────────────────────────────────────────┤
noisy action chunk (prefix=clean GT under RTC) ────────────────►│
per-position timestep ─────────────────────────────────────────►│
                                                                 ▼
                                  Action Expert (DiT, System-1, trained)
                                  obs_proj → q-former → DiT(noisy, t, cond)
                                                                 │
                                                                 ▼
                                       predicted flow velocity  v = noise − action
                                       (MSE loss vs target, suffix-only under RTC)
```

The model `forward` interface ([psi0.py:1605](src/psi/models/psi0.py#L1605)) has **no `prev_actions`
parameter**. The RTC prefix is communicated purely through (a) the *content* of the noisy action
chunk and (b) the *per-position timestep*.

### 1.2 Architecture & sizes (this fine-tune)

**Backbone (System-2)**

| Item | Value |
|---|---|
| Model | `Qwen/Qwen3-VL-2B-Instruct` |
| Hidden size (`view_feature_dim`) | **2048** |
| Trained? | **No** — `--model.no-tune-vlm` (frozen) |
| Init weights | `…/psi0/pre.fast.1by1.…ckpt.ego200k.he30k` |
| Image input (after transform) | resize + center-crop to **180 × 320** |
| Past frames | `num_past_frames=10` → **11 observed frames** per query |

**Action expert (System-1, DiT)** — [psi0.py:1222–1297](src/psi/models/psi0.py#L1222)

| Component | Value |
|---|---|
| `obs_proj` (`ObservationProjection`) | fuses VLM hidden (2048) + state (`odim=36`) → `action_hidden_dim` |
| `action_hidden_dim` | **1536** |
| q-former query tokens | **256** (cross-attn heads = 16) |
| DiT depth | **24** transformer blocks |
| DiT heads | **16**, `token_size=1536` |
| DiT input channels | `action_dim = 36` |
| Init weights | `…/psi0/postpre.1by1.pad36.…ckpt.he30k` |

### 1.3 Action / state / chunk dims

| Symbol | Meaning | Value (train) |
|---|---|---|
| `Da` (`action_dim`) | action dimension (padded) | **36** |
| `Tp` (`action_chunk_size`) | predicted chunk length (= action horizon `H`) | **30** |
| `Ta` (`action_exec_horizon`) | executed steps | **30** (train) |
| `odim` | state dim fed to action expert | **36** (client builds 32, padded → 36) |
| `observation_horizon` | obs frames into the expert | 1 |

**Tensor shapes during a training step** (batch `B`):

| Tensor | Shape |
|---|---|
| `pixel_values` (11 frames) | Qwen3-VL packed image tokens |
| `states` | `(B, 36)` |
| `action_samples` (noisy) | `(B, Tp=30, Da=36)` |
| `timestep` (RTC: per-position) | `(B, Tp=30)` |
| model output (flow velocity) | `(B, 30, 36)` |

### 1.4 Flow matching + RTC training

[`finetune.py::forward_and_loss`](src/psi/trainers/finetune.py)

- Noise scheduler = **flow**, `train_diffusion_steps=1000`.
- Velocity target: `v = noise − action`; loss = `MSE(pred, v)`.
- **RTC** (`--model.rtc`, `--model.max-delay=8`):
  1. sample `delay ~ U[0, max_delay)` per sample,
  2. `prefix_mask = arange(Tp) < delay`,
  3. on the prefix set `σ=0` ⇒ those positions hold **clean GT** (timestep 0),
  4. mask the loss to the **suffix only**.
- Net effect: the model learns to *inpaint* — fill the suffix given a clean prefix — which is
  exactly what inference does. This makes the inference-time prefix pinning **in-distribution**.

### 1.5 Training launch (recorded)

| Setting | Value |
|---|---|
| Trainer | `finetune` (DDP, bf16) |
| Global batch | **128** (`64 × 2 GPU × 1` grad-accum) |
| LR / schedule | `1e-4`, cosine, warmup 1000 |
| Max steps | 40000, ckpt every 2000, val every 500 |
| Data | `bedroom_v2_grasp_thermos_psi0/{train,val}` (LeRobot) |
| Norm | action `bounds` (maxmin), state normalized; pad action & state → 36 |

---

## 2. Inference-time system design

Two processes, one HTTP channel — **same protocol as real-robot deployment**.

```
┌──────────────────────────────┐         POST /act  (numpy-over-JSON, base64)        ┌──────────────────────────────┐
│  SIMPLE client                │ ──────────────────────────────────────────────────► │  Psi0 policy server          │
│  simple-3dgs/.venv            │   image(s) + instruction + state + history          │  .venv-psi                   │
│                              │                                                      │  serve_psi0 (FastAPI)        │
│  MuJoCo + gsplat sim          │ ◄────────────────────────────────────────────────── │  Psi0Model (System-2 + S-1)  │
│  Psi0Agent (baselines/psi0)   │            action chunk (Ta × Da)                    │  GET /config, /health        │
└──────────────────────────────┘                                                      └──────────────────────────────┘
```

### 2.1 Server — [`psi0_serve_simple.py`](src/psi/deploy/psi0_serve_simple.py)

Restores the `LaunchConfig` from `argv.txt` + `run_config.json`, loads `Psi0Model.from_pretrained`,
and pulls the maxmin / repack / model transforms.

**Serve-time horizon override** (eval script passes `--action-exec-horizon=24 --rtc`):

| Symbol | Value (serve) | Note |
|---|---|---|
| `Tp` | 30 | from checkpoint |
| `Ta` | **24** | override → returns 24 of 30 steps |
| `d = Tp − Ta` (`inference_delay`) | **6** | frozen-prefix overlap |
| `rtc_max_delay` | 8 | asserts `d ≤ max_delay` ⇒ `6 ≤ 8` ✓ |

**`predict_action`** ([psi0.py path](src/psi/deploy/psi0_serve_simple.py#L90)):

```python
# 1. normalize state (maxmin.normalize_state + pad→36); resize+center_crop images
# 2. RTC branch:
if self.previous_action is None or "reset" in history_dict:
    raw = model.predict_action(...)                       # unconditioned (episode start)
else:
    prev_actions = concat([ previous_action[None, Ta:, :],          # (1, Tp−Ta, Da) un-executed tail
                            zeros((1, Ta, Da)) ], axis=1)           # (1, Ta, Da) pad → (1, Tp, Da)
    raw = model.predict_action_with_training_rtc_flow(
              ..., prev_actions=prev_actions,
              inference_delay=Tp−Ta, max_delay=rtc_max_delay)       # hard-pin prefix, denoise suffix
# 3. raw → (Tp, Da);  pred = maxmin.denormalize(raw)
# 4. self.previous_action = raw.copy()                    # store NORMALIZED full chunk (server state)
# 5. return pred[:Ta]                                     # only first Ta=24 steps go back
```

Key facts:
- The RTC "action block" is **server-side state** (`self.previous_action`), *not* part of the
  request. The client only resets it via the `reset` flag in `history`.
- `predict_action_with_training_rtc_flow` ([psi0.py:1739](src/psi/models/psi0.py#L1739)) hard-pins
  prefix positions to `prev_actions` each denoise step with per-position `timestep=0`; the suffix is
  denoised from noise. Asserts `d < H` and `d < max_delay`.
- `GET /config` returns `num_past_frames` so the client auto-matches the training obs window.

### 2.2 Client agent — [`simple-3dgs/.../baselines/psi0.py`](../simple-3dgs/src/simple/baselines/psi0.py)

`Psi0Agent` (extends `PrimitiveAgent`) keeps a **frame buffer** (deque, len `num_past_frames+1`) and
an **action queue**. Per control step `get_action(obs)`:

1. buffer the current head-stereo-left frame;
2. on step 0, queue **60 "stand" warm-up** loco commands;
3. only **query the server when the action queue is empty**;
4. build the request:
   - `image_dict`: `rgb_head_stereo_left_t-N … t-0` (oldest→newest) or single frame,
   - `state` `(1, 32)` = hand/arm slices of `joint_qpos` + `_last_cmd_torso_rpyh` (see `STATE_SLICES`),
   - `history` = `{reset?, session_id, episode_index, step_index}`;
5. decode each returned action row (`Da=36`) into:
   - upper-body `target_qpos` (`from_psi0_upper_joints`, joints 15:),
   - `waist_qpos` (roll/pitch/yaw),
   - 8-element loco command `[vx, target_yaw, vy, d_height, torso_yaw, pitch, roll, turn_flag]`;
6. queue `ActionCmd("eval_move_actuators", …)` and `popleft` one per step.

### 2.3 Request / response wire format

`RequestMessage{image, instruction, history, state, condition, gt_action, dataset_name, timestamp}`
→ `ResponseMessage{action, err, traj_image}`, serialized via base64 numpy-over-JSON
([helpers.py](src/psi/deploy/helpers.py), [client.py](../simple-3dgs/src/simple/baselines/client.py)).

### 2.4 Orchestration — [`eval_closeloop_psi0_simple.sh`](scripts/deploy/eval_closeloop_psi0_simple.sh)

1. Start `serve_psi0 --port 22085 --policy=psi0 --run-dir=$RUN_DIR --ckpt-step=$CKPT_STEP
   --action-exec-horizon=24 --rtc` in the background; wait on a `/dev/tcp` port probe.
2. Export two **train/eval alignment** env vars (see §3.4):
   - `SIMPLE_TABLE_SPAWN_X=0.60`
   - `SIMPLE_INSTRUCTION_OVERRIDE="pick up the thermos from the nightstand."`
3. Run the SIMPLE `eval` console script (client) against the server.

---

## 3. Inference-time SIMPLE initialization

Two layers: **build the engine once**, then **rebuild the scene per episode** from the recorded
config. Two data sources:

| Source | Content | Where |
|---|---|---|
| **Static assets** (shared by all episodes) | robot MJCF, room/object meshes, gaussian `.ply`, `.mdl` materials | `data/` (auto-downloaded from HF `SIMPLE-org/SIMPLE`) + packaged `resources/` |
| **Per-episode config** | poses, lighting, instruction, which asset | LeRobot `meta/episodes.jsonl['environment_config']` |

### 3.1 Layer 0 — `gym.make(env_id)` (once per process)

[`base_dual_env.py`](../simple-3dgs/src/simple/envs/base_dual_env.py) builds the empty stage and the
agent. **Reused across all episodes** (`raw_env` is never rebuilt, only `reset`).

| Loads | From |
|---|---|
| Task object (DR sample-boxes, `sensor_cfgs`) | `TaskRegistry.make("g1_wholebody_bedroom_move_thermos")` |
| Robot (G1 wholebody) | `RobotRegistry.make(uid="g1_wholebody")` |
| MuJoCo engine | robot + table MJCF |
| Gaussian-splat renderer | `data/assets/objects/composed_bedroom_v2/scene/*.ply` |
| `Psi0Agent` | `importlib` on `policy="psi0"` |

### 3.2 Layer 1 — `get_episode_lerobot` (per episode)

[`datasets/lerobot.py:9`](../simple-3dgs/src/simple/datasets/lerobot.py#L9): pulls
`dataset.meta.episodes[idx]['environment_config']` (a JSON string), patches stale res-ids
(`composed_bedroom`→`_v2`, `102344280`→`scene3`), returns `env_conf`. Its `dr_state_dict` carries
**8 randomizers**: `language, target, distractors, spatial, camera, scene, lighting, material`.

### 3.3 Layer 2 — `env.reset(options={"state_dict": env_conf})`

`LocoManipulationEnv.reset` → `task.reset` → the **key fork**
([`core/task.py:136`](../simple-3dgs/src/simple/core/task.py#L136)):

```python
if options and options.get("state_dict"):
    self.dr.load_state_dict(env_conf)   # EVAL: replay recorded config
else:
    self.dr.reset(seed)                 # DATAGEN: fresh random sample
```

`load_state_dict` sets each randomizer's `_inner_state` to the recorded value. **Real episode-0
values:**

| randomizer | replayed value |
|---|---|
| `spatial` | robot `g1_wholebody` pos `[1.2, 0.6456, 0.772]`; thermos `"103"` pos `[0.184, 0.480, 0.876]` |
| `target` | `uid 103`, thermos, mesh `…/composed_bedroom_v2/models/103/coacd_0.05/coacd_merge.obj` |
| `scene` | `composed:scene_bedroom_v2`, `table_size=[0.74,1.63,0.7785]` |
| `lighting` | 2–3 `CylinderLight`, intensity `5e4`, color-temp ~6437 K |
| `distractors` | `{}` (only the thermos) |
| `material` | table `Zinc_Galvanizing.mdl`, ground `Ceramic_Tiles_Glazed.mdl` |
| `camera` | `null` → falls back to the task class `sensor_cfgs` |

Then `Task.reset` builds the `Layout` (add robot → load scene+table → load target mesh → apply
spatial poses → lights → materials), loading real assets from `data/assets/objects/composed_bedroom_v2/`.

### 3.4 Layer 3 — task overrides (eval-only) and physics push

[`g1_wholebody_bedroom_move_thermos.py:241`](../simple-3dgs/src/simple/tasks/custom_scene_tasks/g1_wholebody_bedroom_move_thermos.py#L241),
after `super().reset()`:

| Override | Env var | Why |
|---|---|---|
| instruction | `SIMPLE_INSTRUCTION_OVERRIDE` | task emits "move …", but the fine-tune trained on "pick up the thermos from the nightstand." — VLM is language-conditioned, must match |
| robot teleport `X 1.2 → 0.60` | `SIMPLE_TABLE_SPAWN_X` | `env_conf` restores the far spawn, but datagen dropped the walk-up; training frame 0 is *already at the table* — teleport to match the trained initial distribution |
| re-couple target Y to robot Y, re-face yaw | — | keep geometry consistent after teleport |

Finally back in `LocoManipulationEnv.reset`:

```python
self.mujoco.update_layout()    # write robot qpos + object xpos/xquat into MuJoCo mj_data
self.mujoco.step(render=False) # settle one frame
self.gsplat.reset()            # align renderer to new layout
obs = self._get_obs()          # render head-stereo-left + read joint_qpos → first obs
```

`obs` is handed to `Psi0Agent` and the closed loop begins.

---

## 4. One-line summary

`gym.make` builds the engine + robot + agent once (reused) → each episode pulls a recorded
`environment_config` from the LeRobot dataset → `dr.load_state_dict` replays 8 randomizers → the
`Layout` loads real meshes/`.ply`/materials from `data/` and places everything → the thermos task
teleports the robot and overrides the instruction to **match the training distribution** →
`mujoco.update_layout` produces the first frame → the agent queries the **stateful RTC policy
server** (Tp=30, Ta=24, d=6, max_delay=8), which normalizes, inpaints the suffix onto the frozen
prefix, denormalizes, and returns 24 action steps per query.

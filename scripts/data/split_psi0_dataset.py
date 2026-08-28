#!/usr/bin/env python3
"""Split a flat psi0 LeRobot dataset into self-contained train/val/test sub-datasets.

`postprocess_psi0.py` emits a single flat dataset (data/, videos/, meta/). The Psi0
trainer, however, loads each split as its OWN complete LeRobot dataset at
`<root>/<repo_id>` (e.g. `--data.root_dir=<DS> --data.train-repo-ids=train
--data.val-repo-ids=val`), and reads normalization stats from
`<DS>/train/meta/stats_psi0.json`. There is no "select a split inside one dataset"
path. This script materializes those sub-datasets.

For each split it:
  * re-indexes episodes from 0 (rewrites parquet `episode_index` and the global
    `index` column, renames parquet + video files to episode_{new:06d}),
  * writes split-specific meta/info.json (total_*, splits={name: "0:k"}),
    episodes.jsonl, episodes_stats.jsonl (both re-indexed) and copies tasks.jsonl,
  * copies the GLOBAL stats.json / stats_psi0.json / modality.json / lang_map.json /
    relative_stats.json verbatim into the split (matches the existing psi0 datasets:
    every split shares the dataset-wide normalization stats).

Default split is sequential (test = last episodes, val = the block before test, the
rest = train), mirroring the raw `info.json` split convention. Pass --shuffle for a
seeded random partition instead.
"""
import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

META_COPY_FILES = [
    "stats.json",
    "stats_psi0.json",
    "modality.json",
    "lang_map.json",
    "relative_stats.json",
]


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, separators=(",", ":"))
            f.write("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, help="Flat psi0 dataset (output of postprocess_psi0.py).")
    ap.add_argument("--out-dir", default=None, help="Where to write train/val/test subdirs. Default: --in-dir (in place).")
    ap.add_argument("--val-episodes", type=int, default=6, help="Number of validation episodes.")
    ap.add_argument("--test-episodes", type=int, default=12, help="Number of test episodes.")
    ap.add_argument("--shuffle", action="store_true", help="Random (seeded) partition instead of sequential.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunks-size", type=int, default=1000)
    ap.add_argument("--link", action="store_true", help="Hardlink videos instead of copying (saves space; same FS only).")
    ap.add_argument("--exclude-episodes", default="",
                    help="Comma-separated FLAT source episode indices to drop before splitting "
                         "(e.g. corrupt episodes whose video frame count != parquet rows). "
                         "Excluded episodes are removed entirely; remaining episodes keep their "
                         "sequential order so val/test still come from the tail.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else in_dir
    meta_in = in_dir / "meta"

    info = json.loads((meta_in / "info.json").read_text())
    chunks_size = int(info.get("chunks_size") or args.chunks_size)
    data_tmpl = info["data_path"]      # data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
    video_tmpl = info["video_path"]    # videos/chunk-{episode_chunk:03d}/egocentric/episode_{episode_index:06d}.mp4
    video_keys = [k for k, ft in info["features"].items() if ft.get("dtype") == "video"]

    episodes = sorted(load_jsonl(meta_in / "episodes.jsonl"), key=lambda r: r["episode_index"])
    ep_stats = {int(r["episode_index"]): r for r in load_jsonl(meta_in / "episodes_stats.jsonl")}
    exclude = {int(x) for x in args.exclude_episodes.split(",") if x.strip() != ""}
    if exclude:
        before = len(episodes)
        episodes = [e for e in episodes if int(e["episode_index"]) not in exclude]
        print(f"Excluding {len(exclude)} episode(s) {sorted(exclude)}: {before} -> {len(episodes)} episodes")
    n = len(episodes)
    src_indices = [int(e["episode_index"]) for e in episodes]

    n_val, n_test = args.val_episodes, args.test_episodes
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise SystemExit(f"Not enough episodes: n={n}, val={n_val}, test={n_test} -> train={n_train}")

    order = list(src_indices)
    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        order = [int(x) for x in rng.permutation(order)]
    # sequential convention: train | val | test  (test = last episodes)
    split_map = {
        "train": order[:n_train],
        "val": order[n_train:n_train + n_val],
        "test": order[n_train + n_val:],
    }
    print(f"Splitting {n} episodes -> train={len(split_map['train'])} "
          f"val={len(split_map['val'])} test={len(split_map['test'])}  (shuffle={args.shuffle})")

    for split, srcs in split_map.items():
        sp = out_dir / split
        if sp.exists():
            shutil.rmtree(sp)
        (sp / "meta").mkdir(parents=True, exist_ok=True)

        out_episodes, out_ep_stats = [], []
        running_index = 0  # global frame counter within this split (rebased to 0)

        for new_idx, src_idx in enumerate(srcs):
            chunk = new_idx // chunks_size
            src_parquet = in_dir / data_tmpl.format(episode_chunk=src_idx // chunks_size, episode_index=src_idx)
            table = pq.read_table(src_parquet)
            m = table.num_rows

            # rebase the two episode-global columns; leave per-frame columns untouched
            cols = {name: table.column(name) for name in table.column_names}
            cols["episode_index"] = np.full(m, new_idx, dtype=np.int64)
            cols["index"] = np.arange(running_index, running_index + m, dtype=np.int64)
            out_table = pa.table({name: (pa.array(val) if isinstance(val, np.ndarray) else val)
                                  for name, val in cols.items()})

            dst_parquet = sp / data_tmpl.format(episode_chunk=chunk, episode_index=new_idx)
            dst_parquet.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(out_table, dst_parquet)

            for vk in video_keys:
                src_v = in_dir / video_tmpl.format(episode_chunk=src_idx // chunks_size, episode_index=src_idx, video_key=vk)
                dst_v = sp / video_tmpl.format(episode_chunk=chunk, episode_index=new_idx, video_key=vk)
                dst_v.parent.mkdir(parents=True, exist_ok=True)
                if src_v.exists():
                    if args.link:
                        try:
                            __import__("os").link(src_v, dst_v)
                        except OSError:
                            shutil.copy2(src_v, dst_v)
                    else:
                        shutil.copy2(src_v, dst_v)

            ep = dict(episodes[src_indices.index(src_idx)])
            ep["episode_index"] = new_idx
            ep["length"] = m
            ep["dataset_from_index"] = running_index
            ep["dataset_to_index"] = running_index + m - 1
            out_episodes.append(ep)

            if src_idx in ep_stats:
                es = dict(ep_stats[src_idx])
                es["episode_index"] = new_idx
                out_ep_stats.append(es)

            running_index += m

        total_frames = running_index
        k = len(srcs)
        split_info = dict(info)
        split_info["total_episodes"] = k
        split_info["total_videos"] = k
        split_info["total_frames"] = int(total_frames)
        split_info["total_chunks"] = math.ceil(k / chunks_size) if chunks_size else 1
        split_info["splits"] = {split: f"0:{k}"}
        (sp / "meta" / "info.json").write_text(json.dumps(split_info, indent=4))
        write_jsonl(sp / "meta" / "episodes.jsonl", out_episodes)
        write_jsonl(sp / "meta" / "episodes_stats.jsonl", out_ep_stats)
        # tasks.jsonl + dataset-wide stats are shared verbatim across splits
        shutil.copy2(meta_in / "tasks.jsonl", sp / "meta" / "tasks.jsonl")
        for fn in META_COPY_FILES:
            src = meta_in / fn
            if src.exists():
                shutil.copy2(src, sp / "meta" / fn)
        print(f"  [{split}] {k} eps, {total_frames} frames -> {sp}")

    print("Done.")


if __name__ == "__main__":
    main()

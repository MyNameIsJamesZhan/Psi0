from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from psi.config.data_lerobot import LerobotDataConfig
    # from psi.config.data_simple import SimpleDataConfig

import torch
from psi.data.lerobot.compat import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from psi.utils import resolve_path
from psi.config.transform import LerobotRepackTransform


def _find_corrupt_episodes(meta: LeRobotDatasetMetadata, root: str) -> list[int]:
    """Return episode indices whose video files are missing or zero-byte."""
    corrupt: list[int] = []
    root_path = Path(root)
    video_keys = list(meta.video_keys) if len(meta.video_keys) > 0 else []
    if not video_keys:
        return corrupt
    for ep_idx in range(meta.total_episodes):
        for vid_key in video_keys:
            vpath = root_path / meta.get_video_file_path(ep_idx, vid_key)
            try:
                if not vpath.exists() or vpath.stat().st_size == 0:
                    corrupt.append(ep_idx)
                    break
            except OSError:
                corrupt.append(ep_idx)
                break
    return corrupt


class LeRobotDatasetWrapper(torch.utils.data.Dataset):
    """ A wrapper around LeRobotDataset to support multiple datasets.
    """

    def __init__(
        self,
        data_cfg: LerobotDataConfig,
        split: str = "train"
    ):
        repo_ids = data_cfg.train_repo_ids if split == "train" else data_cfg.val_repo_ids
        first_repo = repo_ids[0] if isinstance(repo_ids, list) else repo_ids
        dataset_meta = LeRobotDatasetMetadata(first_repo, resolve_path(f"{data_cfg.root_dir}/{first_repo}"))
        assert isinstance(data_cfg.transform.repack, LerobotRepackTransform)
        delta_timestamps = data_cfg.transform.repack.delta_timestamps(dataset_meta.fps)

        if len(repo_ids) > 1:
            root_dir = data_cfg.root_dir
            lerobot_dataset_class = MultiLeRobotDataset
            episodes_arg = None
        else:
            repo_ids = first_repo
            root_dir = resolve_path(f"{data_cfg.root_dir}/{first_repo}")
            lerobot_dataset_class = LeRobotDataset
            corrupt = _find_corrupt_episodes(dataset_meta, root_dir)
            if corrupt:
                print(
                    f"[LeRobotDatasetWrapper] split={split} repo={first_repo}: "
                    f"excluding {len(corrupt)} episode(s) with missing/empty videos: {corrupt}"
                )
                episodes_arg = [e for e in range(dataset_meta.total_episodes) if e not in set(corrupt)]
            else:
                episodes_arg = None

        kwargs: dict[str, Any] = dict(
            root=root_dir,
            delta_timestamps=delta_timestamps, # type: ignore
            image_transforms=None,
        )
        if episodes_arg is not None:
            kwargs["episodes"] = episodes_arg

        self.base_dataset = lerobot_dataset_class(
            repo_ids,# type: ignore
            **kwargs,
        )

        # Upstream lerobot builds `episode_data_index` as a compact positional tensor
        # (length = num kept episodes), but `_get_query_indices` looks up by the
        # original `ep_idx` from the parquet. When episodes are filtered, these
        # disagree and __getitem__ raises IndexError for any ep_idx past the
        # filtered length. Remap to be indexed by original ep_idx.
        if episodes_arg is not None and lerobot_dataset_class is LeRobotDataset:
            compact = self.base_dataset.episode_data_index  # type: ignore
            max_ep = max(episodes_arg) + 1
            from_remap = torch.zeros(max_ep, dtype=torch.long)
            to_remap = torch.zeros(max_ep, dtype=torch.long)
            for i, ep in enumerate(episodes_arg):
                from_remap[ep] = compact["from"][i]
                to_remap[ep] = compact["to"][i]
            self.base_dataset.episode_data_index = {"from": from_remap, "to": to_remap}  # type: ignore

        self._cache = {}

    def __getitem__(self, idx) -> dict:
        return self.base_dataset[idx]
    
    def __len__(self):
        return len(self.base_dataset)

    @property
    def episode_data_index(self):
        return self.base_dataset.episode_data_index # type: ignore

    @property
    def num_episodes(self):
        return self.base_dataset.num_episodes
    
    @property
    def num_frames(self):
        return self.base_dataset.num_frames
    
    @property
    def meta(self):
        return self.base_dataset.meta # type: ignore

    @property
    def stats(self):
        return self.base_dataset.stats if type(self.base_dataset) == MultiLeRobotDataset else self.base_dataset.meta.stats # type: ignore

"""Validate the saved held-out map used by the shared Ant evaluation task."""

import hashlib
import json
import math
from pathlib import Path


def load_evaluation_metadata(usd_path):
    """Validate the fixed map file and its 100 ordered flat start conditions."""
    usd_path = Path(usd_path)
    metadata = json.loads(usd_path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata["usd_file"] != usd_path.name:
        raise ValueError("Manifest USD filename does not match the requested map.")
    if hashlib.sha256(usd_path.read_bytes()).hexdigest() != metadata["usd_sha256"]:
        raise ValueError("Evaluation USD SHA256 does not match its manifest.")
    evaluation = metadata["evaluation"]
    if evaluation["purpose"] != "evaluation_only" or evaluation["seed"] != 24 or evaluation["num_envs"] != 100:
        raise ValueError("Expected evaluation-only map with seed 24 and 100 saved starts.")
    starts = evaluation["starts"]
    if len(starts) != 100:
        raise ValueError("Evaluation needs exactly 100 saved starts.")
    origins = metadata["terrain_origins"]
    size = metadata["generator"]["tile_size"]
    for index, start in enumerate(starts):
        ground, root = start["ground_position"], start["root_position"]
        row, col = start["tile"]
        if start["env_id"] != index or len(ground) != 3 or len(root) != 3:
            raise ValueError("Start indices and xyz coordinates must be complete and ordered.")
        if not all(math.isfinite(value) for value in ground + root):
            raise ValueError("Evaluation start coordinates must be finite.")
        if metadata["tile_types"][row][col] != "flat" or abs(ground[2]) > 1e-6:
            raise ValueError("Every evaluation start must use a flat tile at world z=0.")
        if any(abs(root[axis] - ground[axis] - (0.5 if axis == 2 else 0)) > 1e-6 for axis in range(3)):
            raise ValueError("Ant root must be 0.5 m above its saved ground start.")
        if any(abs(ground[axis] - origins[row][col][axis]) > size / 2 - 2.0 for axis in (0, 1)):
            raise ValueError("Evaluation start must remain at least 2 m inside its flat tile.")
    return metadata

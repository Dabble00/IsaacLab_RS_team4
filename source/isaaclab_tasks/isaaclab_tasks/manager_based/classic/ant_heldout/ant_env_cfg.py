"""Saved held-out map for the original or a team's Ant task."""

import json
import os
from pathlib import Path

import torch
from pxr import Usd

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
from isaaclab_tasks.manager_based.classic.humanoid import mdp
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from ant_maps.evaluation_metadata import load_evaluation_metadata


# Reward definitions from Isaac Lab 2.3.0 (BSD-3-Clause, copyright The Isaac Lab Project Developers).
@configclass
class OriginalV0RewardsCfg:
    """Frozen Isaac Lab 2.3.0 Isaac-Ant-v0 reward settings for fair evaluation."""

    progress = RewTerm(func=mdp.progress_reward, weight=1.0, params={"target_pos": (1000.0, 0.0, 0.0)})
    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.1, params={"threshold": 0.93})
    move_to_target = RewTerm(
        func=mdp.move_to_target_bonus,
        weight=0.5,
        params={"threshold": 0.8, "target_pos": (1000.0, 0.0, 0.0)},
    )
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.005)
    energy = RewTerm(func=mdp.power_consumption, weight=-0.05, params={"gear_ratio": {".*": 15.0}})
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits_penalty_ratio,
        weight=-0.1,
        params={"threshold": 0.99, "gear_ratio": {".*": 15.0}},
    )


class EvaluationTerrainImporter(TerrainImporter):
    def __init__(self, cfg):
        self.map_metadata = load_evaluation_metadata(cfg.usd_path)
        if cfg.num_envs != 100:
            raise ValueError("The held-out map uses 100 fixed starts; pass --num_envs 100.")
        saved = Usd.Stage.Open(cfg.usd_path)
        expected = {key: value for key, value in self.map_metadata.items() if key not in ("usd_file", "usd_sha256")}
        if not saved or json.loads(saved.GetDefaultPrim().GetCustomDataByKey("map_metadata")) != expected:
            raise ValueError("USD embedded settings and JSON manifest disagree.")
        super().__init__(cfg)
        print(f"[EVAL MAP] Loaded {Path(cfg.usd_path).name}; SHA256={self.map_metadata['usd_sha256']}", flush=True)
        print("[EVAL MAP] 100 saved flat starts; start seed=24.", flush=True)

    def configure_env_origins(self, origins=None):
        # Native reset adds Ant's default root height (0.5 m) to these ground coordinates.
        self.terrain_origins = torch.tensor(self.map_metadata["terrain_origins"], device=self.device, dtype=torch.float32)
        self.env_origins = torch.tensor(
            [start["ground_position"] for start in self.map_metadata["evaluation"]["starts"]],
            device=self.device, dtype=torch.float32,
        )


@configclass
class EvaluationTerrainImporterCfg(TerrainImporterCfg):
    class_type: type = EvaluationTerrainImporter


def evaluation_terrain_cfg():
    return EvaluationTerrainImporterCfg(
        prim_path="/World/ground", terrain_type="usd", collision_group=-1,
        usd_path=str(Path(__file__).resolve().parents[6] / "ant_maps/generated/heldout_seed20261008_friction_spectrum.usd"),
        debug_vis=False,
    )


@configclass
class AntHeldoutEnvCfg(AntEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 100
        self.scene.terrain = evaluation_terrain_cfg()
        self.rewards = OriginalV0RewardsCfg()


def _training_task():
    task = os.environ.get("ANT_EVAL_BASE_TASK", "").strip()
    if not task or task.split(":")[-1] in ("Isaac-Ant-Heldout-v0", "Isaac-Ant-SharedEval-v0"):
        raise ValueError("Set ANT_EVAL_BASE_TASK to the registered Ant task used to train this checkpoint.")
    return task


def shared_env_cfg():
    """Keep the team's Ant configuration, replacing only terrain and reward."""
    task = _training_task()
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    if not isinstance(cfg, ManagerBasedRLEnvCfg) or getattr(cfg.scene, "robot", None) is None:
        raise TypeError(f"{task} must be a manager-based Ant task with scene.robot.")
    cfg.scene.num_envs = 100
    cfg.scene.terrain = evaluation_terrain_cfg()
    cfg.rewards = OriginalV0RewardsCfg()
    print(f"[EVAL MAP] Base task={task}; reward=original Isaac-Ant-v0; observations/actions/terminations=base task.")
    return cfg


def shared_agent_cfg():
    return load_cfg_from_registry(_training_task(), "rsl_rl_cfg_entry_point")

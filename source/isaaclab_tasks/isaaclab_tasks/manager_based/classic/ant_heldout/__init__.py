"""Register the fixed and team-model Ant tests without changing Isaac-Ant-v0."""

import sys
from pathlib import Path

import gymnasium as gym

# The shared map helpers live alongside source/ in the editable Isaac Lab repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

gym.register(
    id="Isaac-Ant-Heldout-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_env_cfg:AntHeldoutEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_ppo_cfg:AntPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-SharedEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_env_cfg:shared_env_cfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ant_env_cfg:shared_agent_cfg",
    },
)

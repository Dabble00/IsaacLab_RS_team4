"""ant_robust: 구르기 v15 의 학습 태스크와 평가 태스크 등록. 원본 Isaac Lab 파일은 수정하지 않는다."""
import gymnasium as gym

_ANT_PPO_CFG = "isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_ppo_cfg:AntPPORunnerCfg"  # PPO 설정 = 원본 Ant 것 그대로
_FULL_CFG = "isaaclab_tasks.manager_based.classic.ant_robust.full_env_cfg:"
_EVAL_CFG = "isaaclab_tasks.manager_based.classic.ant_robust.eval_env_cfg:"

# ---- 수업 원본 설정(관측 60)으로 재는 평가 지형: T1 = Isaac Lab 기본 ROUGH 지형, T2 = 개미 크기 울퉁불퉁 메시 ----
for _task_id, _cfg_class in {
    "Isaac-Ant-Eval-StockRough-v0": "AntStockRoughEvalEnvCfg",
    "Isaac-Ant-Eval-AntRough-v0": "AntRoughEvalEnvCfg",
}.items():
    gym.register(id=_task_id, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                 kwargs={"env_cfg_entry_point": _EVAL_CFG + _cfg_class, "rsl_rl_cfg_entry_point": _ANT_PPO_CFG})

# ---- 학습·제출·평가 태스크 (full_env_cfg.py) ----
#   학습: Fun-Roll11 → Fun-Roll13 → Fun-Roll15 (제출 모델). 제출·평가: Isaac-Ant-Final-J200C-v0 (평지), Eval-StockRough (T1), Eval-AntRough (T2), TeamMap (팀원 맵)
#   Final 가족 = 관측 240·관절 원본 모델용, Base 가족 = 원본 모델용
_TASKS = {
    "Isaac-Ant-Fun-Roll11-v0": "AntFunRoll11EnvCfg",
    "Isaac-Ant-Fun-Roll13-v0": "AntFunRoll13EnvCfg",
    "Isaac-Ant-Fun-Roll14-v0": "AntFunRoll14EnvCfg",
    "Isaac-Ant-Fun-Roll15-v0": "AntFunRoll15EnvCfg",
    "Isaac-Ant-Final-J200C-v0": "AntFinalJ200CEnvCfg",
    "Isaac-Ant-Eval-AntRough-Final-J200C-v0": "AntFinalJ200CRoughEvalEnvCfg",
    "Isaac-Ant-Eval-StockRough-Final-J200C-v0": "AntFinalJ200CStockRoughEvalEnvCfg",
    "Isaac-Ant-TeamMap-Final-J200C-v0": "AntTeamMapJ200CEnvCfg",
    "Isaac-Ant-TeamMap2-Final-J200C-v0": "AntTeamMap2J200CEnvCfg",
    "Isaac-Ant-Final-v0": "AntFinalEnvCfg",
    "Isaac-Ant-Eval-AntRough-Final-v0": "AntFinalRoughEvalEnvCfg",
    "Isaac-Ant-Eval-StockRough-Final-v0": "AntFinalStockRoughEvalEnvCfg",
    "Isaac-Ant-TeamMap-Final-v0": "AntTeamMapFinalEnvCfg",
    "Isaac-Ant-TeamMap2-Final-v0": "AntTeamMap2FinalEnvCfg",
    "Isaac-Ant-TeamMap2-Base-v0": "AntTeamMap2BaseEnvCfg",
    "Isaac-Ant-TeamMapN-Base-v0": "AntTeamMapN_Base",  # 팀 내부 비교용: 0.31 m 탈락만 끔 (짧은 맵)
    "Isaac-Ant-TeamMap2N-Base-v0": "AntTeamMap2N_Base",  # 팀 내부 비교용: 0.31 m 탈락만 끔 (긴 맵)
    "Isaac-Ant-TeamMapN-Final-v0": "AntTeamMapN_Final",  # 팀 내부 비교용: 0.31 m 탈락만 끔 (짧은 맵)
    "Isaac-Ant-TeamMap2N-Final-v0": "AntTeamMap2N_Final",  # 팀 내부 비교용: 0.31 m 탈락만 끔 (긴 맵)
    "Isaac-Ant-TeamMapN-J200C-v0": "AntTeamMapN_J200C",  # 팀 내부 비교용: 0.31 m 탈락만 끔 (짧은 맵)
    "Isaac-Ant-TeamMap2N-J200C-v0": "AntTeamMap2N_J200C",  # 팀 내부 비교용: 0.31 m 탈락만 끔 (긴 맵)
}
for _task_id, _cfg_class in _TASKS.items():
    gym.register(id=_task_id, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                 kwargs={"env_cfg_entry_point": _FULL_CFG + _cfg_class, "rsl_rl_cfg_entry_point": _ANT_PPO_CFG})

# ---- 단일 지형 시험장 35종: Isaac-Ant-Battery-<종류>[-Final|-J200C]-v0 (설명 = eval_env_cfg.py 의 BATTERY 표) ----
#   이름 목록은 BATTERY 표와 같아야 한다. 설정 클래스는 full_env_cfg.py 가 루프로 만든다 (AntBattery<가족>_<종류>)
BATTERY_NAMES = (
    "plane", "stairs_pit", "blocks_grid", "moguls", "washboard", "craters", "gravel", "cobble_street",
    "spike_field", "posts", "tilted_rubble", "fallen_logs", "riverbed_boulders", "angular_scree",
    "railway_sleepers", "undercut_hurdles", "star_beams", "sawtooth", "honeycomb_walls", "mud_cracks",
    "longitudinal_ruts", "wide_ditches", "potholes", "drainage_grating", "stepping_stones", "moat_gaps",
    "pit", "box", "rails",  # (19단계) 기본 지형 중 학습에 안 넣은 3종
    "ice", "stick_slip", "sticky_rubber", "bouncy", "mud", "sponge",
)
for _name in BATTERY_NAMES:
    for _family, _suffix in (("Base", ""), ("Final", "-Final"), ("J200C", "-J200C")):
        gym.register(
            id=f"Isaac-Ant-Battery-{_name}{_suffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": f"{_FULL_CFG}AntBattery{_family}_{_name}", "rsl_rl_cfg_entry_point": _ANT_PPO_CFG},
        )

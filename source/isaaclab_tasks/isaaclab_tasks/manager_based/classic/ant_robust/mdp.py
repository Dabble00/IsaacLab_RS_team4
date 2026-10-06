from __future__ import annotations

import numpy as np

import os

import torch

from typing import TYPE_CHECKING

from isaaclab.assets import Articulation

from isaaclab.envs.mdp import randomize_rigid_body_material

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from isaaclab.managers import ActionTerm, ActionTermCfg

from isaaclab.sensors import ContactSensor, RayCaster

from isaaclab.terrains import TerrainImporter, TerrainImporterCfg

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import CurriculumTermCfg


def height_scan_safe(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5,
    clip_range: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """몸통 밑 바닥까지의 "상대 높이" 지도. 항상 유한하고 [clip_range] 안에 들어온다.

    값의 뜻:
      0 근처  = 평소 평지와 같은 높이
      + 쪽    = 바닥이 평소보다 낮다 (내리막 / 구덩이)
      - 쪽    = 바닥이 평소보다 높다 (턱 / 오르막 / 계단)
      정확히 +1 = 바닥이 아예 없다(허공) 이거나 아주 깊다
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    low, high = clip_range
    # (환경수, 81) 각 광선이 맞은 지점의 월드 z. 못 맞힌 광선은 inf
    z_hit = sensor.data.ray_hits_w[..., 2]
    # (환경수, 1) 센서(= 몸통 torso)의 월드 z
    z_sensor = sensor.data.pos_w[:, 2].unsqueeze(1)
    height = z_sensor - z_hit - offset
    # (1) 빗나간 광선 -> "바닥 없음" -> 가장 깊은 값(high). 여기가 핵심 안전장치
    height = torch.where(torch.isfinite(z_hit), height, torch.full_like(height, high))
    # (2) 혹시 남은 inf / NaN (물리가 터져 센서 자세 자체가 깨진 경우) 제거
    height = torch.nan_to_num(height, nan=0.0, posinf=high, neginf=low)
    # (3) 마지막으로 범위 자르기 -> 신경망은 절대 [-1, 1] 밖 값을 못 본다
    return height.clamp_(low, high)


def torso_below_ground_relative_height(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.31,
    radius: float = 0.25,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """몸통 높이를 "몸통 바로 아래 바닥" 기준으로 쟀을 때 minimum_height 보다 낮으면 True."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    torso = asset.data.root_pos_w  # (환경수, 3)
    hits = sensor.data.ray_hits_w  # (환경수, 광선수, 3)  못 맞힌 광선은 inf
    # 몸통에서 수평 거리 radius 안에 떨어진, 제대로 맞은 광선만 사용
    dxy = (hits[..., :2] - torso[:, None, :2]).norm(dim=-1)
    near = torch.isfinite(hits[..., 2]) & (dxy < radius)
    count = near.sum(dim=1)
    ground_z = torch.where(near, hits[..., 2], torch.zeros_like(hits[..., 2])).sum(dim=1) / count.clamp(min=1)
    # 바닥을 하나도 못 찾으면(허공) 이 판정은 쉬고, 원본 월드 z 판정에 맡긴다
    return (count > 0) & (torso[:, 2] - ground_z < minimum_height)


class PrioritizedTerrainReplay(ManagerTermBase):
    """[C15] PLR 방식 지형 출제자. CurriculumTermCfg(func=PrioritizedTerrainReplay, params=...) 로 붙인다."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        terrain = env.scene.terrain
        num_rows, num_cols = terrain.terrain_origins.shape[:2]
        self.num_cols = num_cols
        self.num_start_rows = min(cfg.params.get("max_start_row", 7), num_rows - 1) + 1
        n = self.num_start_rows * num_cols
        dev = env.device
        self.score = torch.zeros(n, device=dev)  # S_i: 후회 근사 (클수록 "배울 게 많은" 칸)
        self.best = torch.full((n,), -float("inf"), device=dev)  # 이 칸에서 나온 최고 에피소드 보상
        self.latest = torch.zeros(n, device=dev)  # 이 칸의 최근 에피소드 보상
        self.scored = torch.zeros(n, dtype=torch.bool, device=dev)
        self.visits = torch.zeros(n, device=dev)
        self.last_pick = torch.zeros(n, device=dev)  # C_i
        self.total_picks = 0.0  # c
        self.first_done = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)  # 각 env 의 첫 에피소드가 끝났나
        # 열 번호 -> 지형 종류
        gen_cfg = env.cfg.scene.terrain.terrain_generator
        bank = getattr(gen_cfg, "bank_resolved", None)
        if bank:  # C17 지형 은행: 줄마다 적힌 종류 (procedural / flat)
            lane_types = [lane["type"] for lane in bank["lanes"]]
            self.type_names = sorted(set(lane_types))
            col_type = [self.type_names.index(t) for t in lane_types]
        else:  # 일반 코스: terrain_generator.py 240 줄과 같은 규칙
            self.type_names = list(gen_cfg.sub_terrains.keys())
            props = np.array([sub.proportion for sub in gen_cfg.sub_terrains.values()], dtype=float)
            cum = np.cumsum(props / props.sum())
            col_type = [int(np.min(np.where(c / num_cols + 0.001 < cum)[0])) for c in range(num_cols)]
        self.cell_type = torch.tensor(col_type, device=dev).repeat(self.num_start_rows)  # (칸,)  행 우선 순서
        self._last_saved = -1
        self._prob = torch.full((n,), 1.0 / n, device=dev)

    def _replay_distribution(self, beta: float, rho: float) -> torch.Tensor:
        """채점된 칸들에 대한 P = (1 - rho) P_S + rho P_C.  채점 안 된 칸은 0."""
        n = self.score.numel()
        s = torch.where(self.scored, self.score, torch.full_like(self.score, -float("inf")))
        order = torch.argsort(s, descending=True)
        rank = torch.empty(n, device=s.device)
        rank[order] = torch.arange(1, n + 1, device=s.device, dtype=torch.float)
        p_s = torch.where(self.scored, (1.0 / rank) ** (1.0 / beta), torch.zeros_like(rank))
        p_s = p_s / p_s.sum().clamp(min=1e-12)
        stale = torch.where(self.scored, self.total_picks - self.last_pick, torch.zeros_like(self.last_pick))
        p_c = stale / stale.sum() if stale.sum() > 0 else self.scored.float() / self.scored.float().sum().clamp(min=1)
        return (1.0 - rho) * p_s + rho * p_c

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        max_start_row: int = 7,
        beta: float = 1.0,
        rho: float = 0.3,
        save_every_steps: int = 3200,
        score_mode: str = "regret",  # "regret" (C15/C17, 원래) 또는 "learnability" (C33)
        uniform_mix: float = 0.0,  # (C34) 이 비율만큼은 점수와 상관없이 모든 칸에서 고르게 뽑음 -> 잘하는 칸을 잊지 않게
    ) -> dict[str, float]:
        terrain = env.scene.terrain
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        n_cells = self.score.numel()

        # ---- (1) 방금 끝난 에피소드 채점 ----
        ran = env.episode_length_buf[env_ids] > 0  # 맨 처음 reset (아직 안 달림) 제외
        first = ran & ~self.first_done[env_ids]  # 첫 에피소드(길이가 무작위로 잘림)는 채점 안 함
        self.first_done[env_ids[first]] = True
        ids = env_ids[ran & ~first]
        rows, cols = terrain.terrain_levels[ids], terrain.terrain_types[ids]
        keep = rows < self.num_start_rows
        ids, rows, cols = ids[keep], rows[keep], cols[keep]
        if len(ids) > 0:
            # 끝난 에피소드의 보상 합계 (보상 매니저는 이 함수 "다음"에 초기화되므로 아직 남아 있다)
            ret = sum(v[ids] for v in env.reward_manager._episode_sums.values())
            cell = rows * self.num_cols + cols
            total = torch.zeros(n_cells, device=env.device).index_add_(0, cell, ret)
            count = torch.zeros(n_cells, device=env.device).index_add_(0, cell, torch.ones_like(ret))
            hit = count > 0
            self.latest[hit] = total[hit] / count[hit]  # 최근 점수로 덮어쓰기
            self.best[hit] = torch.maximum(self.best[hit], self.latest[hit])
            self.scored[hit] = True
            self.visits[hit] += count[hit]
            if score_mode == "learnability":
                # 기준 = 채점된 평지 칸들의 최근 점수 평균 (평지 줄이 없거나 아직 미채점이면 전체 최고)
                flat = (self.cell_type == self.type_names.index("flat")) & self.scored if "flat" in self.type_names else torch.zeros_like(self.scored)
                ref = self.latest[flat].mean() if flat.any() else self.latest[self.scored].max()
                x = (self.latest[self.scored] / ref.clamp(min=1e-6)).clamp(0.0, 1.0)
                self.score[self.scored] = 4.0 * x * (1.0 - x)  # 반쯤 되는 칸이 1, 다 되거나 전혀 안 되는 칸은 0
            else:
                self.score[hit] = self.best[hit] - self.latest[hit]

        # ---- (2) 다음 출발 칸 뽑기 ----
        n = len(env_ids)
        prob = self._replay_distribution(beta, rho) if self.scored.any() else torch.zeros(n_cells, device=env.device)
        if uniform_mix > 0 and self.scored.any():  # (C34) 잊어버림 방지: 일부는 모든 칸에서 고르게
            prob = (1.0 - uniform_mix) * prob + uniform_mix / n_cells
        unscored = (~self.scored).nonzero().squeeze(-1)
        p_new = len(unscored) / n_cells
        pick = torch.multinomial(prob, n, replacement=True) if self.scored.any() else torch.zeros(n, dtype=torch.long, device=env.device)
        if len(unscored) > 0:
            use_new = torch.rand(n, device=env.device) < p_new
            pick = torch.where(use_new, unscored[torch.randint(0, len(unscored), (n,), device=env.device)], pick)
        self.total_picks += n
        self.last_pick[pick] = self.total_picks
        new_rows, new_cols = pick // self.num_cols, pick % self.num_cols
        terrain.terrain_levels[env_ids] = new_rows
        terrain.terrain_types[env_ids] = new_cols
        terrain.env_origins[env_ids] = terrain.terrain_origins[new_rows, new_cols]
        self._prob = prob

        # ---- (3) 히트맵용 저장 ----
        k = env.common_step_counter // save_every_steps
        if k > self._last_saved and env.cfg.log_dir is not None:
            self._last_saved = k
            out = os.path.join(env.cfg.log_dir, "teacher")
            os.makedirs(out, exist_ok=True)
            shape = (self.num_start_rows, self.num_cols)
            np.savez(
                os.path.join(out, f"step_{env.common_step_counter:08d}.npz"),
                score=self.score.view(shape).cpu().numpy(),
                best=self.best.view(shape).cpu().numpy(),
                latest=self.latest.view(shape).cpu().numpy(),
                visits=self.visits.view(shape).cpu().numpy(),
                prob=prob.view(shape).cpu().numpy(),
                col_type=self.cell_type[: self.num_cols].cpu().numpy(),
                type_names=np.array(self.type_names),
            )

        # ---- (4) 텐서보드 기록 ----
        log = {
            "regret_mean": self.score[self.scored].mean().item() if self.scored.any() else 0.0,
            "unscored_cells": float(len(unscored)),
            "effective_cells": (1.0 / (prob**2).sum()).item() if self.scored.any() else 0.0,  # 1/sum p^2
        }
        for t, name in enumerate(self.type_names):
            mask = self.cell_type == t
            log[f"prob_{name}"] = prob[mask].sum().item()  # 이 지형 종류에 출제될 확률
            log[f"regret_{name}"] = self.score[mask & self.scored].mean().item() if (mask & self.scored).any() else 0.0
        return log


def print_collider_offsets(env: ManagerBasedEnv, env_ids, asset_cfg: SceneEntityCfg, tag: str = ""):
    """env 0 의 충돌 모양별 contact / rest offset 과 전체 범위를 출력한다."""
    asset: Articulation = env.scene[asset_cfg.name]
    co = asset.root_physx_view.get_contact_offsets()
    ro = asset.root_physx_view.get_rest_offsets()
    print(
        f"[COLLIDER {tag}] contact_offset env0={[round(float(x), 4) for x in co[0]]} "
        f"all=[{float(co.min()):.4f}, {float(co.max()):.4f}] | rest_offset env0={[round(float(x), 4) for x in ro[0]]}",
        flush=True,
    )


# =====================================================================
#  - Isaac Lab 기본 randomize_rigid_body_material 은 충돌 모양(몸통·다리·발 13개)마다 따로 뽑는다
#    -> 네 발 마찰이 제각각이라 "바닥 전체가 미끄러운" 상황(빙판)을 거의 못 겪는다
#       (검토 에이전트 모의 계산: 4096 환경 중 네 발 모두 마찰 0.3 이하인 환경 약 3개)
#  - 여기서는 환경마다 재질 하나를 뽑아 몸 전체 13개 모양에 똑같이 준다 (legged_gym 과 같은 방식)
#  - nominal_fraction 만큼의 재질은 원래 값(마찰 1.0 / 1.0, 반발 0)으로 둔다 -> 평소 바닥 걸음도 계속 배움
#  - 설정은 기본 함수와 같고 nominal_fraction 하나만 더 받는다. asset_cfg 는 SceneEntityCfg("robot") (몸 전체)
# =====================================================================
class randomize_robot_material_per_env(randomize_rigid_body_material):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)  # 재질 후보(material_buckets)를 범위 안에서 뽑아 둠
        n_nominal = round(cfg.params.get("nominal_fraction", 0.0) * len(self.material_buckets))
        self.material_buckets[:n_nominal] = torch.tensor([1.0, 1.0, 0.0])

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
        nominal_fraction: float = 0.0,
    ):
        env_ids = torch.arange(env.scene.num_envs) if env_ids is None else env_ids.cpu()
        num_shapes = self.asset.root_physx_view.max_shapes
        picked = self.material_buckets[torch.randint(0, num_buckets, (len(env_ids),))]  # 환경마다 재질 하나
        materials = self.asset.root_physx_view.get_material_properties()
        materials[env_ids] = picked[:, None, :].expand(-1, num_shapes, -1)  # 13개 모양에 똑같이
        self.asset.root_physx_view.set_material_properties(materials, env_ids)
        m = materials[env_ids]
        print(
            f"[MATERIAL per-env] envs={len(env_ids)} static=[{float(m[..., 0].min()):.3f}, {float(m[..., 0].max()):.3f}] "
            f"dynamic=[{float(m[..., 1].min()):.3f}, {float(m[..., 1].max()):.3f}] restitution=[{float(m[..., 2].min()):.3f}, "
            f"{float(m[..., 2].max()):.3f}] nominal={float((m[:, 0, 0] == 1.0).float().mean()):.2f} "
            f"same_within_env={bool((m == m[:, :1]).all())}",
            flush=True,
        )


# =====================================================================
#   - widen_joint_limits: 시작할 때 관절 한계를 넓힘 (원본 개미는 엉덩이 ±30°·발목 30~70° 라 굴러갈 수 없음)
#   - torso_height_bonus: 몸통이 threshold 보다 높이 있으면 그만큼 보상 (높이뛰기)
#   - torso_forward_spin: 앞으로 구르는 회전 속도 |ω_y| (풍차돌리기). 기존 roll_forward_bonus 와 같은 축, 뒤로 구르기도 인정 옵션
# =====================================================================
def widen_joint_limits(env: ManagerBasedEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg, lower: float, upper: float):
    """[재미] 모든 관절의 위치 한계를 (lower, upper) [rad] 로 바꾼다 (startup 이벤트)."""
    asset: Articulation = env.scene[asset_cfg.name]
    old = asset.data.joint_pos_limits[0].clone()
    limits = torch.tensor([lower, upper], device=asset.device).expand(asset.num_instances, asset.num_joints, 2).clone()
    asset.write_joint_position_limit_to_sim(limits, warn_limit_violation=False)
    print(
        f"[FUN joint limits] {asset.joint_names}\n  전: {[tuple(round(float(v), 2) for v in r) for r in old]}"
        f"\n  후: ({lower}, {upper}) 전부 | 토크 한계 {[round(float(v), 1) for v in asset.data.joint_effort_limits[0]]}",
        flush=True,
    )


# =====================================================================
#   - forward_speed_bonus: 앞(+x) 속도 [m/s] (최대 max_speed). 제출용 보상 탐색(C40)에서 켜 봄
#   - roll_forward_gated: 앞 속도 x 앞으로 구르는 회전(정규화). 둘 다 있어야 보상 -> 제자리 회전은 0
#   - bound_forward: 몸통이 바닥(높이 센서 기준)에서 height 이상 떠 있을 때의 앞 속도 -> 앞으로 뛰어야 보상
# =====================================================================
def _torso_height_above_ground(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, radius: float = 0.25):
    """(몸통 높이 - 몸통 아래 바닥 높이, 바닥을 찾았나) — torso_below_ground_relative_height 와 같은 계산."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    torso = asset.data.root_pos_w
    hits = sensor.data.ray_hits_w
    dxy = (hits[..., :2] - torso[:, None, :2]).norm(dim=-1)
    near = torch.isfinite(hits[..., 2]) & (dxy < radius)
    count = near.sum(dim=1)
    ground_z = torch.where(near, hits[..., 2], torch.zeros_like(hits[..., 2])).sum(dim=1) / count.clamp(min=1)
    return torso[:, 2] - ground_z, count > 0


def roll_forward_gated(env: ManagerBasedRLEnv, max_rate: float = 15.0, max_speed: float = 6.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    v = asset.data.root_lin_vel_w[:, 0].clamp(0.0, max_speed)
    w = asset.data.root_ang_vel_w[:, 1].clamp(0.0, max_rate) / max_rate
    return v * w


# =====================================================================
#   v2 구르기 측정: 구르는 중인 시간 55~67 %, 마리당 구르기 구간 약 20번, 가장 긴 연속 구간 2~3초 -> "튕기며 뒤집기" 에 가까움
#   원인: v2 보상(앞 속도 x 회전)은 순간 값만 보므로 세게 한 번 뒤집고 착지하는 쪽도 점수를 받음
#   해결: (1) 연속으로 구른 걸음 수에 비례하는 보상 (멈추면 0 으로 리셋), (2) 일정한 회전 속도를 유지하면 보상, (3) 좌우·수직 축 흔들림 벌점
# =====================================================================
class roll_streak(ManagerTermBase):
    """연속 구르기 보상: 앞으로 구르는 회전(세계 y축 각속도)이 min_rate 이상인 걸음이 몇 걸음째 이어지는지 세어
    hold_steps(기본 120 = 2초) 에서 1.0 이 되도록 선형으로 올린다. 멈추거나 거꾸로 돌면 0 으로 리셋."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        self.count = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.count[:] = 0.0
        else:
            self.count[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, min_rate: float = 3.0, hold_steps: int = 120, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        rolling = self.asset.data.root_ang_vel_w[:, 1] > min_rate
        self.count = torch.where(rolling, self.count + 1.0, torch.zeros_like(self.count))
        return (self.count / hold_steps).clamp(0.0, 1.0)


def roll_rate_target(env: ManagerBasedRLEnv, target: float = 8.0, sigma: float = 3.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """일정한 회전 속도 보상: 앞으로 구르는 각속도가 target(8 rad/s ≈ 1.3바퀴/초) 에 가까울수록 1, sigma 만큼 벗어나면 0.37."""
    asset: Articulation = env.scene[asset_cfg.name]
    w = asset.data.root_ang_vel_w[:, 1]
    return torch.exp(-((w - target) / sigma) ** 2)


def roll_wobble_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """흔들림 벌점용: 구르는 축(y)이 아닌 x·z 축 각속도의 제곱합. 깨끗하게 한 축으로만 구르면 0."""
    asset: Articulation = env.scene[asset_cfg.name]
    w = asset.data.root_ang_vel_w
    return w[:, 0] ** 2 + w[:, 2] ** 2


# =====================================================================
#   = 몸통 평면을 수직(바퀴 자세)으로 세우고 네 다리를 바퀴살처럼 써서 구르기. 몸통 중심은 다리 길이만큼 높이 떠 있어 0.31 m 규칙을 지킴
#   wheel_posture: 바퀴 자세 보상. 몸통 좌표계에서 본 중력의 z 성분이 0 이면(몸통이 옆으로 90° 누움) 1, 똑바로 서 있으면 0
# =====================================================================
def wheel_posture(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return 1.0 - asset.data.projected_gravity_b[:, 2].abs()


# =====================================================================
#   - roll_on_ground: v2 구르기 보상(앞 속도 x 회전)에 "몸 어딘가가 땅에 닿아 있을 때만" 조건을 붙임 (접촉 센서). 공중제비는 0점
#   - height_above:   몸통이 바닥에서 height 보다 높이 떠 있는 만큼 벌점용 값 (튕겨 오르기 억제)
#   - vertical_speed_l2 / lateral_speed_l1: 위아래 속도 제곱, 옆(y) 속도 크기. 벌점용 (튕김·옆으로 새기 억제)
# =====================================================================
def roll_on_ground(env: ManagerBasedRLEnv, max_rate: float = 15.0, max_speed: float = 6.0, threshold: float = 1.0, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"), asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]  # (마리, 몸 부위) 최근 접촉힘
    on_ground = (force > threshold).any(dim=1).float()
    return roll_forward_gated(env, max_rate, max_speed, asset_cfg) * on_ground


def height_above(env: ManagerBasedRLEnv, height: float = 0.6, radius: float = 0.25, sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"), asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    h, ok = _torso_height_above_ground(env, asset_cfg, sensor_cfg, radius)
    return (h - height).clamp(min=0.0) * ok.float()


def vertical_speed_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w[:, 2] ** 2


def lateral_speed_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w[:, 1].abs()


# =====================================================================
#   원인 1: 앤트 자산은 자기 충돌(self collision)이 꺼져 있어 다리가 몸통·다른 다리를 통과함 -> 환경 설정에서 켠다 (full_env_cfg 27단계)
#   원인 2: 보상이 "굴러서 앞으로"만 봐서 다리를 접고 낮게 구르는 쪽이 유리했음 -> 바퀴 자세·곧은 다리·높은 몸통에 보상
#   legs_straight: 모든 관절이 0(다리를 곧게 뻗은 X 자) 에 가까울수록 1.  torso_above: 몸통이 바닥에서 height 보다 높이 있으면 1 (다리 끝으로 구르기)
# =====================================================================
def legs_straight(env: ManagerBasedRLEnv, sigma: float = 0.5, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos
    return torch.exp(-(q / sigma) ** 2).mean(dim=1)


def torso_above(env: ManagerBasedRLEnv, height: float = 0.45, radius: float = 0.25, sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"), asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    h, ok = _torso_height_above_ground(env, asset_cfg, sensor_cfg, radius)
    return ((h > height) & ok).float()


# =====================================================================
#   foot_slip_speed: 땅에 닿아 있는 발의 수평 속도 평균 (닿은 발이 없으면 0). 헛도는 만큼 커진다 → 벌점으로 쓰면 "미끄러우면 살살" 을 배운다
# =====================================================================
def foot_slip_speed(env: ManagerBasedRLEnv, threshold: float = 1.0, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"), asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    force = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]  # (마리, 발)
    contact = (force > threshold).float()
    foot_ids, _ = asset.find_bodies(".*_foot")
    speed = asset.data.body_lin_vel_w[:, foot_ids, :2].norm(dim=-1)  # (마리, 발) 수평 속도
    return (speed * contact).sum(dim=1) / contact.sum(dim=1).clamp(min=1.0)


# =====================================================================
#   팀원 맵(ant_maps_evaluation)은 +x 로 짧아서 출발점에서 지형 타일 끝(x = 80 m)까지 90~155 m 뿐이다. 빠른 모델은 16초 안에 끝을 지나 허공으로 떨어진다.
#   reached_course_end: 몸통 x 가 x_end 를 넘으면 그 환경을 끝낸다(완주). 처음 넘는 순간 "[COURSE END] env=i step=s x=.." 한 줄을 찍는다 (환경마다 한 번).
#   채점 로그의 이 줄을 세면 완주한 마리 수·완주 시간이 나온다 (tools/make_teammap_table.py). 점수는 완주 순간까지의 원본 보상 합.
# =====================================================================
class TeamMapTerrainImporter(TerrainImporter):
    """긴 맵용: 팀원 맵 USD 를 불러오고, 출발점 100개를 옆 JSON 의 저장값으로 고정한다. 팀원의 EvaluationTerrainImporter 와 같은 동작을 우리 코드로.
    JSON: terrain_origins (행×열×3), evaluation.starts[i].ground_position. USD 의 SHA256 이 JSON 과 다르면 거부 (파일이 바뀌었는지 확인)."""

    def __init__(self, cfg):
        import hashlib
        import json

        self.map_metadata = json.load(open(cfg.usd_path[:-4] + ".json", encoding="utf-8"))
        if hashlib.sha256(open(cfg.usd_path, "rb").read()).hexdigest() != self.map_metadata["usd_sha256"]:
            raise ValueError(f"{cfg.usd_path} 의 SHA256 이 JSON 과 다름")
        if cfg.num_envs != len(self.map_metadata["evaluation"]["starts"]):
            raise ValueError(f"이 맵은 출발점 {len(self.map_metadata['evaluation']['starts'])}개 고정: --num_envs 를 그 값으로")
        super().__init__(cfg)
        print(f"[TEAM MAP] {os.path.basename(cfg.usd_path)}: 타일 {self.map_metadata['generator']['rows']}행×{self.map_metadata['generator']['cols']}열, 출발점 {cfg.num_envs}개 (JSON 저장값)", flush=True)

    def configure_env_origins(self, origins=None):
        self.terrain_origins = torch.tensor(self.map_metadata["terrain_origins"], device=self.device, dtype=torch.float32)
        self.env_origins = torch.tensor([s["ground_position"] for s in self.map_metadata["evaluation"]["starts"]], device=self.device, dtype=torch.float32)


@configclass
class TeamMapTerrainImporterCfg(TerrainImporterCfg):
    class_type: type = TeamMapTerrainImporter


class reached_course_end(ManagerTermBase):
    """몸통 x ≥ x_end 이면 끝(완주). 환경마다 처음 한 번 [COURSE END] 줄을 찍는다."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.reported = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)  # 리셋해도 안 지움: 첫 에피소드만 센다 (play_one_episode 와 같은 기준)

    def __call__(self, env: ManagerBasedRLEnv, x_end: float = 80.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        x = asset.data.root_pos_w[:, 0]
        done = x >= x_end
        new = done & ~self.reported
        if new.any():
            steps = env.episode_length_buf
            for i in torch.nonzero(new).flatten().tolist():
                print(f"[COURSE END] env={i} step={int(steps[i])} x={float(x[i]):.1f}", flush=True)
            self.reported |= new
        return done

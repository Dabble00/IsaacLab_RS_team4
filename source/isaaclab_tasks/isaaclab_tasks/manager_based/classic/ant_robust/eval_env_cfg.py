# =====================================================================
# 3단계: "처음 보는 지형" 평가 전용 환경 설정 (학습에는 아직 안 씀)
#
#  - 원본 AntEnvCfg 를 그대로 물려받고(상속) 바닥(scene.terrain)만 바꾼다.
#  - 관측(60차원), 행동, 보상, 종료조건(몸통 z < 0.31 m), 마찰(average 1.0/1.0/0.0),
#    에피소드 길이(16초 = 960 step)는 원본과 완전히 같다.
#  - 원본 파일은 import(가져다 쓰기)만 하고 절대 고치지 않는다.
#  - [주의] self.scene.terrain.xxx 의 xxx 철자를 틀려도 에러가 나지 않는다(그냥 무시됨).
#           실행 후 outputs/.../.hydra/config.yaml 에서 값이 바뀌었는지 꼭 확인한다.
#  - v2 (3단계-b): T2 에서 개미가 옆 평지로 빠져나가지 않도록 "가운데 8열에서만 출발"
# =====================================================================

import math  # (15단계 추가) 단일 지형 시험장
from typing import NamedTuple  # (15단계 추가)

import numpy as np  # (15단계 추가)
import trimesh  # (15단계 추가) 말뚝·통나무·가로대 모양

import isaaclab.sim as sim_utils
from isaaclab.terrains import (  # HfDiscreteObstacles, HfRandomUniform, TerrainGenerator(Cfg) 외에는 15단계 추가
    HfDiscreteObstaclesTerrainCfg,
    HfInvertedPyramidSlopedTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfSteppingStonesTerrainCfg,
    MeshBoxTerrainCfg,  # (19단계) 시험장 상자 단
    MeshGapTerrainCfg,
    MeshPitTerrainCfg,  # (19단계) 시험장 구덩이
    MeshRailsTerrainCfg,  # (19단계) 시험장 레일
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPlaneTerrainCfg,
    MeshRandomGridTerrainCfg,
    MeshRepeatedBoxesTerrainCfg,
    MeshRepeatedCylindersTerrainCfg,
    MeshRepeatedPyramidsTerrainCfg,
    MeshStarTerrainCfg,
    SubTerrainBaseCfg,
    TerrainGenerator,
    TerrainGeneratorCfg,
)
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.terrains.height_field import HfTerrainBaseCfg  # (15단계 추가)
from isaaclab.terrains.height_field.utils import height_field_to_mesh  # (15단계 추가)
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg

# ---------------------------------------------------------------------
# T2 에서 쓸 "개미 크기" 울퉁불퉁 지형
#  - 8 m(x) x 5 m(y) 타일을 x 방향 22칸, y 방향 16칸 (176 m x 80 m), 바깥에 평평한 테두리 50 m
#  - 개미 출발점은 가운데 8열(y = -17.5 ~ +17.5 m)에만 -> 옆 평지까지 최소 22.5 m
#  - 바닥 높이는 전부 0 m 이상 (최대 약 0.12 m) -> 구덩이 없음
# ---------------------------------------------------------------------


class InnerLaneTerrainGenerator(TerrainGenerator):
    """땅은 16열 전부 만들고, 개미 출발점은 가운데 8열에만 둔다."""

    SPAWN_MARGIN_COLS = 4  # 양쪽 바깥에서 출발점으로 안 쓸 열 수 (4열 x 5 m = 20 m)

    def __init__(self, cfg, device: str = "cpu"):
        k = self.SPAWN_MARGIN_COLS
        assert cfg.num_cols > 2 * k, f"num_cols 는 {2 * k} 보다 커야 함 (지금 {cfg.num_cols})"
        super().__init__(cfg, device)  # 원래 방식대로 지형(16열)을 전부 만든다
        # 출발에 쓸 가운데 열 번호 [4, 5, ..., 11] 을 거꾸로 [11, 10, ..., 4] 로
        #  -> env 0 이 y = +17.5 m 에서 출발 (카메라가 -y 쪽을 보므로 울퉁불퉁한 땅이 넓게 보임)
        cols = list(range(k, cfg.num_cols - k))[::-1]
        # TerrainImporter 는 terrain_origins (행, 열, 3) 의 "열 개수"만큼 개미를 나눠 배정한다
        #  -> 출발점 목록에서 바깥 열을 빼면, 땅은 그대로 있고 개미만 가운데 8열에서 출발
        self.terrain_origins = self.terrain_origins[:, cols]
        # (지금은 비어 있음) 나중에 flat_patch_sampling 을 쓸 때도 같은 열을 가리키도록
        for name in self.flat_patches:
            self.flat_patches[name] = self.flat_patches[name][:, cols]


ANT_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    class_type=InnerLaneTerrainGenerator,  # 위의 "가운데 8열에서만 출발" 생성기 사용
    size=(8.0, 5.0),  # 타일 한 칸 크기 (x, y) [m]
    border_width=50.0,  # 전체 지형 바깥의 평평한 테두리 폭 [m]
    num_rows=22,  # x 방향(개미가 달리는 방향) 타일 수: 22 x 8 = 176 m
    num_cols=16,  # y 방향(옆 방향) 타일 수: 16 x 5 = 80 m (전체 폭은 예전과 같음)
    horizontal_scale=0.1,  # 높이맵 격자 간격 [m]
    vertical_scale=0.005,  # 높이 최소 단위 [m]
    slope_threshold=0.75,  # 기본 ROUGH 와 같은 값
    curriculum=False,  # 쉬운 것 -> 어려운 것 순서 배치 안 함 (타일 종류를 무작위로 섞음)
    difficulty_range=(1.0, 1.0),  # 모든 타일의 난이도를 1.0 으로 고정
    use_cache=False,  # 지형을 파일로 저장해 두지 않음
    sub_terrains={
        # (1) 자갈길: 10 cm 간격마다 높이 0 / 2 / 4 / 6 cm 중 하나를 무작위로
        "gravel": HfRandomUniformTerrainCfg(
            proportion=0.4, noise_range=(0.0, 0.06), noise_step=0.02, border_width=0.25
        ),
        # (2) 살짝 솟은 물결 바닥: 50 cm 간격으로 4 / 6 / 8 cm 중 하나를 뽑아 부드럽게 이음
        #     -> 타일 안쪽은 약 0.5~11 cm 높이로 솟아 있고, 잔물결은 2~3 cm 정도
        #     -> 타일 가장자리 0.3 m 는 높이 0 이라서, 앞으로 8 m 마다 2~11 cm 짜리 턱이 생김
        "wavy": HfRandomUniformTerrainCfg(
            proportion=0.3, noise_range=(0.04, 0.08), noise_step=0.02, downsampled_scale=0.5, border_width=0.25
        ),
        # (3) 낮은 블록: 높이 8 cm, 한 변 0.4 m 또는 0.8 m 블록 25개 (위로만 솟음, 구덩이 없음)
        #     (타일이 64 -> 40 m² 로 작아져서 40 -> 25개. 블록 밀도는 예전과 같음)
        #     (1.2 는 "미만"이라 0.4 / 0.8 두 가지만 나옴. 가운데 1.5 m 는 평평하게 비워 둠)
        "blocks": HfDiscreteObstaclesTerrainCfg(
            proportion=0.3,
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.4, 1.2),
            obstacle_height_range=(0.08, 0.08),
            num_obstacles=25,
            platform_width=1.5,
            border_width=0.25,
        ),
    },
)


# ---------------------------------------------------------------------
# T1: Isaac Lab 기본 ROUGH 지형을 "수정 없이" 그대로 사용 (이전과 동일)
# ---------------------------------------------------------------------
@configclass
class AntStockRoughEvalEnvCfg(AntEnvCfg):
    def __post_init__(self):
        # 1) 원본 Ant 설정을 먼저 전부 적용
        super().__post_init__()
        # 2) 바닥을 평면(plane) -> 지형 생성기(generator) 로 교체
        #    (configclass 가 마지막에 복사본을 만들므로 원본 ROUGH_TERRAINS_CFG 자체는 안 바뀜)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        # 3) 바닥 색 (기본값은 검정이라 GUI 에서 지형이 안 보임). 물리에는 영향 없음
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.55, 0.45))
        # (출발 행 제한 max_init_terrain_level 은 기본값 None 그대로 -> 10개 행 중 무작위 행에서 출발)


# ---------------------------------------------------------------------
# T2: 위의 ANT_ROUGH_TERRAINS_CFG 사용
# ---------------------------------------------------------------------
@configclass
class AntRoughEvalEnvCfg(AntEnvCfg):
    def __post_init__(self):
        # 1) 원본 Ant 설정을 먼저 전부 적용
        super().__post_init__()
        # 2) 바닥을 개미 크기 지형으로 교체
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ANT_ROUGH_TERRAINS_CFG
        # 3) 모든 개미를 0번 행(맨 뒤, x = -84 m), 가운데 8열(y = -17.5 ~ +17.5 m)에서 출발시킴
        #    -> 앞으로 울퉁불퉁한 땅 172 m, 옆 평지까지 최소 22.5 m (개미는 평지에서 16초에 약 141 m 달림)
        self.scene.terrain.max_init_terrain_level = 0
        # 4) 바닥 색 (물리에는 영향 없음)
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.60, 0.50, 0.40))

# #####################################################################
#
#    -> 시험장 하나에는 지형 "한 종류"만 둔다. 종류마다 따로 점수와 영상을 남긴다.
#  - 틀은 T2 와 같다: 가운데 열에서만 출발(옆 평지까지 20 m 이상), 0번 행에서 +x 로 약 176 m.
#  - 타일마다 난이도(0~1)를 무작위로 뽑는다 -> 같은 종류 안에서 크기·배치만 달라진다 (종류는 절대 안 섞음).
#  - 평균 바닥 높이를 약 0 m 로 맞춘다.
#      높은 곳에서 뒤집힌 Ant 가 탈락하지 않는 착시, 넓게 꺼진 곳에서 걷던 Ant 가 탈락하는 불공정을 피하려고.
#  - 종류마다 새로움을 정직하게 표시한다 (BATTERY 표의 두 번째 칸).
#  - 태스크 (full_env_cfg.py 15단계에서 만들고 __init__.py 에서 등록):
#      Isaac-Ant-Battery-<종류>-v0        기준 모델용 (원본 관측 60개, 원본 물리)
#      Isaac-Ant-Battery-<종류>-Final-v0  최종 모델용 (센서 + 접촉 여유 0.04 = Isaac-Ant-Final-v0 과 같은 설정)
#  - 보상 7개 / 종료조건(몸통 월드 z < 0.31 m) / 에피소드 16초는 전부 원본 그대로.
# #####################################################################
NEW = "학습에 없던 종류"
SCALE_NEW = "크기·배치만 새로움"
REFERENCE = "학습에 있던 종류(기준용)"

BATTERY_LENGTH = 176.0  # [m] 달리는 방향(x) 길이 (T2 와 같음)
BATTERY_WIDTH = 80.0  # [m] 옆 방향(y) 폭 (T2 와 같음)
BATTERY_SIDE_MARGIN = 20.0  # [m] 출발 열에서 옆 평지까지 최소 거리


# ---------------------------------------------------------------------
# 15-1) 공통 틀: T2 의 "가운데 열에서만 출발" + 타일 크기에 맞춘 제외 열 수 + (필요하면) 출발점 옮기기
# ---------------------------------------------------------------------
class BatteryTerrainGenerator(InnerLaneTerrainGenerator):
    def __init__(self, cfg, device: str = "cpu"):
        self.SPAWN_MARGIN_COLS = cfg.spawn_margin_cols  # 타일 폭이 달라도 옆 평지까지 20 m 이상 되게
        super().__init__(cfg, device)
        if cfg.spawn_shift_x:  # 예: 계단 구덩이는 가운데(바닥)가 아니라 -x 쪽 가장자리(높이 0)에서 출발
            self.terrain_origins[..., 0] -= cfg.spawn_shift_x
            self.terrain_origins[..., 2] = 0.0


@configclass
class BatteryTerrainGeneratorCfg(TerrainGeneratorCfg):
    class_type: type = BatteryTerrainGenerator
    spawn_margin_cols: int = 4  # 출발에 안 쓰는 바깥 열 수 (battery_course 가 계산)
    spawn_shift_x: float = 0.0  # [m] 출발점을 -x 로 옮기는 거리 (0 이면 타일 가운데)


def battery_course(sub_terrains, tile=(8.0, 5.0), slope_threshold=0.75, spawn_shift_x=0.0):
    """지형 한 종류로만 된 시험장. sub_terrains = 설정 하나, 또는 같은 종류의 크기 단계 여러 개 {이름: 설정}."""
    if not isinstance(sub_terrains, dict):
        sub_terrains = {"main": sub_terrains}
    return BatteryTerrainGeneratorCfg(
        size=tile,
        border_width=50.0,
        num_rows=round(BATTERY_LENGTH / tile[0]),
        num_cols=round(BATTERY_WIDTH / tile[1]),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=slope_threshold,  # 이웃 칸 높이 차가 0.75 x 0.1 m 를 넘으면 수직 벽으로 (None = 안 함)
        curriculum=False,
        difficulty_range=(0.0, 1.0),  # 타일마다 난이도를 무작위로 -> 같은 종류 안에서 크기·배치만 다름
        use_cache=False,
        sub_terrains=sub_terrains,
        spawn_margin_cols=math.ceil(BATTERY_SIDE_MARGIN / tile[1]),
        spawn_shift_x=spawn_shift_x,
    )


# ---------------------------------------------------------------------
# 15-2) 직접 만든 높이맵 지형: 종류마다 함수 하나  f(rng, nx, ny, hs, difficulty) -> 높이 [m], 모양 (nx, ny)
#   - 0번 축 = +x (달리는 방향), 1번 축 = y. hs = 격자 간격 0.1 m. 무작위 수는 전부 rng 에서.
# ---------------------------------------------------------------------
def _grid(nx, ny, hs):
    """격자점 좌표 X, Y [m] (0번 축 = +x)"""
    return np.meshgrid(np.arange(nx) * hs, np.arange(ny) * hs, indexing="ij")


def _darts(rng, lx, ly, n_max, min_dist, margin=0.0, max_fail=200):
    """점 뿌리기: 서로 min_dist 이상 떨어진 점을 최대 n_max 개 (연속 max_fail 번 실패하면 그만)"""
    pts, fails = np.zeros((0, 2)), 0
    while len(pts) < n_max and fails < max_fail:
        p = rng.uniform((margin, margin), (lx - margin, ly - margin))
        if len(pts) and np.hypot(*(pts - p).T).min() < min_dist:
            fails += 1
        else:
            pts, fails = np.vstack([pts, p]), 0
    return pts


def _rows_along_x(rng, lx, gap_lo, gap_hi, first_lo, end_margin):
    """길을 가로지르는 줄들의 x 위치: 첫 줄 U(first_lo, gap_hi), 다음 줄부터 U(gap_lo, gap_hi) 간격"""
    xs, x = [], rng.uniform(first_lo, gap_hi)
    while x < lx - end_margin:
        xs.append(x)
        x += rng.uniform(gap_lo, gap_hi)
    return xs


def _across(X, Y, x0, y0, yaw):
    """(x0, y0) 를 지나 길을 가로지르는 선(y 방향)을 yaw 만큼 돌린 선까지의 부호 있는 거리 [m]"""
    return (X - x0) * np.cos(yaw) - (Y - y0) * np.sin(yaw)


def _voronoi(X, Y, C):
    """격자점마다 가장 가까운 점까지 거리 F1, 두 번째로 가까운 점까지 거리 F2, 가장 가까운 점 번호"""
    D = np.hypot(X[..., None] - C[:, 0], Y[..., None] - C[:, 1])
    F = np.partition(D, 1, axis=-1)
    return F[..., 0], F[..., 1], D.argmin(axis=-1)


def _hf_unfold(H, hs, vs=0.005, slope=0.75):
    """수직 벽(slope_threshold) 지형용 손질: 삼각형이 뒤집힐 점만 조금 올린다 (높이 [m] -> 높이 [m])
    Isaac Lab 은 이웃보다 slope*hs (= 7.5 cm) 넘게 낮은 점을 그 이웃 쪽으로 한 칸 옮겨 수직 벽을 만든다.
    벽 안쪽 모서리의 점은 x 로 한 칸, y 로 반대 방향 한 칸 (+x-y 또는 -x+y) 옮겨져 삼각형이 뒤집힌다.
    -> 그런 점은 가장 높은 이웃보다 딱 7.5 cm 낮게 올려서 안 옮겨지게 한다 (그 모서리만 작은 경사가 됨)"""
    Z = np.pad(np.rint(H / vs), 1)  # Isaac Lab 과 같은 정수 높이 + 높이 0 테두리 1칸
    t = slope * (hs / vs)  # 벽이 되는 높이 차 (정수 단위, 15. Isaac Lab 과 같은 식)
    for _ in range(3):  # 보통 한 번 손질하면 끝남
        mx, my, mc = np.zeros_like(Z), np.zeros_like(Z), np.zeros_like(Z)
        mx[:-1] += Z[1:] - Z[:-1] > t
        mx[1:] -= Z[:-1] - Z[1:] > t
        my[:, :-1] += Z[:, 1:] - Z[:, :-1] > t
        my[:, 1:] -= Z[:, :-1] - Z[:, 1:] > t
        mc[:-1, :-1] += Z[1:, 1:] - Z[:-1, :-1] > t
        mc[1:, 1:] -= Z[:-1, :-1] - Z[1:, 1:] > t
        bad = (mx + mc * (mx == 0)) * (my + mc * (my == 0)) < 0  # x 와 y 로 서로 반대 방향 이동
        bad[[0, -1], :] = False  # 테두리(높이 0)는 그대로
        bad[:, [0, -1]] = False
        P = np.pad(Z, 1, mode="edge")  # 주변 9점 (자기 포함) 중 최댓값
        top = np.max([np.roll(P, (a, b), (0, 1))[1:-1, 1:-1] for a in (-1, 0, 1) for b in (-1, 0, 1)], axis=0)
        Z[bad] = np.maximum(Z[bad], top[bad] - t)  # 가장 높은 이웃보다 7.5 cm 낮게
    return Z[1:-1, 1:-1] * vs


def _no_walls(H, step=0.07):
    """이웃 8칸(테두리 0 포함)보다 step 넘게 낮은 칸을 올린다 -> 턱은 모두 7 cm 이하 (수직 벽 보정 안 일어남)."""
    for _ in range(4):
        P = np.pad(H, 1)
        nb = np.max([np.roll(P, (a, b), (0, 1))[1:-1, 1:-1] for a in (-1, 0, 1) for b in (-1, 0, 1)], axis=0)
        H = np.maximum(H, nb - step)
    return H

def hf_washboard(rng, nx, ny, hs, difficulty):
    """빨래판 잔물결: 길을 가로지르는 파장 0.4~0.6 m 물결, 높이 최대 ±3.5~5.5 cm (옆으로 가며 70~100 %)"""
    X, Y = _grid(nx, ny, hs)
    a = 0.035 + 0.02 * difficulty  # 물결 높이
    lam = rng.uniform(0.4, 0.6)  # 파장 (10 cm 격자로 4~6점)
    u = _across(X, Y, 0.0, ny * hs / 2, rng.uniform(-0.3, 0.3))  # 물결 마루는 y 축에서 ±17도
    mod = 0.85 + 0.15 * np.sin(2 * np.pi * Y / rng.uniform(1.5, 3.0) + rng.uniform(0, 2 * np.pi))  # 옆으로 가며 세기 70~100 %
    return a * mod * np.sin(2 * np.pi * u / lam + rng.uniform(0, 2 * np.pi))


def hf_craters(rng, nx, ny, hs, difficulty):
    """분화구: 지름 0.8~1.6 m 둥근 사발(깊이 6~12 cm) + 둘레 둔덕(+1.8~5.5 cm)"""
    X, Y = _grid(nx, ny, hs)
    bowl, rim = np.zeros((nx, ny)), np.zeros((nx, ny))
    d_max = 0.08 + 0.04 * difficulty  # 이 타일의 가장 깊은 사발
    for cx, cy in _darts(rng, nx * hs, ny * hs, int(rng.uniform(0.3, 0.5) * nx * ny * hs * hs), 1.5, margin=0.5):
        R, d = rng.uniform(0.4, 0.8), d_max * rng.uniform(0.75, 1.0)  # 사발 반지름, 깊이
        rh = min(rng.uniform(0.3, 0.6) * d, 0.055)  # 둔덕 높이
        rho = np.hypot(X - cx, Y - cy)
        z = np.where(rho < R, -d + (d + rh) * (rho / R) ** 2, rh * np.clip(1 - (rho - R) / (0.4 * R), 0, 1) ** 2)
        bowl, rim = np.minimum(bowl, z), np.maximum(rim, z)  # 사발끼리는 깊은 쪽, 둔덕끼리는 높은 쪽
    return bowl + rim  # 겹친 곳도 끊김 없이 이어짐 (다른 분화구 둔덕이 사발 가장자리를 살짝 메움)


def hf_longitudinal_ruts(rng, nx, ny, hs, difficulty):
    """바퀴 자국: +x 로 길게 난 U자 홈 두 줄씩 (폭 0.2~0.34 m, 깊이 6~12 cm) + 옆 흙둔덕 (+1.8~3.6 cm)"""
    X, Y = _grid(nx, ny, hs)
    ruts, berms = np.zeros((nx, ny)), np.zeros((nx, ny))
    d_max = 0.08 + 0.04 * difficulty  # 이 타일의 가장 깊은 홈
    T, Py = rng.uniform(0.6, 1.0), rng.uniform(1.4, 2.0)  # 두 바퀴 사이 폭, 바퀴 자국 한 쌍의 옆 간격
    for y in np.arange(rng.uniform(-Py, 0), ny * hs + Py, Py):
        for yc in (y - T / 2, y + T / 2):
            w, d = rng.uniform(0.10, 0.17), d_max * rng.uniform(0.75, 1.0)  # 홈 반폭, 깊이
            wiggle = rng.uniform(0.05, 0.25) * np.sin(2 * np.pi * X / rng.uniform(3, 8) + rng.uniform(0, 2 * np.pi))
            dy = np.abs(Y - yc - wiggle)  # 살짝 구불구불한 홈 중심선까지 거리
            ruts = np.minimum(ruts, -d * np.clip(1 - (dy / w) ** 2, 0, None))
            berms = np.maximum(berms, np.where(dy < 2 * w, 0.3 * d * np.clip(np.sin(np.pi * (dy - w) / w), 0, None), 0))
    fade = np.clip(np.minimum(X, (nx - 1) * hs - X) / 0.4, 0, 1)  # 타일 앞뒤 끝 0.4 m 에서 얕아져 높이 0 테두리와 이어짐
    return fade * np.where(ruts < 0, ruts, berms)


def hf_sawtooth(rng, nx, ny, hs, difficulty):
    """톱니: 길이 0.6~1.2 m 완만한 경사 끝에 10~12 cm 수직 턱이 반복 (높이 -6.6 ~ +5.4 cm)"""
    X, Y = _grid(nx, ny, hs)
    P, h = rng.uniform(0.6, 1.2), 0.10 + 0.02 * difficulty  # 톱니 길이, 턱 높이 (한 칸 사이 낙차가 늘 7.5 cm 초과 -> 수직)
    f = np.mod(_across(X, Y, 0.0, ny * hs / 2, rng.uniform(-0.2, 0.2)) + rng.uniform(0, P), P) / P  # 톱니 안 위치 0 -> 1
    if rng.uniform() < 0.5:
        f = 1 - f  # 타일 절반: 올라가는 턱을 만나는 방향
    return _hf_unfold(h * (f - 0.55), hs)  # 꼭대기 0.45h (+5.4 cm 이하)


def hf_potholes(rng, nx, ny, hs, difficulty):
    """구멍밭: 평평한 바닥(0 m)에 지름 0.3~0.7 m, 깊이 6~12 cm 둥근 구멍"""
    X, Y = _grid(nx, ny, hs)
    H = np.zeros((nx, ny))
    d_max = 0.08 + 0.04 * difficulty  # 이 타일의 가장 깊은 구멍
    # 중심 간격 0.9 m: 구멍 사이 바닥이 20 cm (2점) 이상 -> 얇은 칼날 벽이 안 생김
    # 가장자리 0.45 m 안쪽에만: 옆 타일 구멍과 사이에도 바닥이 남음
    for cx, cy in _darts(rng, nx * hs, ny * hs, int(rng.uniform(0.6, 1.0) * nx * ny * hs * hs), 0.9, margin=0.45):
        r, d = rng.uniform(0.15, 0.35), d_max * rng.uniform(0.75, 1.0)
        H = np.minimum(H, np.where(np.hypot(X - cx, Y - cy) < r, -d, 0.0))
    return _hf_unfold(H, hs)


def hf_honeycomb_walls(rng, nx, ny, hs, difficulty):
    """벌집 담: 0.6~0.9 m 육각 칸을 윗면 폭 0.1~0.2 m 낮은 수직 담이 둘러쌈 (담 위 +3.2~4 cm, 칸 바닥 -4.8~-6 cm)"""
    X, Y = _grid(nx, ny, hs)
    c, h = rng.uniform(0.6, 0.9), 0.08 + 0.02 * difficulty  # 칸 크기, 담 높이 (칸 바닥에서, 7.5 cm 넘어야 수직 벽)
    gx, gy = np.meshgrid(np.arange(-c, nx * hs + c, c), np.arange(-c, ny * hs + c, c * np.sqrt(3) / 2), indexing="ij")
    gx = gx + 0.5 * c * (np.arange(gy.shape[1]) % 2)  # 칸 중심: 줄 간격 c*0.87, 홀수 줄은 반 칸 밀기 -> 육각 배치
    C = np.stack([gx.ravel(), gy.ravel()], 1) + rng.uniform(-0.12, 0.12, (gx.size, 2)) * c  # 칸 중심을 조금씩 흔듦
    F1, F2, _ = _voronoi(X, Y, C)
    wall = F2 - F1 < 0.22  # 두 칸의 경계에서 약 0.11 m 안쪽 = 담 (어느 방향으로도 2점 이상 -> 두께 0 인 벽 없음)
    return _hf_unfold(np.where(wall, 0.4 * h, -0.6 * h), hs)


def hf_wide_ditches(rng, nx, ny, hs, difficulty):
    """넓은 도랑: 폭 0.5~0.8 m, 깊이 6~10 cm, 바닥이 평평한 도랑이 1.5~3 m 마다 길을 가로지름"""
    X, Y = _grid(nx, ny, hs)
    H = np.zeros((nx, ny))
    d_max = 0.08 + 0.02 * difficulty  # 이 타일의 가장 깊은 도랑
    yaw = rng.uniform(-0.3, 0.3)  # 도랑 방향: y 축에서 ±17도, 한 타일 안에서는 모두 나란히 (도랑끼리 만나는 곳의 두께 0 벽 방지)
    for x0 in _rows_along_x(rng, nx * hs, 1.5, 3.0, 1.0, 1.0):
        w, d = rng.uniform(0.4, 0.7), d_max * rng.uniform(0.75, 1.0)  # 도랑 폭 (벽 보정으로 약 0.1 m 넓어짐), 깊이
        u = _across(X, Y, x0, ny * hs / 2, yaw)
        H = np.minimum(H, np.where(np.abs(u) < w / 2, -d, 0.0))
    H[:, [0, -1]] = 0.0  # 타일 가장자리 한 줄은 바닥 높이: 옆 타일 도랑과 맞닿아 두께 0 인 벽이 생기지 않게
    H[[0, -1], :] = 0.0
    return _hf_unfold(H, hs)


def hf_cobble_street(rng, nx, ny, hs, difficulty):
    """돌 박힌 옛 포장길: 둥근 돌(간격 0.30~0.45 m) 벌집 배열, 줄눈~꼭대기 3~7 cm (평균 0, -3~+4 cm)."""
    X, Y = _grid(nx, ny, hs)
    s, yaw = rng.uniform(0.30, 0.45), rng.uniform(0, np.pi / 3)  # 돌 간격, 벌집 방향
    h0 = 0.035 + 0.025 * difficulty  # 돌 높이 (난이도가 클수록 높음)
    n = int(np.hypot(nx, ny) * hs / s)  # 돌린 뒤에도 타일을 덮는 격자 크기
    i, j = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1), indexing="ij")
    u, v = (i + 0.5 * (j % 2)) * s, j * s * np.sqrt(3) / 2  # 벌집 격자 (홀수 줄은 반 칸 밀림)
    cx = nx * hs / 2 + u * np.cos(yaw) - v * np.sin(yaw) + rng.uniform(-0.08, 0.08, u.shape) * s
    cy = ny * hs / 2 + u * np.sin(yaw) + v * np.cos(yaw) + rng.uniform(-0.08, 0.08, u.shape) * s
    near = (np.abs(cx - nx * hs / 2) < nx * hs / 2 + s) & (np.abs(cy - ny * hs / 2) < ny * hs / 2 + s)
    H = np.zeros((nx, ny))
    for x, y in zip(cx[near], cy[near]):
        r, h = s * rng.uniform(0.42, 0.5), min(h0 * rng.uniform(0.8, 1.25), 0.07)  # 7 cm 이하 -> 수직 벽 없음
        if rng.uniform() > 0.03:  # 3 % 는 빠진 돌 (줄눈 높이 그대로)
            H = np.maximum(H, h * np.sqrt(np.clip(1 - ((X - x) ** 2 + (Y - y) ** 2) / r**2, 0, None)))
    return H - H.mean()


def hf_riverbed_boulders(rng, nx, ny, hs, difficulty):
    """돌 많은 강바닥: 둥근 큰 돌(지름 0.3~0.8 m, 모래 위로 5~13 cm)이 면적의 약 3/4, 높이 -7.5(모래)~+5.5 cm."""
    X, Y = _grid(nx, ny, hs)
    H = np.zeros((nx, ny))
    spacing = 0.32 + 0.12 * difficulty  # 돌 중심 사이 최소 거리: 돌이 클수록 넓게 -> 난이도와 상관없이 모래 약 1/4
    for cx, cy in _darts(rng, nx * hs, ny * hs, 400, spacing, margin=-0.2):  # 돌 중심
        r = rng.uniform(0.15, 0.25 + 0.15 * difficulty)  # 긴 반지름 (난이도가 클수록 큰 돌)
        b, yaw, h = r * rng.uniform(0.6, 1.0), rng.uniform(0, np.pi), min(rng.uniform(0.35, 0.6) * r, 0.13)
        u = (X - cx) * np.cos(yaw) + (Y - cy) * np.sin(yaw)
        v = (Y - cy) * np.cos(yaw) - (X - cx) * np.sin(yaw)
        H = np.maximum(H, h * np.clip(1 - (u / r) ** 2 - (v / b) ** 2, 0, None) ** 0.4)  # 옆면이 가파른 둥근 돌
    # 가장 높은 곳을 +5.5 cm 에 맞추고(모래 약 -7.5 cm), 돌 옆 모래를 쌓아 턱을 7 cm 이하로
    return _no_walls(H - H.max() + 0.055)


def hf_railway_sleepers(rng, nx, ny, hs, difficulty):
    """철길 침목: 길에 수직인 침목(윗면 폭 0.2~0.3 m, 간격 0.6~0.75 m, 수직 옆면 9~12 cm), 윗면 +5 cm, 자갈 -4~-7.5 cm."""
    # 참고: 길을 따라 만나는 턱 높이·간격은 학습 때 상자 지형과 비슷함. 새로운 점은 길 전체를 가로지르는 규칙성뿐
    X, _ = _grid(nx, ny, hs)
    s, w = rng.uniform(0.6, 0.75), rng.uniform(0.3, 0.35)  # 침목 간격, 폭 (벽 보정이 0.1 m 깎아 윗면 0.2~0.3 m)
    h = 0.09 + 0.03 * difficulty  # 자갈 바닥 ~ 침목 윗면
    # 침목은 비스듬히 두지 않음: 비스듬한 침목의 계단 모서리가 타일 가장자리와 만나면 벽 보정이 삼각형을 뒤집음
    # 자갈 틈은 0.25 m 이상 (한 칸짜리 틈도 삼각형을 뒤집음)
    on = np.mod(X + rng.uniform(0, s), s) < w
    ballast = 0.05 - h + rng.uniform(-0.005, 0.005, X.shape)  # 자갈 (±0.5 cm), 가장 낮아도 -7.5 cm
    return np.where(on, 0.05, ballast)  # 침목 윗면 +5 cm: 뒤집힌 몸통이 얹혀도 중심 0.30 m


def hf_angular_scree(rng, nx, ny, hs, difficulty):
    """모난 돌 너덜지대: 제각각 기운 평평한 돌 조각(0.35~0.7 m 다각형)과 조각 사이 턱, 높이 -10~+5.5 cm."""
    X, Y = _grid(nx, ny, hs)
    C = _darts(rng, nx * hs, ny * hs, 1000, rng.uniform(0.35, 0.7), margin=-0.7)  # 조각 중심
    k = _voronoi(X, Y, C)[2]  # 칸마다 가장 가까운 조각 번호
    a = 0.04 + 0.03 * difficulty  # 조각 높이 차 (난이도가 클수록 큼)
    z0, g, th = rng.uniform(-a, a, len(C)), rng.uniform(0, 0.3, len(C)), rng.uniform(0, 2 * np.pi, len(C))
    H = z0[k] + g[k] * ((X - C[k, 0]) * np.cos(th[k]) + (Y - C[k, 1]) * np.sin(th[k]))  # 조각마다 기운 평면 (0~17도)
    H = H - H.mean()
    top = np.full(len(C), -1.0)
    np.maximum.at(top, k, H)  # 조각마다 가장 높은 점
    H = H - np.maximum(top - 0.055, 0)[k]  # 튀어나온 조각은 통째로 내려 꼭대기 +5.5 cm 이하 (뒤집힌 몸통 방지)
    # 아래는 -10 cm 에서 자르고(부스러기 바닥), 7 cm 넘는 턱은 낮은 쪽에 부스러기를 쌓아 두 단으로
    return _no_walls(np.maximum(H, -0.10))


def hf_drainage_grating(rng, nx, ny, hs, difficulty):
    """배수 격자 발판: 폭 0.3 m 발판(높이 0) 사이 사각 구멍(0.2/0.3 m, 벽 보정 뒤 0.3/0.4 m), 깊이 8~10 cm."""
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")  # 칸 번호
    bar, ox, oy = round(0.3 / hs), round(rng.choice([0.2, 0.3]) / hs), round(rng.choice([0.2, 0.3]) / hs)  # 칸 수
    hole = ((i + rng.integers(bar + ox)) % (bar + ox) < ox) & ((j + rng.integers(bar + oy)) % (bar + oy) < oy)
    hole[[0, -1], :] = False  # 타일 가장자리 한 줄은 발판
    hole[:, [0, -1]] = False
    d = 0.08 + 0.02 * difficulty  # 구멍 깊이
    H = np.where(hole, -d, 0.0)
    # 구멍의 (+x,-y) 모서리 바로 -y 쪽, (-x,+y) 모서리 바로 +y 쪽 발판을 조금 낮춰 턱을 7 cm 로:
    # 그대로 두면 Isaac Lab 벽 보정이 이 두 모서리에서 삼각형을 뒤집음
    P = np.pad(hole, 1)
    a = hole & ~P[2:, 1:-1] & ~P[1:-1, :-2]
    b = hole & ~P[:-2, 1:-1] & ~P[1:-1, 2:]
    H[np.roll(a, -1, 1) | np.roll(b, 1, 1)] = 0.07 - d
    return H


def hf_mud_cracks(rng, nx, ny, hs, difficulty):
    """갈라진 마른 논바닥: 흙판(1.0~1.5 m) 사이 V자 균열(폭 0.1~0.16 m), 가장자리 +1~1.5 cm, 균열 바닥 -4~-6 cm."""
    X, Y = _grid(nx, ny, hs)
    C = _darts(rng, nx * hs, ny * hs, 300, rng.uniform(0.9, 1.3), margin=-1.5)  # 흙판 중심 (간격 약 1.0~1.5 m)
    F1, F2, _ = _voronoi(X, Y, C)
    e = 0.5 * (F2 - F1)  # 가장 가까운 균열(흙판 경계)까지 거리
    cw, lip = rng.uniform(0.05, 0.08), rng.uniform(0.01, 0.015)  # 균열 반폭 (>= 0.05 -> 0.1 m 격자에 꼭 걸림), 말림 높이
    depth = 0.055 - lip + 0.015 * difficulty  # 가장자리 ~ 균열 바닥 7 cm 이하 -> 수직 벽 없음
    plate = lip * np.exp(-np.clip(e - cw, 0, None) / 0.10)  # 균열 쪽으로 말려 올라간 흙판
    return np.where(e < cw, -depth * (1 - e / cw), plate)


BATTERY_HF_KINDS = {  # 이름 -> 높이맵 함수
    "washboard": hf_washboard,
    "craters": hf_craters,
    "longitudinal_ruts": hf_longitudinal_ruts,
    "sawtooth": hf_sawtooth,
    "potholes": hf_potholes,
    "honeycomb_walls": hf_honeycomb_walls,
    "wide_ditches": hf_wide_ditches,
    "cobble_street": hf_cobble_street,
    "riverbed_boulders": hf_riverbed_boulders,
    "railway_sleepers": hf_railway_sleepers,
    "angular_scree": hf_angular_scree,
    "drainage_grating": hf_drainage_grating,
    "mud_cracks": hf_mud_cracks,
}


@height_field_to_mesh
def battery_hf_terrain(difficulty: float, cfg: "BatteryHfTerrainCfg") -> np.ndarray:
    """Isaac Lab 높이맵 지형 함수 형식. cfg.kind 이름의 함수로 높이맵을 만든다."""
    nx, ny = int(cfg.size[0] / cfg.horizontal_scale), int(cfg.size[1] / cfg.horizontal_scale)
    rng = np.random.default_rng(int(difficulty * 1_000_003))  # 타일마다 다른 난이도 값 -> 다른 모양 (재현 가능)
    H = BATTERY_HF_KINDS[cfg.kind](rng, nx, ny, cfg.horizontal_scale, difficulty)
    return np.rint(H / cfg.vertical_scale).astype(np.int16)


@configclass
class BatteryHfTerrainCfg(HfTerrainBaseCfg):
    function = battery_hf_terrain
    kind: str = ""  # BATTERY_HF_KINDS 의 이름


# ---------------------------------------------------------------------
# 15-3) 라이브러리 지형용 작은 도우미
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# 라이브러리 물체 대신 쓰는 작은 모양 2개 (MeshRepeatedCylinders 의 object_type 으로 끼움)
#  - 라이브러리는 물체마다 make_xxx(radius=, height=, center=, max_yx_angle=, degrees=) 를 부른다
#  - 무작위 값은 라이브러리 make_cylinder 와 똑같이 np.random 을 쓴다 (env seed 로 재현됨)
# ---------------------------------------------------------------------
def make_post(radius, height, center, **_):
    """말뚝·그루터기: 4~5각 기둥(절반 묻힘), 윗면을 8~14° 비스듬히 자른다."""
    post = trimesh.creation.cylinder(radius, height, sections=np.random.randint(4, 6))
    v = post.vertices.copy()
    top = v[:, 2] > 0.0  # 윗면 꼭짓점
    v[top, 2] += np.tan(np.radians(np.random.uniform(8.0, 14.0))) * v[top, 0]  # +x 쪽이 높게 비스듬히 자름
    post.vertices = v
    # 기둥 모서리 방향과 잘린 방향을 함께 무작위로 돌린다 (안 돌리면 모든 기둥이 같은 방향)
    post.apply_transform(trimesh.transformations.rotation_matrix(np.random.uniform(0.0, 2.0 * np.pi), (0, 0, 1)))
    post.apply_translation(center)  # 중심 높이 0 -> 높이의 절반만 땅 위로
    return post


def make_log(radius, height, center, **_):
    """쓰러진 통나무: 굵기 radius, 길이 height 인 둥근 기둥을 아무 방향으로 눕혀 반쯤 묻는다."""
    # 세운 기둥(z 축)을 y 축으로 90° 눕히고, z 축으로 무작위 방향만큼 돌린다
    lie = trimesh.transformations.rotation_matrix(np.pi / 2, (0, 1, 0))
    turn = trimesh.transformations.rotation_matrix(np.random.uniform(0.0, np.pi), (0, 0, 1))
    T = turn @ lie
    T[:3, 3] = center  # 중심 높이 0 -> 반지름만큼 땅 위로
    return trimesh.creation.cylinder(radius=radius, height=height, sections=16, transform=T)


# @configclass 를 꼭 붙여야 object_type 이 바뀐다 (빼면 조용히 원래 기둥이 나옴)
@configclass
class MeshRepeatedPostsTerrainCfg(MeshRepeatedCylindersTerrainCfg):
    object_type = make_post


@configclass
class MeshRepeatedLogsTerrainCfg(MeshRepeatedCylindersTerrainCfg):
    object_type = make_log


# ---------------------------------------------------------------------
# 밑이 뚫린 둥근 가로대 (라이브러리 floating_ring 의 네모 막대를 둥근 관으로 바꾼 것)
#  - 네모 막대는 윗면이 평평해서(폭 4~6 cm) 뒤집힌 몸통이 그 위에 딱 균형 잡히면 0.35~0.39 m 에 멈춘다
#  - 둥근 관은 꼭대기가 모서리 한 줄이라 얹혀도 굴러떨어진다
# ---------------------------------------------------------------------
def floating_pipes_terrain(difficulty, cfg):
    """둥근 관 4개(지름 4~6 cm)가 땅에서 6~10 cm 떠서 네모 틀을 이룬다 (관 꼭대기 12~14 cm)."""
    gap = cfg.gap_range[1] - difficulty * (cfg.gap_range[1] - cfg.gap_range[0])  # 관 밑 틈 (어려울수록 좁음)
    r = 0.5 * (cfg.pipe_diameter_range[0] + difficulty * (cfg.pipe_diameter_range[1] - cfg.pipe_diameter_range[0]))
    cx, cy = 0.5 * cfg.size[0], 0.5 * cfg.size[1]
    d = 0.5 * cfg.frame_width + r  # 틀 가운데 -> 관 중심
    lie = trimesh.transformations.rotation_matrix(np.pi / 2, (0, 1, 0))  # 세운 관을 x 방향으로 눕힘 (모서리가 위로)
    meshes = []
    for yaw, ox, oy in ((0.0, 0.0, d), (0.0, 0.0, -d), (np.pi / 2, d, 0.0), (np.pi / 2, -d, 0.0)):
        T = trimesh.transformations.rotation_matrix(yaw, (0, 0, 1)) @ lie
        T[:3, 3] = (cx + ox, cy + oy, gap + r)
        meshes.append(trimesh.creation.cylinder(radius=r, height=2.0 * (d + r), sections=16, transform=T))
    ground = trimesh.creation.box((cfg.size[0], cfg.size[1], 1.0))  # 두께 1 m 땅, 윗면 z = 0
    ground.apply_translation((cx, cy, -0.5))
    return meshes + [ground], np.array([cx, cy, 0.0])


@configclass
class MeshFloatingPipesTerrainCfg(SubTerrainBaseCfg):
    function = floating_pipes_terrain
    pipe_diameter_range: tuple[float, float] = (0.04, 0.06)  # 관 지름 (난이도 0 -> 1)
    gap_range: tuple[float, float] = (0.06, 0.10)  # 관 밑 틈 (난이도 0 -> 1 이면 10 -> 6 cm)
    frame_width: float = 2.8  # 네모 틀 안쪽 한 변 [m]


# ---------------------------------------------------------------------
# 15-4) 시험장 표: 이름 -> (한글 이름, 새로움, 지형 또는 바닥 재질)
#   - course  : 지형 모양 시험 (battery_course)
#   - material: 바닥 재질 시험. 원본 평면은 그대로 두고 scene.terrain.physics_material 의 값만 바꾼다
# ---------------------------------------------------------------------
class BatteryKind(NamedTuple):
    name_ko: str
    novelty: str
    course: TerrainGeneratorCfg | None = None
    material: dict | None = None


BATTERY = {
    # ================= 학습에 있던 종류 (기준용: 다른 시험장 점수와 비교하는 대조군) =================
    # 평지: 8 x 5 m 평면 타일. 원본의 무한 평면과는 "타일 메시"라는 점만 다르다
    "plane": BatteryKind("평지 (기준용)", REFERENCE, course=battery_course(MeshPlaneTerrainCfg())),
    # 계단 구덩이: 4 m 타일 가운데가 네모나게 꺼진다. 디딤판 0.5 m, 한 칸 1.5~2.5 cm
    #  -> 3칸 내려가 바닥(0.5 m, -6~-10 cm)을 지나 3칸 올라온다 (4 m 마다 반복)
    #  -> 타일 가운데(바닥)는 낮아서 출발점을 2 m 뒤 타일 가장자리(높이 0)로 옮긴다
    "stairs_pit": BatteryKind(
        "내려갔다 올라오는 계단 (기준용)",
        REFERENCE,
        course=battery_course(
            MeshInvertedPyramidStairsTerrainCfg(
                step_height_range=(0.015, 0.025), step_width=0.5, platform_width=1.0, border_width=0.25
            ),
            tile=(4.0, 4.0),
            spawn_shift_x=2.0,
        ),
    ),
    # 블록 바닥: 0.45 m 칸마다 높이 -h~+h (h = 3~5.5 cm). 가운데 한 칸(+h)에서 출발
    #  -> 가장 높은 칸도 +5.5 cm 라서 뒤집힌 몸통이 얹혀도 중심 0.305 m < 0.31 -> 종료됨
    #  (정사각 타일만 됨. 타일이 0.45 로 나누어떨어지면 에러 -> 5 m = 11칸 + 가장자리 5 cm)
    "blocks_grid": BatteryKind(
        "높낮이 블록 바닥 (기준용)",
        REFERENCE,
        course=battery_course(
            MeshRandomGridTerrainCfg(grid_width=0.45, grid_height_range=(0.03, 0.055), platform_width=0.45),
            tile=(5.0, 5.0),
        ),
    ),
    # 둔덕·오목: 2 m 타일마다 둔덕(꼭대기 +7.6~12 cm) 또는 오목(바닥 -7.6~12 cm)을 반반.
    #  꼭대기가 뾰족한 점이라(platform_width 0) 뒤집힌 몸통이 얹힐 평평한 곳이 없다
    "moguls": BatteryKind(
        "작은 둔덕·오목 (기준용)",
        REFERENCE,
        course=battery_course(
            {
                "up": HfPyramidSlopedTerrainCfg(proportion=0.5, slope_range=(0.08, 0.125), platform_width=0.0),
                "down": HfInvertedPyramidSlopedTerrainCfg(proportion=0.5, slope_range=(0.08, 0.125), platform_width=0.0),
            },
            tile=(2.0, 2.0),
        ),
    ),
    # 빨래판 잔물결: 학습 때 물결(파장 약 2 m)과 같은 사인 물결. 파장 0.4~0.6 m 만 새로움
    "washboard": BatteryKind(
        "빨래판 잔물결 (기준용)", REFERENCE, course=battery_course(BatteryHfTerrainCfg(kind="washboard"), slope_threshold=None)
    ),
    # 분화구: 둥근 오목·둔덕은 학습 때 물결·둔덕과 비슷함
    "craters": BatteryKind(
        "분화구밭 (기준용)", REFERENCE, course=battery_course(BatteryHfTerrainCfg(kind="craters"), slope_threshold=None)
    ),
    # ================= 크기·배치만 새로움 =================
    # 자갈밭: 0.1 m 점마다 높이 무작위, 평균 0, 타일마다 ±3 / ±4 / ±5 cm 중 하나 (라이브러리 지형은 difficulty 를 안 씀)
    #  학습 때 무작위 높이 바닥은 0.32~0.45 m 칸 크기로만 나옴 -> 0.1 m 크기만 새로움
    #  slope_threshold=None: ±4/±5 cm 는 이웃 차가 8~10 cm 라서 0.75 로 두면 한 점짜리 뾰족 벽·뒤집힌 삼각형이 생김
    "gravel": BatteryKind(
        "자갈밭",
        SCALE_NEW,
        course=battery_course(
            {
                "g3": HfRandomUniformTerrainCfg(proportion=1 / 3, noise_range=(-0.03, 0.03), noise_step=0.01),
                "g4": HfRandomUniformTerrainCfg(proportion=1 / 3, noise_range=(-0.04, 0.04), noise_step=0.01),
                "g5": HfRandomUniformTerrainCfg(proportion=1 / 3, noise_range=(-0.05, 0.05), noise_step=0.01),
            },
            slope_threshold=None,
        ),
    ),
    # 돌 박힌 옛 포장길: 돌 간격·높이는 학습 때 상자 칸(0.32~0.45 m)에서 겪음. 둥근 꼭대기와 벌집 배치만 새로움
    "cobble_street": BatteryKind("돌 박힌 옛 포장길", SCALE_NEW, course=battery_course(BatteryHfTerrainCfg(kind="cobble_street"))),
    # ================= 학습에 없던 종류 (모양) =================
    # [물체 뿌리기 4종 공통] 라이브러리는 타일마다 가운데를 비워 두는데(출발 자리),
    #  0번 행에서만 출발하므로 비울 필요가 없다 -> platform_width 0.1 (가운데 0.9 x 0.6 m 에만 물체 중심 없음)
    #  -> 대신 출발점을 5.3 m 뒤(첫 타일 앞 1.3 m 평지 테두리)로 옮겨 물체와 겹치지 않게 한다
    # 가시밭: 4·5각뿔 가시(높이 5.6~14.7 cm, 밑면 반지름 8~10 cm, 최대 10° 기울어짐)가 3~6 개/m²
    "spike_field": BatteryKind(
        "뾰족한 가시밭",
        NEW,
        course=battery_course(
            MeshRepeatedPyramidsTerrainCfg(
                object_params_start=MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                    num_objects=120, height=0.08, radius=0.08, max_yx_angle=0.0
                ),
                object_params_end=MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                    num_objects=240, height=0.14, radius=0.10, max_yx_angle=10.0
                ),
                rel_height_noise=(0.7, 1.05),
                platform_width=0.1,
                platform_height=0.01,
            ),
            spawn_shift_x=5.3,
        ),
    ),
    # 말뚝·그루터기: 가는 말뚝(반지름 6 cm, 3 개/m²) -> 굵은 그루터기(반지름 20 cm, 1 개/m²)
    #  보이는 높이 3.6~10 cm + 비스듬히 잘린 윗면(최고 약 15 cm). 윗면이 기울어서 뒤집힌 몸통이 못 얹힌다
    "posts": BatteryKind(
        "말뚝·그루터기 밭",
        NEW,
        course=battery_course(
            MeshRepeatedPostsTerrainCfg(
                object_params_start=MeshRepeatedPostsTerrainCfg.ObjectCfg(num_objects=120, height=0.12, radius=0.06),
                object_params_end=MeshRepeatedPostsTerrainCfg.ObjectCfg(num_objects=40, height=0.20, radius=0.20),
                rel_height_noise=(0.6, 1.0),
                platform_width=0.1,
                platform_height=0.01,
            ),
            spawn_shift_x=5.3,
        ),
    ),
    # 돌판 잔해: 0.3 x 0.2 ~ 0.45 x 0.3 m 돌판(절반 묻힘, 무작위 방향, 최대 10~15° 기울어짐)이 1.25~2.5 개/m²
    "tilted_rubble": BatteryKind(
        "기울어진 돌판 잔해",
        NEW,
        course=battery_course(
            MeshRepeatedBoxesTerrainCfg(
                object_params_start=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                    num_objects=50, height=0.10, size=(0.30, 0.20), max_yx_angle=10.0
                ),
                object_params_end=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                    num_objects=100, height=0.12, size=(0.45, 0.30), max_yx_angle=15.0
                ),
                rel_height_noise=(0.7, 1.0),
                platform_width=0.1,
                platform_height=0.01,
            ),
            spawn_shift_x=5.3,
        ),
    ),
    # 쓰러진 통나무: 길이 0.5~1.1 m, 반지름(=튀어나온 높이) 5~9 cm 통나무가 아무 방향으로 1~1.5 개/m²
    #  (통나무 3개가 좁게 둘러싼 드문 곳(레인의 약 0.2%)에서는 뒤집힌 몸통이 0.31~0.34 m 에 끼여 종료가 안 될 수 있다)
    "fallen_logs": BatteryKind(
        "쓰러진 통나무",
        NEW,
        course=battery_course(
            MeshRepeatedLogsTerrainCfg(
                object_params_start=MeshRepeatedLogsTerrainCfg.ObjectCfg(num_objects=40, height=0.6, radius=0.05),
                object_params_end=MeshRepeatedLogsTerrainCfg.ObjectCfg(num_objects=60, height=1.0, radius=0.09),
                rel_height_noise=(0.8, 1.1),
                platform_width=0.1,
                platform_height=0.01,
            ),
            spawn_shift_x=5.3,
        ),
    ),
    "riverbed_boulders": BatteryKind("돌 많은 강바닥", NEW, course=battery_course(BatteryHfTerrainCfg(kind="riverbed_boulders"))),
    "angular_scree": BatteryKind("모난 돌 너덜지대", NEW, course=battery_course(BatteryHfTerrainCfg(kind="angular_scree"))),
    # 철길 침목: 길을 따라 만나는 턱 높이(9~12 cm)·간격은 학습 때 상자 지형과 비슷함. 길 전체를 가로지르는 규칙성이 새로움
    "railway_sleepers": BatteryKind("철길 침목", NEW, course=battery_course(BatteryHfTerrainCfg(kind="railway_sleepers"))),
    # 밑이 뚫린 가로대: 땅에서 6~10 cm 떠 있는 둥근 관(지름 4~6 cm)이 2.8 m 네모 틀을 이룬다
    #  -> 4 m 마다 두 번 넘는다 (간격 약 2.9 m / 1.1 m). 발끝이 관 밑에 끼일 수 있다. 관 사이는 평지 (레인의 약 절반)
    "undercut_hurdles": BatteryKind(
        "밑이 뚫린 가로대(허들)", NEW, course=battery_course(MeshFloatingPipesTerrainCfg(), tile=(4.0, 4.0))
    ),
    # 별 모양 빔: 바닥이 4~8 cm 꺼져 있고, 가운데 원판에서 폭 25~40 cm 빔 5개(36° 간격)가 뻗는다
    #  빔 하나(+x 방향)가 달리는 선 바로 밑이라 몸통 밑은 평평하지만, 발(옆 0.48 m)은 빔과 바닥을 오르내린다
    #  (빔 길이를 size[0] 로만 계산하므로 정사각 타일)
    "star_beams": BatteryKind(
        "별 모양 빔",
        NEW,
        course=battery_course(
            MeshStarTerrainCfg(num_bars=5, bar_width_range=(0.25, 0.40), bar_height_range=(0.04, 0.08), platform_width=1.0),
            tile=(4.0, 4.0),
        ),
    ),
    "sawtooth": BatteryKind("톱니 경사", NEW, course=battery_course(BatteryHfTerrainCfg(kind="sawtooth"))),
    "honeycomb_walls": BatteryKind("벌집 낮은 담", NEW, course=battery_course(BatteryHfTerrainCfg(kind="honeycomb_walls"))),
    "mud_cracks": BatteryKind("갈라진 마른 논바닥", NEW, course=battery_course(BatteryHfTerrainCfg(kind="mud_cracks"))),
    "longitudinal_ruts": BatteryKind(
        "바퀴 자국 홈", NEW, course=battery_course(BatteryHfTerrainCfg(kind="longitudinal_ruts"), slope_threshold=None)
    ),
    "wide_ditches": BatteryKind("넓은 가로 도랑", NEW, course=battery_course(BatteryHfTerrainCfg(kind="wide_ditches"))),
    "potholes": BatteryKind("움푹 파인 구멍밭", NEW, course=battery_course(BatteryHfTerrainCfg(kind="potholes"))),
    "drainage_grating": BatteryKind("배수 격자 발판", NEW, course=battery_course(BatteryHfTerrainCfg(kind="drainage_grating"))),
    # 징검돌: 0.4~0.8 m 네모 돌(윗면 -4.5~+3.5 cm) 사이가 폭 0.2 m, 깊이 20 cm 인 V자 홈 격자
    #  (돌 사이 거리가 int() 로 잘려 늘 1칸 = V자 홈. 수직 틈은 안 생김)
    #  가운데 출발판은 없애고(platform_width 0) 출발점을 4.7 m 뒤(첫 타일 앞 0.7 m 평지)로 옮긴다
    #  벽 보정(slope_threshold)은 끈다: 홈은 원래 V자라 모양은 같고, 접힌 삼각형(PhysX 충돌 위험)이 0개가 됨
    "stepping_stones": BatteryKind(
        "징검돌 (V자 홈 격자)",
        NEW,
        course=battery_course(
            HfSteppingStonesTerrainCfg(
                stone_height_max=0.04,
                stone_width_range=(0.4, 0.8),
                stone_distance_range=(0.1, 0.2),
                holes_depth=-0.20,
                platform_width=0.0,
            ),
            slope_threshold=None,
            spawn_shift_x=4.7,
        ),
    ),
    # 바닥 없는 틈: 3 m 타일 가운데 2.2 m 판 둘레를 폭 10~25 cm 틈이 네모나게 두른다 (약 1.5 m 마다 틈)
    #  [주의] 이 시험장들 중 유일하게 틈 밑이 "허공"이다. 몸통(지름 0.5 m)은 안 빠지지만,
    #         빠진 다리를 못 빼면 몸통이 틈 가장자리까지 내려앉아 종료될 수 있다 -> 결과는 따로 본다
    "moat_gaps": BatteryKind(
        "바닥 없는 틈 건너기",
        NEW,
        course=battery_course(MeshGapTerrainCfg(gap_width_range=(0.10, 0.25), platform_width=2.2), tile=(3.0, 3.0)),
    ),
    # ================= 학습에 없던 종류 (바닥 재질: 원본 평면 그대로, 재질만. 3 계열) =================
    #  - 개미 몸에는 재질이 없어서 기본값(마찰 1.0 / 1.0, 반발 0, 섞는 방식 "average")을 쓴다
    #  - 두 재질이 닿으면 PhysX 는 섞는 방식 중 순위가 높은 것을 쓴다: average < min < multiply < max
    #    -> 바닥을 "multiply" 로 두면 실제 마찰 = 바닥 값 x 1.0 = 바닥 값 그대로
    #    -> 반발은 "max" 로 둬야 max(0.8, 0) = 0.8 ("average" 면 0.4 로 줄어듦)
    #  - 푹신함(compliant): 한쪽만 푹신하면 그쪽 값을 쓴다 (개미는 딱딱함 -> 바닥 값이 그대로 쓰임)
    # (1) 마찰 계열
    # ================= (19단계) Isaac Lab 기본 지형 중 학습에 안 넣은 3종 (기본 지형 7종은 C38 학습 줄로 들어감) =================
    # 구덩이: 타일 가운데가 5~12 cm 파이고 가장자리로 계단이 올라옴. 출발은 구덩이 바닥 -> 올라와서 다음 타일 구덩이로 내려감
    "pit": BatteryKind("구덩이 (기본 지형)", NEW, course=battery_course(MeshPitTerrainCfg(pit_depth_range=(0.05, 0.12), platform_width=2.0))),
    # 상자 단: 타일 가운데 상자 두 개가 쌓임 (한 단 5~15 cm, 최대 30 cm). 출발은 맨 위 -> 내려가서 다음 상자를 오름
    "box": BatteryKind("상자 단 (기본 지형)", NEW, course=battery_course(MeshBoxTerrainCfg(box_height_range=(0.05, 0.15), platform_width=2.0, double_box=True))),
    # 레일: 가운데 판 둘레와 가장자리에 폭 5~10 cm, 높이 5~15 cm 레일 두 겹
    "rails": BatteryKind("레일 (기본 지형)", NEW, course=battery_course(MeshRailsTerrainCfg(rail_thickness_range=(0.05, 0.10), rail_height_range=(0.05, 0.15), platform_width=2.0))),
    # ================= 재질 =================
    "ice": BatteryKind(
        "빙판 평지", NEW, material=dict(static_friction=0.05, dynamic_friction=0.05, friction_combine_mode="multiply")
    ),
    "stick_slip": BatteryKind(  # 가만히 디디면 1.0, 한번 미끄러지면 0.15
        "한번 미끄러지면 쭉 미끄러지는 바닥",
        NEW,
        material=dict(static_friction=1.0, dynamic_friction=0.15, friction_combine_mode="multiply"),
    ),
    "sticky_rubber": BatteryKind(
        "끈끈한 고무 바닥", NEW, material=dict(static_friction=2.0, dynamic_friction=1.8, friction_combine_mode="multiply")
    ),
    # (2) 반발 계열: 0.2 m/s 보다 빠르게 닿으면 튕긴다 (bounce_threshold_velocity 는 원본 그대로)
    "bouncy": BatteryKind("통통 튀는 바닥", NEW, material=dict(restitution=0.8, restitution_combine_mode="max")),
    # (3) 푹신함 계열: 바닥이 용수철(N/m) + 댐퍼(N·s/m). Ant 무게 약 9 N -> 발 하나 약 2 N -> 1~2 cm 꺼짐
    "mud": BatteryKind(  # 과감쇠: 꺼진 뒤 되튀지 않음
        "푹 꺼지는 진흙 바닥", NEW, material=dict(compliant_contact_stiffness=100.0, compliant_contact_damping=15.0)
    ),
    "sponge": BatteryKind(  # 저감쇠: 약 4~5 Hz 로 출렁임
        "출렁이는 스펀지 바닥", NEW, material=dict(compliant_contact_stiffness=200.0, compliant_contact_damping=1.0)
    ),
}


def use_battery_terrain(cfg, name):
    """평가 환경 cfg 의 바닥을 시험장 name 으로 바꾼다 (재질 시험이면 평면 그대로 + 재질만)."""
    kind = BATTERY[name]
    if kind.material is not None:
        cfg.scene.terrain.physics_material = cfg.scene.terrain.physics_material.replace(**kind.material)
        return
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = kind.course
    cfg.scene.terrain.max_init_terrain_level = 0  # 모든 Ant 가 0번 행(맨 뒤)에서 출발
    cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.50, 0.42))

# =====================================================================
# 4단계: 첫 번째 강건성 학습 조건 "C2" = 울퉁불퉁한 메시(mesh) 코스에서 학습
#
#  - 원본 AntEnvCfg 를 그대로 물려받고(상속) 바닥(scene.terrain)만 "학습 코스"로 바꾼다.
#  - 관측(60차원, 같은 순서), 행동, 보상 7개, 종료조건(몸통 월드 z < 0.31 m),
#    마찰(average 1.0/1.0/0.0), 에피소드 16초, 환경 4096개, PPO 설정은 baseline 과 완전히 같다.
#    -> baseline(평면 학습)과 "바닥만 다른" 비교.
#    -> 관측/네트워크가 원본과 같으므로 학습된 가중치는 원본 Isaac-Ant-v0 에서도 그대로 돈다 (SAFE 트랙).
#  - T2 평가 지형의 종류(HfRandomUniform 자갈/물결, HfDiscreteObstacles 블록)는 여기에 절대 넣지 않는다
#    -> T2 는 "학습 때 한 번도 못 본 지형 종류"로 남는다.
#  - 원본 파일은 import(가져다 쓰기)만 하고 절대 고치지 않는다.
#  - [주의] self.scene.terrain.xxx 의 xxx 철자를 틀려도 에러가 안 날 수 있다.
#           학습 폴더의 params/env.yaml 에서 값이 바뀌었는지 꼭 확인한다.
# =====================================================================

import copy  # (9단계 추가) C17 지형 은행
import json  # (9단계 추가)
import numpy as np  # (9단계 추가)

import isaaclab.sim as sim_utils
from isaaclab.terrains import (
    HfDiscreteObstaclesTerrainCfg,  # (19단계) 기본 지형 줄
    HfInvertedPyramidSlopedTerrainCfg,  # (19단계) 기본 지형 줄
    HfPyramidSlopedTerrainCfg,
    HfWaveTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    MeshRandomGridTerrainCfg,
    HfSteppingStonesTerrainCfg,  # (26단계 v12) 어려운 기본 지형 줄
    MeshBoxTerrainCfg,
    MeshGapTerrainCfg,
    MeshPitTerrainCfg,
    MeshRailsTerrainCfg,
    TerrainGenerator,  # (9단계 추가)
    TerrainGeneratorCfg,
)
from isaaclab.terrains.height_field import HfTerrainBaseCfg  # (9단계 추가) 절차적 지형
from isaaclab.terrains.height_field.utils import height_field_to_mesh  # (9단계 추가) 높이맵 -> 메시
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg

# ---------------------------------------------------------------------
# C2 학습 코스
#  - 8 m x 8 m 정사각형 타일 (박스 지형 MeshRandomGrid 는 정사각형 타일에서만 만들어짐)
#  - x 방향 24행 (192 m, 개미가 달리는 방향) x y 방향 16열 (128 m)
#  - curriculum=True : 뒤쪽(-x) 행일수록 쉽고 앞쪽(+x) 행일수록 어렵다 (난이도 0 -> 1)
#    개미는 맨 뒤 0~1행(가장 쉬움)에서만 출발 -> 멀리 갈수록 어려운 땅 = "거리 커리큘럼"
#  - 열마다 지형 종류가 하나로 정해진다 (비율대로):
#      0~3열 박스 / 4~7열 계단 / 8~11열 경사 / 12~13열 물결 / 14~15열 평지
#  - 코스 바깥 테두리 1 m 뒤는 허공: 옆이나 끝으로 빠지면 떨어져서 "몸통 높이" 종료
#    -> 평평한 곳으로 도망가서 점수를 버는 꼼수가 불가능
#  - 바닥 높이: 가장 낮은 곳 약 -0.12 m (박스 칸), 가장 높은 곳 약 0.83 m (맨 앞 행 계단 꼭대기)
#  - 삼각형 약 283만 개 (T2 v2 의 282만 개와 거의 같음)
# ---------------------------------------------------------------------
ANT_C2_COURSE_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),  # 타일 한 칸 크기 (x, y) [m]
    border_width=1.0,  # 코스 바깥 평평한 테두리 폭 [m]. 그 바깥은 허공
    num_rows=24,  # x 방향 타일 수: 24 x 8 = 192 m
    num_cols=16,  # y 방향 타일 수: 16 x 8 = 128 m
    horizontal_scale=0.1,  # 높이맵 격자 간격 [m] (T1, T2 와 같음)
    vertical_scale=0.005,  # 높이 최소 단위 [m]
    slope_threshold=0.75,  # 기본 ROUGH 와 같은 값
    curriculum=True,  # 행 번호가 클수록(앞쪽일수록) 어렵게
    difficulty_range=(0.0, 1.0),  # 0행 난이도 약 0 -> 23행 난이도 약 1
    use_cache=False,  # 지형을 파일로 저장해 두지 않음
    sub_terrains={
        # (1) 박스: 0.45 m 칸마다 높이 -h ~ +h 무작위. 기본 ROUGH 의 boxes 와 같은 종류, 높이만 개미용으로 낮춤
        #     h = 1 cm (0행) -> 10 cm (23행)          [기본 ROUGH: 5 -> 20 cm]
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.25, grid_width=0.45, grid_height_range=(0.01, 0.10), platform_width=2.0
        ),
        # (2) 계단 (오르막 피라미드만): 디딤판 0.3 m 6계단 + 꼭대기 판
        #     한 칸 높이 0 cm -> 12 cm, 꼭대기 높이 0.01 m(0행) -> 0.83 m(23행)
        #     [아래끝을 0 으로 둔 이유] 출발 행(0~1행)의 꼭대기 판이 0.06 m 보다 높으면,
        #     그 위에서 개미가 넘어져도 몸통 z 가 0.31 m 아래로 안 내려가 16초 내내 안 끝난다(샘플 낭비).
        #     0 으로 두면 출발 32곳 중 3곳만 0.06 m 이상 (구안은 9~10곳). 23행 난이도는 구안과 동일.
        #     움푹 꺼진 역계단은 넣지 않음 (월드 z 종료 때문에 들어가자마자 끝나서 배울 것이 없음)
        "stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.0, 0.12),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # (3) 경사 (오르막 피라미드만): 기울기 0 -> 0.4 (약 22도), 꼭대기 0 -> 약 0.8 m  [기본 ROUGH 와 같은 기울기 범위]
        "slopes": HfPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        # (4) 물결: 파장 2 m 사인파, 높이 0 -> ±0.09 m   [기본 ROUGH 에는 없는 종류]
        "waves": HfWaveTerrainCfg(proportion=0.125, amplitude_range=(0.0, 0.1), num_waves=4, border_width=0.25),
        # (5) 평지: 경사 0 짜리 피라미드 = 0.1 m 간격 삼각형으로 잘게 쪼갠 평평한 바닥
        #     baseline 이 55점으로 무너진 "T2 평지 대조"와 같은 종류의 바닥 -> 메시 접촉에 익숙해지기
        "flat": HfPyramidSlopedTerrainCfg(
            proportion=0.125, slope_range=(0.0, 0.0), platform_width=2.0, border_width=0.25
        ),
    },
)


# ---------------------------------------------------------------------
# C2 학습 환경: 원본 AntEnvCfg + 바닥만 위의 코스로 교체
# ---------------------------------------------------------------------
@configclass
class AntRoughTrainEnvCfg(AntEnvCfg):
    def __post_init__(self):
        # 1) 원본 Ant 설정을 먼저 전부 적용 (관측, 보상, 종료, 마찰, 환경 4096개, 16초)
        super().__post_init__()
        # 2) 바닥을 평면(plane) -> C2 학습 코스(generator) 로 교체
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ANT_C2_COURSE_CFG
        # 3) 출발은 맨 뒤 0~1행(x = -92 m, -84 m)에서만 -> 16열 x 2행 = 출발점 32곳 (한 곳에 약 128마리)
        #    -> 앞쪽 코스 180~188 m 확보 (개미는 평면에서 16초에 약 141 m 달림)
        self.scene.terrain.max_init_terrain_level = 1
        # 4) 바닥 색 (GUI 에서 보기 위한 것. 물리에는 영향 없음)
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.40, 0.50, 0.65))
        # 5) PhysX 접촉 버퍼를 기본값의 2배로 (Isaac Lab 울퉁불퉁 지형 학습 velocity_env_cfg.py 와 같은 값)
        #    이 값은 "호스트(PC) 쪽 고정 메모리 칸 수"라서 넘치지 않는 한 물리 결과는 똑같다.
        #    넘치면 PhysX 가 로그에 "Patch buffer overflow detected ..." 오류를 찍는다(조용히 넘어가지 않음).
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


# =====================================================================
#
#  - C2 코스와 같은 틀: 8 m 정사각형 타일, 24행(192 m, 달리는 방향), 맨 뒤 0~1행에서 출발,
#    앞으로 갈수록 어려워지는 "거리 커리큘럼", 테두리 1 m 바깥은 허공.
#  - C2 대비 바뀐 점 (영상에서 본 "발이 걸려 넘어짐", "T1 구덩이에 갇힘" 을 겨냥):
#      박스   10 cm -> 15 cm        계단 한 칸 12 cm -> 20 cm (T1 최대 23 cm 에 가깝게)
#      경사   0.4  -> 0.5          물결  10 cm -> 15 cm
#      [새로] 얕은 역계단(구덩이): 한 칸 최대 3 cm x 5칸 = 최대 15 cm 깊이
#      열 수 16 -> 20 (종류별 열이 늘어 다양성 증가)
#  - [역계단을 15 cm 까지만 두는 이유] 몸통 월드 z < 0.31 m 이면 종료(채점 쪽이 강제).
#    Ant 몸통은 평소 약 0.5 m 높이 -> 약 0.19 m 보다 깊은 구덩이는 들어가는 순간 끝나서 배울 게 없다.
#  - T2 평가 지형의 종류(HfRandomUniform 자갈/물결, HfDiscreteObstacles 블록)는 여전히 넣지 않는다.
# =====================================================================
ANT_C7_COURSE_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=1.0,
    num_rows=24,  # 24 x 8 = 192 m (C2 와 같음)
    num_cols=20,  # 20 x 8 = 160 m (C2 는 16열)
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        # (1) 박스: 칸마다 높이 -h ~ +h,  h = 2 cm -> 15 cm        [C2: 1 -> 10 cm]
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.02, 0.15), platform_width=2.0
        ),
        # (2) 오르막 계단: 한 칸 0 -> 20 cm                          [C2: 0 -> 12 cm]
        "stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # (3) [새로] 얕은 구덩이(역계단): 한 칸 0 -> 3 cm, 5칸 -> 깊이 최대 15 cm
        "pits": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.03),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # (4) 오르막 경사: 기울기 0 -> 0.5 (약 27도)                  [C2: 0 -> 0.4]
        "slopes": HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.25
        ),
        # (5) 물결: 높이 0 -> ±15 cm                                  [C2: 0 -> 10 cm]
        "waves": HfWaveTerrainCfg(proportion=0.15, amplitude_range=(0.0, 0.15), num_waves=4, border_width=0.25),
        # (6) 평지(잘게 쪼갠 메시): 평지 실력을 잃지 않도록 20% 유지
        "flat": HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.0), platform_width=2.0, border_width=0.25
        ),
    },
)


# =====================================================================
#
#  - 정해진 지형 종류의 숫자만 바꾸는 것이 아니라, 기본 도형들을 무작위로 조합해 "처음 보는 모양"의 지형을 만든다.
#  - 지형 유전자(genome) = 기본 도형 1~4개의 목록. 도형마다 종류 + 설정값 + seed(무작위 배치).
#      noise   : 부드러운 굴곡 (여러 크기의 언덕·골짜기, 파장 1~4 m)
#      bumps   : 둥근 혹 / 웅덩이 여러 개
#      stairs  : 계단 언덕 (올라갔다 평평했다 다시 내려옴) 또는 계단 골짜기(down=1). 살짝 비스듬할 수 있음
#      boxgrid : 칸마다 높이가 다른 박스 바닥
#      ramp    : 경사로 언덕 (오르막 -> 평평한 꼭대기 -> 내리막). 오르막 길이 1~4 m
#      ridges  : 비스듬한 물결 줄무늬 (파장 1~3 m)
#  - 같은 줄(lane)의 타일들은 같은 유전자를 쓰되 타일마다 seed 가 달라서(난이도 값에서 파생) 모양이 조금씩 다르다.
#    행이 클수록(앞쪽) 높이가 커진다: 높이 x (0.3 + 0.7 x 난이도)  -> 줄 안에서도 점진적으로 어려워짐
#  - [풀 수 있게 보정] (1) 가장 낮은 곳 -15 cm (몸통 월드 z < 0.31 m 종료 때문에 약 0.19 m 보다 깊으면 무조건 탈락)
#                     (2) 이웃한 두 점(0.1 m 간격)의 높이 차 최대 25 cm. 넘으면 타일 전체 높이를 비율대로 줄임
#                     (3) 가장 높은 곳 +1.2 m
#                     (4) 타일 가장자리 0.5 m 는 부드럽게 높이 0 으로 -> 타일 이음매에 절벽이 안 생김
#  - [T2 를 "처음 보는 지형"으로 지키기] T2 의 자갈(0.1 m 간격 무작위)·물결(0.5 m 간격 무작위)·평평한 사각 블록은 흉내 내지 않는다:
#      굴곡·능선의 파장은 1 m 이상, 평평한 사각 장애물 도형은 없음.
# =====================================================================

# 도형별 설정값 범위: 이름 -> (최솟값, 최댓값, 정수 여부)
PROC_PRIMITIVES = {
    "noise": {"amp": (0.04, 0.25, False), "wavelength": (1.0, 4.0, False), "octaves": (1, 3, True)},
    "bumps": {"count": (2, 20, True), "radius": (0.2, 1.2, False), "height": (0.03, 0.25, False)},
    "stairs": {"step_h": (0.03, 0.20, False), "step_w": (0.25, 0.60, False), "angle": (-0.6, 0.6, False), "down": (0, 1, True)},
    "boxgrid": {"cell": (0.30, 0.60, False), "height": (0.02, 0.15, False)},
    "ramp": {"slope": (0.10, 0.50, False), "angle": (-0.8, 0.8, False), "length": (1.0, 4.0, False)},
    "ridges": {"amp": (0.02, 0.12, False), "wavelength": (1.0, 3.0, False), "angle": (-1.2, 1.2, False)},
    #      일부러 넣지 않음 -> "학습에 없던 종류" 시험으로 남김. 아래 도형은 일반 형태(다각형 조각, 막대, 원판, 불규칙 계단, 2차원 물결)
    "cells": {"cell": (0.4, 1.5, False), "height": (0.03, 0.15, False), "tilt": (0.0, 0.3, False)},
    "bars": {"count": (1, 8, True), "width": (0.2, 0.8, False), "height": (0.03, 0.15, False), "length": (1.0, 6.0, False)},
    "discs": {"count": (2, 12, True), "radius": (0.15, 0.7, False), "height": (0.03, 0.15, False)},
    "terraces": {"steps": (2, 6, True), "rise": (0.03, 0.15, False), "tread": (0.4, 1.5, False), "angle": (-0.8, 0.8, False)},
    "waves2d": {"amp": (0.02, 0.15, False), "wx": (0.6, 3.0, False), "wy": (0.6, 3.0, False), "angle": (-1.2, 1.2, False)},
}
PROC_KINDS_V1 = ("noise", "bumps", "stairs", "boxgrid", "ramp", "ridges")  # C17/C17b/C27/C29/C30 이 쓴 원래 6종
PROC_MIN_HEIGHT = -0.15  # [m] 가장 깊은 곳
PROC_MAX_HEIGHT = 1.2  # [m] 가장 높은 곳
PROC_MAX_STEP = 0.25  # [m] 이웃한 두 점의 최대 높이 차
PROC_TAPER = 0.5  # [m] 가장자리를 0 으로 잇는 폭
PROC_MAX_PRIMITIVES = 4


def random_primitive(rng, kind=None, kinds=None):
    """무작위 도형 하나 (종류를 안 주면 kinds 목록(기본: 전부)에서 무작위)."""
    kinds = list(kinds or PROC_PRIMITIVES)
    kind = kind or kinds[int(rng.integers(len(kinds)))]
    prim = {"kind": kind, "seed": int(rng.integers(1_000_000))}
    for key, (lo, hi, is_int) in PROC_PRIMITIVES[kind].items():
        prim[key] = int(rng.integers(lo, hi + 1)) if is_int else round(float(rng.uniform(lo, hi)), 4)
    return prim


def random_genome(rng, max_primitives=2, kinds=None):
    """무작위 지형 유전자 (도형 1~max_primitives 개, 종류는 kinds 목록에서)."""
    return [random_primitive(rng, kinds=kinds) for _ in range(int(rng.integers(1, max_primitives + 1)))]


def _perturb_primitive(prim, rng, hard_scale=1.0):
    """도형 하나의 설정값을 조금 흔든다 (정수 +-1, 각도 +-0.2 rad, 나머지 x0.8~1.25). 범위 밖은 자름.

    hard_scale: (C30) 세대가 지날수록 "상한"을 이 배수만큼 올린다 (1.0 = 원래 상한).
                구덩이(pits 성격의 stairs down 깊이)·각도·개수는 올리지 않는다.
                어차피 render_genome 이 최저 -0.15 m / 이웃 턱 0.25 m / 최고 1.2 m 로 잘라서 "풀 수 있는 범위"는 유지된다.
    """
    for key, (lo, hi, is_int) in PROC_PRIMITIVES[prim["kind"]].items():
        if hard_scale != 1.0 and not is_int and key != "angle":
            hi = hi * hard_scale
        if is_int:
            prim[key] = int(np.clip(prim[key] + rng.integers(-1, 2), lo, hi))
        elif key == "angle":
            prim[key] = round(float(np.clip(prim[key] + rng.uniform(-0.2, 0.2), lo, hi)), 4)
        else:
            prim[key] = round(float(np.clip(prim[key] * rng.uniform(0.8, 1.25), lo, hi)), 4)


def mutate_genome(genome, rng, hard_scale=1.0, kinds=None):
    """유전자 변형을 1~2번: 설정값 흔들기 50% / 도형 추가 20% / 도형 삭제 15% / 새 배치(seed) 15%.
    (추가·삭제가 불가능하면 대신 설정값을 흔든다).  hard_scale: C30 의 난이도 상한 배수. kinds: 추가할 도형 종류 목록."""
    child = copy.deepcopy(genome)
    for _ in range(int(rng.integers(1, 3))):
        op = rng.uniform()
        if op < 0.5:
            _perturb_primitive(child[int(rng.integers(len(child)))], rng, hard_scale)
        elif op < 0.7:
            if len(child) < PROC_MAX_PRIMITIVES:
                child.append(random_primitive(rng, kinds=kinds))
            else:
                _perturb_primitive(child[int(rng.integers(len(child)))], rng, hard_scale)
        elif op < 0.85:
            if len(child) > 1:
                child.pop(int(rng.integers(len(child))))
            else:
                _perturb_primitive(child[0], rng, hard_scale)
        else:
            child[int(rng.integers(len(child)))]["seed"] = int(rng.integers(1_000_000))
    return child


def _value_noise(X, Y, wavelength, rng):
    """부드러운 값 노이즈: wavelength 간격 격자에 무작위 값 -> 부드럽게(smoothstep) 보간. 범위 약 [-1, 1]."""
    gx, gy = X / wavelength, Y / wavelength
    nx, ny = int(gx.max()) + 2, int(gy.max()) + 2
    grid = rng.uniform(-1.0, 1.0, size=(nx, ny))
    ix, iy = np.floor(gx).astype(int), np.floor(gy).astype(int)
    fx, fy = gx - ix, gy - iy
    sx, sy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a, b = grid[ix, iy], grid[ix + 1, iy]
    c, d = grid[ix, iy + 1], grid[ix + 1, iy + 1]
    return (a * (1 - sx) + b * sx) * (1 - sy) + (c * (1 - sx) + d * sx) * sy


def render_genome(genome, size_x, size_y, hs, difficulty, tile_seed=0):
    """지형 유전자 -> 높이맵 [m] (x 점 수, y 점 수). 풀 수 있게 보정까지 끝낸 값."""
    nx, ny = int(size_x / hs), int(size_y / hs)
    X, Y = np.meshgrid(np.arange(nx) * hs, np.arange(ny) * hs, indexing="ij")
    H = np.zeros((nx, ny))
    for prim in genome:
        rng = np.random.default_rng(prim["seed"] + tile_seed)
        k = prim["kind"]
        if k == "noise":
            for o in range(prim["octaves"]):
                H += prim["amp"] * (0.5**o) * _value_noise(X, Y, max(prim["wavelength"] / 2**o, 1.0), rng)
        elif k == "bumps":
            for _ in range(prim["count"]):
                cx, cy = rng.uniform(0, size_x), rng.uniform(0, size_y)
                r = rng.uniform(0.5, 1.0) * prim["radius"]
                h = prim["height"] * rng.uniform(0.5, 1.0) * (1 if rng.uniform() < 0.6 else -1)
                H += h * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * r * r))
        elif k in ("stairs", "ramp", "ridges"):
            u = X * np.cos(prim["angle"]) + Y * np.sin(prim["angle"])  # 대략 달리는 방향(x)을 따라가는 좌표
            u = u - u.min()
            if k == "ridges":
                H += prim["amp"] * np.sin(2 * np.pi * u / prim["wavelength"] + rng.uniform(0, 2 * np.pi))
            else:
                # 언덕 모양: 앞 가장자리에서 u0 만큼 들어간 곳부터 오르고, 뒤 가장자리 u1 만큼 전에 다 내려온다
                # t = "가까운 쪽 끝까지의 거리" -> 올라갔다 내려오는 대칭 모양이라 타일 끝에서 절벽이 안 생김
                u0, u1 = rng.uniform(0, 0.2) * u.max(), rng.uniform(0, 0.2) * u.max()
                t = np.clip(np.minimum(u - u0, (u.max() - u1) - u), 0, None)
                if k == "stairs":
                    H += (-1 if prim["down"] else 1) * prim["step_h"] * np.floor(t / prim["step_w"])
                else:
                    H += prim["slope"] * np.clip(t, 0, prim["length"])
        elif k == "boxgrid":
            cx, cy = (X // prim["cell"]).astype(int), (Y // prim["cell"]).astype(int)
            table = rng.uniform(-prim["height"], prim["height"], size=(cx.max() + 1, cy.max() + 1))
            H += table[cx, cy]
        elif k == "cells":  # 다각형 조각 바닥: 흔든 격자점의 보로노이 칸마다 높이 ±height, 칸마다 살짝 기울어짐(tilt)
            c = prim["cell"]
            gx, gy = np.meshgrid(np.arange(-c, size_x + c, c), np.arange(-c, size_y + c, c), indexing="ij")
            C = np.stack([gx.ravel(), gy.ravel()], 1) + rng.uniform(-0.4, 0.4, (gx.size, 2)) * c
            idx = np.argmin((X[..., None] - C[:, 0]) ** 2 + (Y[..., None] - C[:, 1]) ** 2, axis=-1)
            z0, g, th = rng.uniform(-prim["height"], prim["height"], len(C)), rng.uniform(0, prim["tilt"], len(C)), rng.uniform(0, 2 * np.pi, len(C))
            H += z0[idx] + g[idx] * ((X - C[idx, 0]) * np.cos(th[idx]) + (Y - C[idx, 1]) * np.sin(th[idx]))
        elif k == "bars":  # 곧은 막대 (60% 솟음 / 40% 꺼짐), 아무 방향, 윗면 평평
            for _ in range(prim["count"]):
                cx, cy, yaw = rng.uniform(0, size_x), rng.uniform(0, size_y), rng.uniform(0, np.pi)
                u, v = (X - cx) * np.cos(yaw) + (Y - cy) * np.sin(yaw), -(X - cx) * np.sin(yaw) + (Y - cy) * np.cos(yaw)
                inside = (np.abs(u) < prim["length"] / 2) & (np.abs(v) < prim["width"] / 2)
                H += np.where(inside, prim["height"] * rng.uniform(0.5, 1.0) * (1 if rng.uniform() < 0.6 else -1), 0.0)
        elif k == "discs":  # 원판 (60% 솟음 / 40% 꺼짐), 윗면 평평
            for _ in range(prim["count"]):
                cx, cy, r = rng.uniform(0, size_x), rng.uniform(0, size_y), prim["radius"] * rng.uniform(0.5, 1.0)
                sign = 1 if rng.uniform() < 0.6 else -1
                H += np.where((X - cx) ** 2 + (Y - cy) ** 2 < r * r, sign * prim["height"] * rng.uniform(0.5, 1.0), 0.0)
        elif k == "terraces":  # 불규칙 계단 언덕: 단마다 높이·너비가 다름 (stairs 는 모두 같음)
            u = X * np.cos(prim["angle"]) + Y * np.sin(prim["angle"])
            u = u - u.min()
            u0, u1 = rng.uniform(0, 0.2) * u.max(), rng.uniform(0, 0.2) * u.max()
            t = np.clip(np.minimum(u - u0, (u.max() - u1) - u), 0, None)
            edge, z = 0.0, np.zeros_like(H)
            for _ in range(prim["steps"]):
                edge += prim["tread"] * rng.uniform(0.5, 1.0)
                z += np.where(t > edge, prim["rise"] * rng.uniform(0.3, 1.0), 0.0)
            H += z
        elif k == "waves2d":  # 2차원 물결 (달걀판): 두 방향 파장이 다름
            u = X * np.cos(prim["angle"]) + Y * np.sin(prim["angle"])
            v = -X * np.sin(prim["angle"]) + Y * np.cos(prim["angle"])
            H += prim["amp"] * np.sin(2 * np.pi * u / prim["wx"] + rng.uniform(0, 2 * np.pi)) * np.cos(2 * np.pi * v / prim["wy"] + rng.uniform(0, 2 * np.pi))
    # 행이 클수록 높게 (줄 안에서의 점진적 난이도)
    H *= 0.3 + 0.7 * difficulty
    # 가장자리를 부드럽게 0 으로 (타일 이음매 절벽 방지)
    def taper(n):
        d = np.minimum(np.arange(n), np.arange(n)[::-1]) * hs / PROC_TAPER
        d = np.clip(d, 0, 1)
        return d * d * (3 - 2 * d)
    H *= taper(nx)[:, None] * taper(ny)[None, :]
    # 풀 수 있게 보정
    H = np.clip(H, PROC_MIN_HEIGHT, PROC_MAX_HEIGHT)
    step = max(np.abs(np.diff(H, axis=0)).max(initial=0), np.abs(np.diff(H, axis=1)).max(initial=0))
    if step > PROC_MAX_STEP:
        H *= PROC_MAX_STEP / step
    return H


@height_field_to_mesh
def procedural_terrain(difficulty: float, cfg: "ProceduralTerrainCfg") -> np.ndarray:
    """Isaac Lab 높이맵 지형 함수 (hf_terrains.py 의 함수들과 같은 형식). 정수 높이맵을 돌려준다."""
    tile_seed = int(difficulty * 1_000_003)  # 난이도 값(행 + 무작위)에서 타일마다 다른 seed
    H = render_genome(cfg.genome or [], cfg.size[0], cfg.size[1], cfg.horizontal_scale, difficulty, tile_seed)
    return np.rint(H / cfg.vertical_scale).astype(np.int16)


@configclass
class ProceduralTerrainCfg(HfTerrainBaseCfg):
    """절차적 지형 한 칸의 설정. genome 은 지형 은행(LevelBankTerrainGenerator)이 줄마다 채운다."""

    function = procedural_terrain
    genome: list | None = None


# =====================================================================
#
#  한 줄 요약: "Ant 가 가장 배울 게 많은 지형 줄의 유전자를 조금씩 변형해서, 처음 보는 지형을 계속 만든다."
#
#  - 계보: ACCEL (Parker-Holder, Jiang et al., ICML 2022) = PLR 로 약점 레벨을 고르고, 그 레벨을 "편집"해서 새 레벨을 만든다.
#          지형 유전자를 변형해 걷는 로봇이 배울 수 있는 지형만 남기는 방식은 POET (Wang, Lehman, Clune, Stanley 2019) / Enhanced POET (2020) 이 먼저.
#          여기서는 Isaac Lab 의 "지형은 시작할 때 한 번만" 제약 때문에 세대 단위(체크포인트 이어서)로 돌린 것이 다른 점.
#          GAN 이 아니다 (판별자·진짜 데이터 없음). "GAN 처럼 두 쪽이 겨루는 구도"까지만 비유 가능.
#  - Isaac Lab 제약: 지형 메시는 학습 "시작할 때 한 번만" 만들어진다 -> "세대" 단위로 돈다.
#      세대 g 학습 (PLR 출제자 mdp.py PrioritizedTerrainReplay 가 칸별 점수를 npz 로 남김)
#      -> 이 파일의 evolve_bank() 가 약점 줄의 유전자를 변형한 새 은행(json) -> 체크포인트에서 이어서 세대 g+1 학습
#      (반복 스크립트: RS_수업자료/3주차/tools/run_accel.sh)
#  - 은행 = 20개 "줄(lane, = 열)":  0~15열 = 절차적 지형(유전자) / 16~19열 = 평지(절대 안 바꿈, 평지 실력 유지)
#    줄 안에서는 뒤(0행) -> 앞(23행)으로 갈수록 높이가 커진다 (render_genome 의 0.3 + 0.7 x 난이도)
#  - 진화 규칙 (세대마다):
#      줄 점수 = 그 줄 출발 칸(0~7행)들의 PLR 점수(후회 근사) 평균
#      점수 상위 4줄 = 부모 / 하위 4줄 = 교체 대상
#      교체 4줄 중 1줄 = 완전히 새로운 무작위 유전자 ("새 종", 다양성 유지), 3줄 = 부모 유전자를 변형한 자식
# =====================================================================
C17_NUM_FLAT_LANES = 4  # 오른쪽 끝 4줄은 평지로 고정


def c17_default_bank(gen_cfg, seed=0, kinds=None):
    """0세대 은행: 절차적 줄 = 무작위 유전자(도형 1~2개, 종류는 kinds 에서), 마지막 4줄 = 평지."""
    rng = np.random.default_rng(seed)
    n_proc = gen_cfg.num_cols - C17_NUM_FLAT_LANES
    lib = list(getattr(gen_cfg, "library_lanes", None) or [])  # (19단계) Isaac Lab 기본 지형 줄 (줄 종류 = sub_terrains 키)
    n_proc -= len(lib)
    lanes = [{"type": "procedural", "genome": random_genome(rng, kinds=kinds), "born": 0} for _ in range(n_proc)]
    lanes += [{"type": key} for key in lib]
    lanes += [{"type": "flat"} for _ in range(C17_NUM_FLAT_LANES)]
    return {"generation": 0, "lanes": lanes, "history": []}


def evolve_bank(bank, lane_scores, rng, num_replace=4, num_new=1, min_kind_lanes=0, max_kind_lanes=0, hard_growth=0.0, hard_growth_max=0.5, kinds=None, num_anchor=0):
    """ACCEL 식 편집: 점수 상위 줄의 유전자를 변형해 하위 줄을 교체한 다음 세대 은행을 돌려준다.

    min_kind_lanes: (C17b) 도형 종류(noise/bumps/stairs/boxgrid/ramp/ridges)마다 그 종류를 가진 줄이 최소 이만큼은 남도록
                    교체 대상을 고른다. 0 이면 제한 없음(C17).
                    [이유] C17 은 세대가 지날수록 경사로·구덩이 위주로 쏠리고(figures/accel_evolution.png),
                    T2 점수가 70.8(5세대) -> 60.4(30세대)로 떨어졌다 = 다양성 붕괴.
    max_kind_lanes: (C17b) 한 도형 종류를 가진 줄이 이보다 많아지지 않게. 자식이 상한을 넘기면 새 무작위 유전자로 대체.
                    0 이면 제한 없음. (CPU 시험: 최소 규칙만으로는 경사로가 16줄 중 14줄 -> 상한 필요)
    hard_growth:    (C30) 세대마다 설정값 "상한"을 이만큼씩 키운다 (0.01 = 세대당 +1%). 0 이면 고정.
    hard_growth_max: 상한 확대의 최대치 (0.5 = 원래 상한의 1.5 배까지). 안전 보정(render_genome)은 그대로 적용된다.
    kinds:          (C33) 새 유전자·추가 도형에 쓸 도형 종류 목록 (None = PROC_PRIMITIVES 전부)
    num_anchor:     (C34) 앞쪽 절차적 줄 이만큼은 "닻"으로 절대 교체하지 않음 (0세대 지형을 끝까지 유지 -> 기본 걸음을 잊지 않게)
    """
    new = copy.deepcopy(bank)
    lanes = new["lanes"]
    gen = bank["generation"] + 1
    hard_scale = 1.0 + min(hard_growth * gen, hard_growth_max)
    mutable = [c for c, lane in enumerate(lanes) if lane["type"] == "procedural"][num_anchor:]  # 닻 줄은 교체 대상에서 뺌
    order = sorted(mutable, key=lambda c: lane_scores[c], reverse=True)
    k = min(num_replace, len(order) // 2)
    parents = order[:k]
    # 교체 대상: 점수 낮은 줄부터. 단 그 줄을 빼면 어떤 도형 종류가 min_kind_lanes 아래로 떨어지면 건너뜀
    kind_count = {kind: 0 for kind in PROC_PRIMITIVES}
    for c, lane in enumerate(lanes):  # 종류별 줄 수는 닻 줄까지 포함해서 셈
        if lane["type"] == "procedural":
            for kind in {p["kind"] for p in lane["genome"]}:
                kind_count[kind] += 1
    losers = []
    for c in reversed(order):
        if len(losers) == k:
            break
        if c in parents:
            continue
        lane_kinds = {p["kind"] for p in lanes[c]["genome"]}  # (이름 주의: 인자 kinds 와 다름)
        if min_kind_lanes and any(kind_count[kind] - 1 < min_kind_lanes for kind in lane_kinds):
            continue
        losers.append(c)
        for kind in lane_kinds:
            kind_count[kind] -= 1
    def over_cap(genome):
        return bool(max_kind_lanes) and any(kind_count[kind] + 1 > max_kind_lanes for kind in {p["kind"] for p in genome})

    for i, col in enumerate(losers):
        if i < num_new:
            child, parent = random_genome(rng, kinds=kinds), None
        else:
            parent = parents[(i - num_new) % k]
            child = mutate_genome(lanes[parent]["genome"], rng, hard_scale, kinds)
        tries = 0
        while over_cap(child) and tries < 20:  # 상한을 넘기면 새 무작위 유전자로
            child, parent, tries = random_genome(rng, kinds=kinds), None, tries + 1
        for kind in {p["kind"] for p in child}:
            kind_count[kind] += 1
        lanes[col] = {"type": "procedural", "genome": child, "born": gen, "parent": parent}
    new["generation"] = gen
    new["history"].append(
        {"generation": gen, "parents": parents, "replaced": losers, "hard_scale": round(hard_scale, 3),
         "lane_scores": [round(float(x), 3) for x in lane_scores]}
    )
    return new


class LevelBankTerrainGenerator(TerrainGenerator):
    """줄(lane)마다 지형이 적힌 "지형 은행"대로 지형을 만든다. (C17)

    - cfg.bank_in 이 없으면 0세대 (c17_default_bank)
    - cfg.scores_in (이전 세대 PLR 점수 npz) 가 있으면 evolve_bank() 로 한 세대 진화시킨 뒤 만든다
    - 최종 은행은 cfg.bank_out 에 저장하고, 출제자가 읽을 수 있게 cfg.bank_resolved 에도 넣어 둔다
    """

    def __init__(self, cfg, device: str = "cpu"):
        bank = json.load(open(cfg.bank_in)) if cfg.bank_in else c17_default_bank(cfg, cfg.evolve_seed, cfg.primitive_kinds)
        if cfg.scores_in:
            lane_scores = np.load(cfg.scores_in)["score"].mean(axis=0)  # (출발 행, 열) -> 줄별 평균
            rng = np.random.default_rng(cfg.evolve_seed + 1000 * (bank["generation"] + 1))
            bank = evolve_bank(
                bank, lane_scores, rng, cfg.evolve_num_replace, cfg.evolve_num_new, cfg.evolve_min_kind_lanes,
                cfg.evolve_max_kind_lanes, cfg.evolve_hard_growth, cfg.evolve_hard_growth_max, cfg.primitive_kinds,
                cfg.evolve_num_anchor,
            )
        assert len(bank["lanes"]) == cfg.num_cols, f"은행 줄 수 {len(bank['lanes'])} != num_cols {cfg.num_cols}"
        if cfg.bank_out:
            with open(cfg.bank_out, "w") as f:
                json.dump(bank, f, indent=1)
        cfg.bank_resolved = bank
        self.bank = bank
        super().__init__(cfg, device)

    def _generate_curriculum_terrains(self):
        """원래 규칙(열마다 지형 1종, 행이 클수록 어려움)과 같되, 각 줄의 지형을 은행에서 읽는다."""
        lower, upper = self.cfg.difficulty_range
        for col, lane in enumerate(self.bank["lanes"]):
            if lane["type"] == "procedural":
                sub_cfg = self.cfg.sub_terrains["procedural"].replace(genome=lane["genome"])
            else:
                sub_cfg = self.cfg.sub_terrains[lane["type"]]
            for row in range(self.cfg.num_rows):
                difficulty = lower + (upper - lower) * (row + self.np_rng.uniform()) / self.cfg.num_rows
                mesh, origin = self._get_terrain_mesh(difficulty, sub_cfg)
                self._add_sub_terrain(mesh, origin, row, col, sub_cfg)


@configclass
class LevelBankTerrainGeneratorCfg(TerrainGeneratorCfg):
    """C17 지형 은행 설정. 세대 반복 스크립트가 bank_in / scores_in / bank_out 을 Hydra 로 넘긴다."""

    class_type: type = LevelBankTerrainGenerator
    curriculum: bool = True  # 행이 클수록 어려움 (반드시 True)
    # 경로 3개는 기본값이 빈 문자열("") = 설정 안 함.
    bank_in: str = ""  # 이전 세대 은행 json (없으면 0세대)
    scores_in: str = ""  # 이전 세대 PLR 점수 npz (있으면 한 세대 진화)
    bank_out: str = ""  # 이번 세대 은행을 저장할 경로
    evolve_seed: int = 0
    evolve_num_replace: int = 4  # 세대마다 교체할 줄 수
    evolve_num_new: int = 1  # 그중 완전히 새 유전자로 채울 줄 수 (C17b: 2)
    evolve_min_kind_lanes: int = 0  # 도형 종류마다 최소 몇 줄 남길지 (C17: 0 = 제한 없음, C17b: 1)
    evolve_max_kind_lanes: int = 0  # 한 도형 종류를 가진 줄의 최대 수 (C17: 0 = 제한 없음, C17b: 8)
    evolve_hard_growth: float = 0.0  # (C30) 세대마다 설정값 상한을 키우는 비율 (0.01 = 세대당 +1%)
    evolve_hard_growth_max: float = 0.5  # 상한 확대의 최대치 (원래 상한의 1.5 배)
    primitive_kinds: list | None = None  # (C33) 쓸 도형 종류 목록. None = PROC_PRIMITIVES 전부 (11종). C17 계열은 원래 6종
    evolve_num_anchor: int = 0  # (C34) 앞쪽 절차적 줄 이만큼은 절대 교체하지 않는 "닻" (0 = 없음)
    bank_resolved: dict | None = None  # (자동) 생성기가 채움 -> 출제자가 줄 종류를 읽음
    library_lanes: list | None = None  # (19단계) Isaac Lab 기본 지형 줄 목록 (sub_terrains 의 키). 절차적 줄 대신 들어가며 진화 대상이 아님


# C17 은행: 크기·행·열은 C7 코스와 같게. 줄 종류는 절차적 지형 + 평지
ANT_C17_BANK_CFG = LevelBankTerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=1.0,
    num_rows=24,
    num_cols=20,
    # 시뮬레이션 시작 순간 PhysX 가 "malloc(): corrupted top size" 로 죽음 (한 줄씩은 정상, 4줄 함께면 죽음).
    # 격자 0.2 m(삼각형 1/4) 또는 수직벽 보정 끄기로 회피됨 -> 평가 지형처럼 수직 벽은 유지하려고 격자를 키움.
    # 도형 크기는 전부 0.25 m 이상이라 모양은 거의 같다.
    horizontal_scale=0.2,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    primitive_kinds=list(PROC_KINDS_V1),  # (16단계 추가) 도형 5종이 늘어난 뒤에도 C17 계열은 원래 6종만
    sub_terrains={
        "procedural": ProceduralTerrainCfg(proportion=0.8, border_width=0.0),
        "flat": HfPyramidSlopedTerrainCfg(proportion=0.2, slope_range=(0.0, 0.0), platform_width=2.0, border_width=0.25),
    },
)


# =====================================================================
#   - 진화 규칙은 C27 과 같이 세대마다 8줄 교체(4줄 완전히 새 유전자), 종류당 최소 1줄·최대 10줄 (tools/run_accel_v3.sh)
#   - 난이도 상한 확대(C30 hard_growth)는 쓰지 않음: C30 이 "안 넘어지지만 앞으로 안 가는" 걸음으로 무너짐
# =====================================================================
ANT_C33_BANK_CFG = ANT_C17_BANK_CFG.replace(num_cols=40, primitive_kinds=None)
# (C34) C33 + 닻 줄 8개: 절차적 36줄 중 앞 8줄은 0세대 지형 그대로 끝까지 유지 (기본 걸음을 잊지 않게)
ANT_C34_BANK_CFG = ANT_C33_BANK_CFG.replace(evolve_num_anchor=8)

# =====================================================================
#   - 채점 쪽이 지형을 바꿀 때 가장 쓸 법한 것이 Isaac Lab 기본 지형(T1 = 계단·역계단·상자·요철·경사·역경사, T2 = 요철·물결·네모 장애물)
#   - 학습에 넣는 7종: 계단 위, 계단 아래(얕게), 무작위 격자 상자, 네모 장애물, 물결, 경사 위, 경사 아래(얕게)
#     평가 전용으로 남기는 기본 지형: 징검돌·틈·별·물체 뿌리기(이미 시험장) + 구덩이·상자 단·레일(시험장 새로 추가)
#   - 줄 종류 이름 = sub_terrains 키. 은행 json 에는 {"type": "lib_stairs_up"} 처럼 적힘. 진화(evolve_bank)는 절차적 줄만 건드림
#   - 깊이 규칙: 학습 넘어짐 판정은 바닥 기준이지만 채점은 월드 높이 0.31 m 기준이라, 아래로 파이는 지형은 -0.15 m 안쪽으로
#     (T1 의 역계단 -1.4 m 는 채점 판정으로는 내려가는 즉시 탈락 = 배워도 소용없음)
#   - 높이맵 지형은 은행 격자 0.2 m 로 그려짐 (평가 지형은 0.1 m)
# =====================================================================
LIBRARY_LANES = {
    "lib_stairs_up": MeshPyramidStairsTerrainCfg(proportion=0.0, step_height_range=(0.03, 0.15), step_width=0.5, platform_width=2.5, border_width=0.5),  # 계단 4~5개 x 최대 0.15 = 0.75 m
    "lib_stairs_down": MeshInvertedPyramidStairsTerrainCfg(proportion=0.0, step_height_range=(0.02, 0.035), step_width=1.0, platform_width=2.0, border_width=0.5),  # 계단 4개 x 최대 0.035 = 깊이 0.14 m (채점 판정 0.31 m 안전. CPU 시험으로 확인)
    "lib_boxes": MeshRandomGridTerrainCfg(proportion=0.0, grid_width=0.45, grid_height_range=(0.03, 0.12), platform_width=1.0),
    "lib_obstacles": HfDiscreteObstaclesTerrainCfg(proportion=0.0, obstacle_height_mode="choice", obstacle_width_range=(0.4, 1.2), obstacle_height_range=(0.05, 0.15), num_obstacles=25, platform_width=1.0, border_width=0.25),  # choice 모드는 위·아래(-h) 장애물 섞임 -> 깊이 -0.15 m 까지
    "lib_wave": HfWaveTerrainCfg(proportion=0.0, amplitude_range=(0.03, 0.15), num_waves=4, border_width=0.25),
    "lib_slope_up": HfPyramidSlopedTerrainCfg(proportion=0.0, slope_range=(0.05, 0.30), platform_width=2.0, border_width=0.25),
    "lib_slope_down": HfInvertedPyramidSlopedTerrainCfg(proportion=0.0, slope_range=(0.01, 0.05), platform_width=2.0, border_width=0.25),
}
ANT_C38_BANK_CFG = ANT_C33_BANK_CFG.replace(
    library_lanes=list(LIBRARY_LANES),
    sub_terrains={**ANT_C33_BANK_CFG.sub_terrains, **LIBRARY_LANES},
)


LIBRARY_LANES_HARD = {
    "libh_stairs_up": MeshPyramidStairsTerrainCfg(proportion=0.0, step_height_range=(0.10, 0.25), step_width=0.40, platform_width=2.0, border_width=0.25, holes=False),
    "libh_boxes": MeshRandomGridTerrainCfg(proportion=0.0, grid_width=0.45, grid_height_range=(0.08, 0.20), platform_width=1.0),
    "libh_obstacles": HfDiscreteObstaclesTerrainCfg(proportion=0.0, obstacle_height_mode="choice", obstacle_width_range=(0.4, 1.2), obstacle_height_range=(0.10, 0.30), num_obstacles=10, platform_width=1.0, border_width=0.25),
    "libh_wave": HfWaveTerrainCfg(proportion=0.0, amplitude_range=(0.10, 0.25), num_waves=5, border_width=0.25),
    "libh_slope_up": HfPyramidSlopedTerrainCfg(proportion=0.0, slope_range=(0.30, 0.60), platform_width=2.0, border_width=0.25),
    "libh_pit": MeshPitTerrainCfg(proportion=0.0, pit_depth_range=(0.05, 0.25), platform_width=2.0, double_pit=True),
    "libh_gap": MeshGapTerrainCfg(proportion=0.0, gap_width_range=(0.10, 0.50), platform_width=2.0),
    "libh_rails": MeshRailsTerrainCfg(proportion=0.0, rail_thickness_range=(0.05, 0.10), rail_height_range=(0.05, 0.20), platform_width=2.0),
    "libh_stepping": HfSteppingStonesTerrainCfg(proportion=0.0, stone_height_max=0.15, stone_width_range=(0.4, 1.0), stone_distance_range=(0.1, 0.5), holes_depth=-1.0, platform_width=2.0, border_width=0.25),
}
# 어려운 은행 = C38 은행(생성기 + 보통 기본 지형 7줄) + 어려운 기본 지형 9줄, 줄 안 난이도 0.4 → 1 (출발 타일부터 진폭 58 %)
ANT_HARD_BANK_CFG = ANT_C38_BANK_CFG.replace(
    library_lanes=list(LIBRARY_LANES) + list(LIBRARY_LANES_HARD),
    sub_terrains={**ANT_C38_BANK_CFG.sub_terrains, **LIBRARY_LANES_HARD},
    difficulty_range=(0.4, 1.0),
)

import os
import isaaclab.envs.mdp as base_mdp
import isaaclab_tasks.manager_based.classic.humanoid.mdp as ant_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab_assets.robots.ant import ANT_CFG
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import (
    AntEnvCfg,
    EventCfg,
    MySceneCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)
from isaaclab_tasks.manager_based.classic.ant_robust.eval_env_cfg import (
    ANT_ROUGH_TERRAINS_CFG,
    BATTERY,
    use_battery_terrain,
)
from isaaclab_tasks.manager_based.classic.ant_robust.mdp import (  # 구르기 v15 계보·평가가 쓰는 것만 (깃허브용으로 추림)
    PrioritizedTerrainReplay,
    height_scan_safe,
    print_collider_offsets,
    randomize_robot_material_per_env,
    roll_forward_gated,
    roll_streak,
    roll_rate_target,
    roll_wobble_l2,
    wheel_posture,
    widen_joint_limits,
    roll_on_ground,
    height_above,
    vertical_speed_l2,
    lateral_speed_l1,
    legs_straight,
    foot_slip_speed,
    reached_course_end,
    TeamMapTerrainImporterCfg,
    torso_above,
    torso_below_ground_relative_height,
)
from isaaclab_tasks.manager_based.classic.ant_robust.train_env_cfg import (
    ANT_C33_BANK_CFG,
    ANT_C38_BANK_CFG,
    ANT_HARD_BANK_CFG,
)


# ---------------------------------------------------------------------
# 1) 장면: 원본(바닥 + Ant + 조명) + 높이 센서 하나
# ---------------------------------------------------------------------
@configclass
class AntFullSceneCfg(MySceneCfg):
    """원본 MySceneCfg 에 height_scanner 만 추가 (terrain / robot / light 는 그대로 물려받음)."""

    height_scanner = RayCasterCfg(
        # (a) 어디에 붙나: Ant 의 몸통 바디 "torso" (여기에 물리 API 가 붙어 있어 위치를 정확히 읽는다).
        #     /Robot 같은 상위 프림에 붙이면 경고만 뜨고 "움직이지 않는 스캔"이 되어 조용히 틀린다.
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        # (b) 광선 출발점을 몸통보다 20 m 위로 올린 뒤 아래로 쏜다 (지형이 높아도 항상 위에서 내려다봄)
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        # (c) 몸통이 기울거나 뒤집혀도 광선은 항상 "월드 수직 아래". 좌우 회전(yaw)만 따라간다.
        #     기본값 "base" 로 두면 뒤집혔을 때 광선이 하늘을 향해 전부 빗나간다.
        ray_alignment="yaw",
        # (d) 광선 격자: 0.2 m 간격, 1.6 m x 1.6 m -> 9 x 9 = 81개
        #     Ant 발끝이 몸통 중심에서 x, y 로 최대 ±0.66 m -> ±0.8 m 면 네 발 디딜 자리를 모두 덮는다.
        #     0.2 m 간격은 코스 특징(계단 0.3, 박스 0.45, 블록 0.4~0.8)보다 촘촘하고,
        #     T2 자갈(0.1 m 잡음)보다 성글어서 잡음에 휘둘리지 않는다.
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=(1.6, 1.6)),
        # (e) 어느 메시에 쏘나: 바닥 프림 (원본 TerrainImporterCfg.prim_path 와 같아야 함)
        #     terrain_type="plane" 이어도 Isaac Lab 이 무한 평면 메시를 만들어 주므로 평지에서도 동작한다.
        mesh_prim_paths=["/World/ground"],
        # (f) GUI 에서 광선 맞은 점 보기. 학습 때는 반드시 False (명령어로 잠깐 켤 수 있음)
        debug_vis=False,
    )


# ---------------------------------------------------------------------
# 2) 관측: 원본 11개 항목(60차원) 뒤에 height_scan 81차원을 "덧붙인다"
#    (부모 항목이 먼저, 자식에서 새로 만든 항목이 뒤 -> 원본 60개의 순서는 그대로 보존)
# ---------------------------------------------------------------------
@configclass
class AntFullObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        height_scan = ObsTerm(
            func=height_scan_safe,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                # 평지에 서 있을 때 값이 0 근처가 되도록 몸통 기준 높이 0.5 m 를 뺀다
                "offset": 0.5,
                "clip_range": (-1.0, 1.0),
            },
            # 이중 안전장치 (함수 안에서 이미 잘랐지만 한 번 더)
            clip=(-1.0, 1.0),
        )

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------
# 3) [제출용 + 평지 평가용] 원본 바닥 그대로 + 높이 센서
#    scene.terrain 을 절대 건드리지 않는다 -> 채점용 지형이 그대로 적용된다
# ---------------------------------------------------------------------
@configclass
class AntFullEnvCfg(AntEnvCfg):
    """원본 Ant 환경 + 높이 센서. 바닥은 AntEnvCfg 가 정한 것을 그대로 쓴다."""

    scene: AntFullSceneCfg = AntFullSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
    observations: AntFullObservationsCfg = AntFullObservationsCfg()

    def __post_init__(self):
        # 1) 원본 Ant 설정을 먼저 전부 적용 (보상, 종료 0.31 m, 마찰, 16초, dt)
        super().__post_init__()
        # 2) 센서와 fabric 복제는 같이 못 쓴다. 반드시 False
        self.scene.clone_in_fabric = False
        # 3) 센서 갱신 주기 = 제어 주기 (2 x 1/120 = 1/60 초). 매 step 새 값을 읽는다
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        # 4) PhysX 접촉 버퍼를 기본값의 2배로 (큰 메시 지형을 넣어도 버퍼가 안 터지게)
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


# ---------------------------------------------------------------------
# 6-0) 바닥 바꾸기 도우미 (같은 세 줄을 클래스마다 반복하지 않으려고)
# ---------------------------------------------------------------------
def use_T1_terrain(cfg):
    """T1 = Isaac Lab 기본 ROUGH 지형 (위 AntStockRoughFullEvalEnvCfg 와 100% 같은 지형)."""
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
    cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.55, 0.45))


def use_T2_terrain(cfg):
    """T2 = 개미 크기 울퉁불퉁 지형, 0번 행 가운데 8열에서 출발 (위 AntRoughFullEvalEnvCfg 와 100% 같음)."""
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = ANT_ROUGH_TERRAINS_CFG
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.60, 0.50, 0.40))


def use_course(cfg, course):
    """학습 코스로 바꾸기. 맨 뒤 0~1행에서만 출발 (C2 / C5 와 같은 방식)."""
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = course
    cfg.scene.terrain.max_init_terrain_level = 1
    cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.40, 0.50, 0.65))


# ---------------------------------------------------------------------
# 6-1) C6 센서: 앞쪽으로 길게  (C5 센서와 다른 곳은 offset 의 x 와 격자 크기 두 가지뿐)
#   - Ant 는 초속 약 8.8 m 로 달린다. C5 센서는 몸 앞 0.8 m 까지만 봐서 약 0.09 초 앞밖에 못 봤다.
#   - C6 는 몸 앞 3.0 m 까지 본다 -> 약 0.34 초 앞. 좌우는 네 발 디딜 폭(±0.66 m)만큼 ±0.8 m 유지.
#   - 격자 중심을 몸통 앞 1.1 m 로 옮김 -> 앞뒤 범위 = 1.1 ± 1.9 = 뒤 0.8 m ~ 앞 3.0 m
#   - ray_alignment="yaw" 라서 Ant 가 방향을 틀면 "앞"도 같이 돈다 (ray_caster.py 220, 281 줄)
# ---------------------------------------------------------------------
@configclass
class AntC6SceneCfg(AntFullSceneCfg):
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(1.1, 0.0, 20.0)),  # C5: (0, 0, 20)
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=(3.8, 1.6)),  # 20 x 9 = 180개  (C5: 9 x 9 = 81개)
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


# ---------------------------------------------------------------------
# 6-2) C6 가족 평가 태스크 (C6, C7, C9, C10 모델용). 관측 = 60 + 180 = 240
#   관측 함수는 C5 것(AntFullObservationsCfg)을 그대로 쓴다. 센서 이름이 같아서 광선 수만 자동으로 바뀐다.
# ---------------------------------------------------------------------
@configclass
class AntC6FullEnvCfg(AntFullEnvCfg):
    """[C6 가족 제출용 + 평지 평가] 원본 바닥 그대로(scene.terrain 안 건드림) + 앞쪽 긴 센서."""

    scene: AntC6SceneCfg = AntC6SceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)


# ---------------------------------------------------------------------
# 8-1) 바닥 기준 넘어짐 판정 (학습용). 원본 두 판정(time_out, torso_height)은 그대로 + 하나 추가
# ---------------------------------------------------------------------
@configclass
class AntGroundAwareTerminationsCfg(TerminationsCfg):
    torso_height_ground = DoneTerm(
        func=torso_below_ground_relative_height,
        params={"minimum_height": 0.31, "radius": 0.25, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )


@configclass
class AntPLRCurriculumCfg:
    terrain_teacher = CurrTerm(
        func=PrioritizedTerrainReplay,
        params={
            "max_start_row": 7,  # 0~7행에서만 출발 (7행에서도 앞에 136 m)
            "beta": 1.0,  # 순위 온도 (칸 160개에 맞춰 1.0)
            "rho": 0.3,  # staleness 비율 (PLR / Robust PLR 의 MiniGrid 값)
            "save_every_steps": 3200,  # 100회 학습마다 칸별 점수 저장 (학습폴더/teacher/)
        },
    )


# #####################################################################
#
#    contactOffset = 0.02 m 가 PhysX 에 적용되지 않는다. 실제로 들어간 값(실행 중 PhysX 에서 읽음):
#       확실한 것: 시작할 때 0.02 를 명시하면 일반 ant.usd 와 소수점까지 같은 결과(88.080201)가 나온다 = 설계값이 '동작상' 빠져 있음
#    지형이 평평한 메시면 지형 쪽 여유도 거의 0 -> 발이 박힌 뒤에야 접촉 -> 튕겨서 뒤집힘.
#    C5, 평평한 T2: 43.8점 -> 로봇 접촉 여유 0.02 로: 88.1점.
#  - K 가족 = 원본 Isaac-Ant-v0 (관측 60, 보상 7, 종료조건 0.31 m, PPO) 와 완전히 같고,
#    "시작할 때 로봇 충돌 모양 13개의 contact offset 을 0.02 로 설정" 하는 이벤트 하나만 다르다.
#    (Isaac Lab 기본 이벤트 randomize_rigid_body_collider_offsets 를 범위 [0.02, 0.02] 로 사용 = 고정값 설정)
#  - 지형 설정을 전혀 건드리지 않으므로 채점 때 지형만 바꿔도 그대로 적용된다. (로봇 물리 설정이라 채점 확인은 필요)
# #####################################################################
ANT_DESIGN_CONTACT_OFFSET = 0.02  # [m] Ant 파일(ant.usd)의 physxCollision:contactOffset 값


@configclass
class AntKEventCfg(EventCfg):
    """원본 이벤트 2개 + 로봇 접촉 여유를 설계값으로 (시작할 때 한 번)."""

    robot_contact_offset = EventTerm(
        func=base_mdp.randomize_rigid_body_collider_offsets,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_offset_distribution_params": (ANT_DESIGN_CONTACT_OFFSET, ANT_DESIGN_CONTACT_OFFSET),
        },
    )
    offsets_check = EventTerm(  # 로그에 "[COLLIDER K]" 로 실제 값 확인
        func=print_collider_offsets, mode="startup", params={"asset_cfg": SceneEntityCfg("robot"), "tag": "K"}
    )


# #####################################################################
#
#      C5  T2 39.7 -> 86.3 / C6 T2 48.5 -> 84.6 / C5 평면 99.8 -> 108.1
#    반면 평면에서만 학습한 baseline 은 오히려 나빠짐(평평한 T2 55.5 -> 23.7): 메시 접촉을 배운 적이 없어서.
#    -> 메시 코스에서 학습하는 C6 에 이 한 줄을 넣고 "다시 학습" 한 것이 C22.
#  - 접촉 여유 값은 이벤트 파라미터로: 기본 0.02 (Ant 파일 설계값). 0.04 는 명령어로
#       env.events.robot_contact_offset.params.contact_offset_distribution_params=[0.04,0.04]
#  - 보상 / 종료조건(0.31 m) / 관측(240) / 지형 설정은 C6 와 같다.
# #####################################################################
@configclass
class AntQFullEnvCfg(AntC6FullEnvCfg):
    """[Q 가족 제출용 + 평지 평가] C6 가족 + 로봇 접촉 여유 설계값."""

    events: AntKEventCfg = AntKEventCfg()


# #####################################################################
#   = C6 관측(원본 60 + 앞쪽 긴 높이 센서 180 = 240) + 로봇 접촉 여유 0.04 m
#   - 채점는 우리 폴더에서 "지형 부분만" 바꿔 채점한다 -> 이 태스크는 scene.terrain 을 전혀 건드리지 않는다.
#   - 보상 7개 / 종료조건(몸통 월드 z < 0.31 m) / 행동 / PPO 설정은 원본 Isaac-Ant-v0 그대로.
#   - 원본과 다른 점 (README·발표에 명시):
#       (1) 관측: 앞쪽 긴 높이 센서 180 개 추가 (허용 범위의 관찰 차원 변경)
#       (2) 물리: 로봇 충돌 모양의 접촉 여유(contact offset) 를 0.04 m 로 설정
#           (원본 인스턴스 Ant 에는 Ant 파일의 설계값 0.02 가 적용되지 않음 -> 메시 지형에서 발이 박혀 튕겨 뒤집힘)
#   - Q 가족 태스크 + 명령어 오버라이드 [0.04, 0.04] 와 완전히 같은 설정 (명령어에 의존하지 않게 코드에 고정)
# #####################################################################
FINAL_CONTACT_OFFSET = 0.04  # [m]


@configclass
class AntFinalEnvCfg(AntQFullEnvCfg):
    """[최종 제출] 원본 바닥 그대로(채점 때 지형만 바뀜) + 센서 + 접촉 여유 0.04."""

    def __post_init__(self):
        super().__post_init__()
        self.events.robot_contact_offset.params["contact_offset_distribution_params"] = (
            FINAL_CONTACT_OFFSET,
            FINAL_CONTACT_OFFSET,
        )


@configclass
class AntFinalStockRoughEvalEnvCfg(AntFinalEnvCfg):  # 우리 평가용 T1
    def __post_init__(self):
        super().__post_init__()
        use_T1_terrain(self)


@configclass
class AntFinalRoughEvalEnvCfg(AntFinalEnvCfg):  # 우리 평가용 T2
    def __post_init__(self):
        super().__post_init__()
        use_T2_terrain(self)


# #####################################################################
# 15단계: 단일 지형 시험장 태스크 (지형 한 종류씩만. 시험장 목록은 eval_env_cfg.py 15단계의 BATTERY 표)
#   - 시험장마다 두 가족을 만든다:
#       Base  = 원본 Isaac-Ant-v0 (관측 60개, 원본 물리)  -> 기준 모델(C0) 평가용
#       Final = 최종 제출 설정 AntFinalEnvCfg (센서 + 접촉 여유 0.04) -> C22b, C28 평가용
#   - 바닥만 바뀌고 보상 7개 / 종료조건 / 에피소드 길이는 원본 그대로.
#   - 클래스를 시험장 수 x 2 개 손으로 쓰지 않으려고 아래 반복문으로 만든다.
#     만들어진 이름 예: AntBatteryBase_gravel, AntBatteryFinal_gravel (태스크 등록은 __init__.py 15단계)
# #####################################################################
def _battery_env_cfg(base_cls, name, family):
    """base_cls 설정에서 바닥만 시험장 name 으로 바꾼 설정 클래스를 만든다."""

    @configclass
    class BatteryEnvCfg(base_cls):
        def __post_init__(self):
            super().__post_init__()
            use_battery_terrain(self, name)

    BatteryEnvCfg.__name__ = BatteryEnvCfg.__qualname__ = f"AntBattery{family}_{name}"
    return BatteryEnvCfg


for _name in BATTERY:
    for _family, _base_cls in (("Base", AntEnvCfg), ("Final", AntFinalEnvCfg),):
        _cls = _battery_env_cfg(_base_cls, _name, _family)
        globals()[_cls.__name__] = _cls


# #####################################################################
#   - 왜: 단일 지형 시험장(15단계)에서 C22b 가 바닥 재질 시험에 약했다 (seed 42)
#       빙판 0.0 (기준 모델 14.0) / 미끄러지면 쭉 11.1 (37.5) / 통통 튀는 바닥 47.0 (82.1)
#     과제 조건이 "처음 보는 지형과 마찰" 이라서, 학습 때 마찰·반발을 넓게 겪게 한다.
#   - 무엇을 (학습 때만. 평가·제출 태스크 Isaac-Ant-Final-v0 은 그대로):
#       (1) 환경마다 로봇 재질 하나를 뽑아 몸 전체에: 마찰 0.05 ~ 1.5 (동마찰 <= 정마찰), 반발 0 ~ 0.8
#           재질 후보 256개 중 25 % 는 원래 값(1.0 / 1.0 / 0) -> 평소 바닥도 계속 겪음 (mdp.py 16단계)
#       (2) 땅 재질의 섞는 방식을 마찰 "multiply", 반발 "max" 로
#           -> PhysX 는 두 재질의 섞는 방식 중 순위가 높은 것을 쓴다 (average < min < multiply < max)
#           -> 실제 마찰 = 로봇 값 x 땅 1.0 = 로봇 값 (average 면 (로봇 + 1.0) / 2 라서 0.53 밑으로 못 내려감)
#           -> 실제 반발 = max(로봇 값, 땅 0) = 로봇 값
#   - 나머지(센서, 접촉 여유 0.04, C2 코스, 1000회)는 C22b 와 같다.
#   - 평가: 제출 태스크(Final 가족) + 단일 지형 시험장 Final 가족
# #####################################################################
@configclass
class AntC32EventCfg(AntKEventCfg):
    """K 가족 이벤트(접촉 여유) + 환경마다 로봇 재질 무작위."""

    robot_material = EventTerm(
        func=randomize_robot_material_per_env,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.05, 1.5),
            "dynamic_friction_range": (0.05, 1.5),
            "restitution_range": (0.0, 0.8),
            "num_buckets": 256,
            "make_consistent": True,  # 동마찰 <= 정마찰 ("한번 미끄러지면 쭉" 같은 바닥도 겪음)
            "nominal_fraction": 0.25,
        },
    )


# #####################################################################
#                  성능이 수렴하면 거기서 멈춰라."
#   - = AntFinalEnvCfg (센서 240 + 접촉 여유 0.04) + ANT_C33_BANK_CFG (도형 11종, 줄 40개) + AntPLRCurriculumCfg (칸별 점수 -> 진화)
#   - 세대 반복·수렴 판정: tools/run_accel_v3.sh (5세대마다 T2 + 처음 보는 시험장 4종 평가, 20세대 동안 최고점 갱신 없으면 멈춤)
#   - 평가·제출은 Final 가족 (접촉 여유 0.04 고정). 학습 지형에 넣지 않은 시험장(가시밭·허들·별 빔·징검돌·틈·침목·배수 격자·톱니·재질 6종)이 진짜 처음 보는 시험
#       (1) evolve_bank 가 상위 부모를 건너뛰던 인덱스 버그 수정
#       (2) 출제자 점수를 "후회 근사"(= 사실상 난이도) 대신 "배울 수 있음"(score_mode="learnability")으로: 반쯤 되는 칸을 부모로
#   - 넘어짐 판정은 학습 때만 "바닥 기준"(C12/C14 의 AntGroundAwareTerminationsCfg): 언덕 위에서 뒤집혀도 탈락하게 해서
#     "언덕 = 안 넘어지는 지형"이라는 착시로 언덕만 늘어나는 것을 막음. 평가·제출은 원본 판정 그대로
# #####################################################################
@configclass
class AntC33TrainEnvCfg(AntFinalEnvCfg):
    """C22b 물리·센서 + 극단적 다양화 지형 은행 + 배울-수-있음 출제자 + 바닥 기준 판정(학습만)."""

    curriculum: AntPLRCurriculumCfg = AntPLRCurriculumCfg()
    terminations: AntGroundAwareTerminationsCfg = AntGroundAwareTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        use_course(self, ANT_C33_BANK_CFG)
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.60, 0.42, 0.55))
        self.curriculum.terrain_teacher.params["save_every_steps"] = 1600  # 세대가 짧으니 50회마다 칸별 점수 저장
        self.curriculum.terrain_teacher.params["score_mode"] = "learnability"


# #####################################################################
#   - 은행 40줄 = 절차적 29 + 기본 지형 7 + 평지 4. 진화 없이(NO_EVOLVE) 제출 후보(c36ctrl 60세대)에서 이어서 학습: tools/run_queue_36.sh
# #####################################################################
@configclass
class AntC38TrainEnvCfg(AntC33TrainEnvCfg):
    """C33 + Isaac Lab 기본 지형 줄 7개 (Isaac-Ant-Robust-C38-v0)."""

    def __post_init__(self):
        super().__post_init__()
        use_course(self, ANT_C38_BANK_CFG)


@configclass
class AntFunRollRewardsCfg(RewardsCfg):
    roll = RewTerm(func=roll_forward_gated, weight=1.0, params={"max_rate": 15.0, "max_speed": 6.0})
    upright = None
    move_to_target = None
    energy = RewTerm(func=ant_mdp.power_consumption, weight=-0.01, params={"gear_ratio": {".*": 15.0}})


@configclass
class AntFunRoll3RewardsCfg(AntFunRollRewardsCfg):
    """구르기 보상 + 연속성 보상 3개 (공 굴러가듯 끊기지 않게).
    v2 측정: 구르는 시간 55~67 %, 마리당 구르기 구간 약 20번, 가장 긴 연속 2~3초 = 튕기며 뒤집기. 그래서
    - streak: 연속으로 구른 걸음 수에 비례 (2초 이상이면 1.0, 멈추면 0 으로 리셋), weight 2.0
    - rate: 회전 속도가 8 rad/s(1.3바퀴/초) 근처면 1, weight 1.0
    - wobble: 구르는 축이 아닌 x·z 축 흔들림 벌점, weight −0.005 (각속도 5 rad/s 면 걸음당 −0.125)
    - 전진 보상은 1.0 → 0.3 (튕기며 멀리 가는 것보다 계속 구르는 쪽이 유리하게)"""

    progress = RewTerm(func=ant_mdp.progress_reward, weight=0.3, params={"target_pos": (1000.0, 0.0, 0.0)})
    streak = RewTerm(func=roll_streak, weight=2.0, params={"min_rate": 3.0, "hold_steps": 120})
    rate = RewTerm(func=roll_rate_target, weight=1.0, params={"target": 8.0, "sigma": 3.0})
    wobble = RewTerm(func=roll_wobble_l2, weight=-0.005)


# #####################################################################
#   바탕 = C33 학습 설정 (무작위 지형 은행 40줄, 출제자, 바닥 기준 판정, 접촉 여유 0.04). 1,000회 학습해 c33 0세대(원본 보상)와 비교
#   a) 자세 안정(수업 힌트 "발이 걸려도 몸이 기울지 않게"): 몸통 기울기 벌점(flat_orientation_l2 -2.0, 중력 방향의 좌우·앞뒤 성분 제곱 = 20° 기울면 0.12)
#      + 몸통 좌우·앞뒤 회전 속도 벌점(ang_vel_xy_l2 -0.05) + 행동 변화 벌점(action_rate_l2 -0.1)
#   b) 안 걸리게: 몸통·다리가 땅에 닿으면 벌점(undesired_contacts -1.0) + 발 들어 올리기 보상(feet_air_time +2.0, 0.2초 초과분)
#   c) 넘어짐 벌점: 종료(넘어짐) 순간 -600 (dt 곱하면 에피소드 점수에서 -10 = 약 10%)
#   d) 전부: a + b + c
#   접촉 센서가 필요해서 장면에 몸 전체 접촉 센서를 붙임 (관측에는 안 들어감 -> 정책 입력은 C33 과 같음)
# #####################################################################
@configclass
class AntC39SceneCfg(AntC6SceneCfg):
    robot = ANT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=ANT_CFG.spawn.replace(activate_contact_sensors=True),
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)


@configclass
class AntFunRoll7RewardsCfg(AntFunRoll3RewardsCfg):
    """v4 연속 구르기 보상에서 (1) 구르기 점수는 몸이 땅에 닿아 있을 때만, (2) 바닥에서 0.6 m 넘게 뜨면 벌점, (3) 위아래 속도·옆 속도 벌점,
    (4) 전진 보상 0.3 → 1.0 (앞으로 가야). 출발·탈락은 채점 규칙 그대로."""

    progress = RewTerm(func=ant_mdp.progress_reward, weight=1.0, params={"target_pos": (1000.0, 0.0, 0.0)})
    roll = RewTerm(func=roll_on_ground, weight=1.0, params={"max_rate": 15.0, "max_speed": 6.0, "threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces")})
    too_high = RewTerm(func=height_above, weight=-2.0, params={"height": 0.6, "sensor_cfg": SceneEntityCfg("height_scanner")})
    bounce = RewTerm(func=vertical_speed_l2, weight=-0.02)
    sideways = RewTerm(func=lateral_speed_l1, weight=-0.3)


# #####################################################################
#   - 채점 평가 = "기본 지형을 수정하고 마찰계수 같은 파라미터를 무작위로". 그래서 학습도 그렇게.
#   - 토크: make_strong() = 행동 배율 7.5 -> 20, 토크 상한 60 N·m, 관절 속도 한계 100 rad/s (v1 과 같음). 채점: 형태만 유지하면 됨
#   - 평가 가족 J200 (Final + 관절 ±100°), J200S (+ 토크 8배): 학습과 같은 관절·토크로 채점해야 의미 있음
# #####################################################################
JOINT200 = 1.745  # [rad] = 100°


@configclass
class AntFunJoint200MatEventCfg(AntC32EventCfg):
    """학습용: C32 재질 무작위(정·동마찰 0.05~1.5, 반발 0~0.8, 환경마다) + 관절 ±100°."""

    widen_joints = EventTerm(func=widen_joint_limits, mode="startup", params={"asset_cfg": SceneEntityCfg("robot"), "lower": -JOINT200, "upper": JOINT200})


@configclass
class AntFinalJ200EventCfg(AntKEventCfg):
    """평가용: Final 이벤트 + 관절 ±100° (재질 무작위 없음)."""

    widen_joints = EventTerm(func=widen_joint_limits, mode="startup", params={"asset_cfg": SceneEntityCfg("robot"), "lower": -JOINT200, "upper": JOINT200})


# ---- 학습 태스크 ----
@configclass
class AntFunRoll10EnvCfg(AntC38TrainEnvCfg):
    """[구르기 v10] C38 은행(우리 생성기 + 기본 지형 7줄) + 재질 무작위 + 관절 ±100° + 땅에서 구르기 보상(v8). 토크 원본. (Isaac-Ant-Fun-Roll10-v0)"""

    scene: AntC39SceneCfg = AntC39SceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
    events: AntFunJoint200MatEventCfg = AntFunJoint200MatEventCfg()
    rewards: AntFunRoll7RewardsCfg = AntFunRoll7RewardsCfg()


# ---- 평가 가족 J200 / J200S (평지·T1·T2 + 시험장 35종) ----
@configclass
class AntFinalJ200EnvCfg(AntFinalEnvCfg):
    events: AntFinalJ200EventCfg = AntFinalJ200EventCfg()


@configclass
class AntFinalJ200RoughEvalEnvCfg(AntFinalRoughEvalEnvCfg):
    events: AntFinalJ200EventCfg = AntFinalJ200EventCfg()


@configclass
class AntFinalJ200StockRoughEvalEnvCfg(AntFinalStockRoughEvalEnvCfg):
    events: AntFinalJ200EventCfg = AntFinalJ200EventCfg()


# #####################################################################
#   - 앤트 자산은 enabled_self_collisions=False 라 관절을 넓히면 다리가 몸통·다른 다리를 통과한다 -> enable_self_collision() 으로 켠다 (형태는 그대로, 물리 설정)
#   - 예쁜 풍차 = 다리를 곧게 뻗은 X 자(발목 0°, 원본 범위 30~100° 에는 0° 가 없음) + 몸통 평면 수직(바퀴 자세) + 다리 끝으로 굴러 몸통이 높음
#   - 평가 가족 J200C = J200(관절 ±100°) + 자기 충돌 켬. 학습과 같은 물리로 채점
# #####################################################################
def enable_self_collision(cfg):
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = True


@configclass
class AntFunRoll11RewardsCfg(AntFunRoll7RewardsCfg):
    """v10 보상 + 바퀴 자세 0.5 + 곧은 다리 0.5 + 몸통 0.45 m 위 0.5. 높이 벌점 기준은 0.6 → 0.9 m (다리 끝으로 구르면 몸통이 0.5~0.7 m)"""

    wheel = RewTerm(func=wheel_posture, weight=0.5)
    straight = RewTerm(func=legs_straight, weight=0.5, params={"sigma": 0.5})
    up = RewTerm(func=torso_above, weight=0.5, params={"height": 0.45, "sensor_cfg": SceneEntityCfg("height_scanner")})
    too_high = RewTerm(func=height_above, weight=-2.0, params={"height": 0.9, "sensor_cfg": SceneEntityCfg("height_scanner")})


@configclass
class AntFunRoll11EnvCfg(AntFunRoll10EnvCfg):
    """[구르기 v11 예쁜 풍차] v10(관절 ±100°, 기본 지형, 재질 무작위, 토크 원본) + 자기 충돌 켬 + 바퀴 자세·곧은 다리·높은 몸통 보상 (Isaac-Ant-Fun-Roll11-v0)"""

    rewards: AntFunRoll11RewardsCfg = AntFunRoll11RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        enable_self_collision(self)


@configclass
class AntFinalJ200CEnvCfg(AntFinalJ200EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        enable_self_collision(self)


@configclass
class AntFinalJ200CRoughEvalEnvCfg(AntFinalJ200RoughEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        enable_self_collision(self)


@configclass
class AntFinalJ200CStockRoughEvalEnvCfg(AntFinalJ200StockRoughEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        enable_self_collision(self)


for _name in BATTERY:
    _cls = _battery_env_cfg(AntFinalJ200CEnvCfg, _name, "J200C")
    globals()[_cls.__name__] = _cls


#   발견: 로봇 재질만 무작위(0.05~1.5)이고 지형의 마찰 결합이 "average" 면 실제 마찰 = (로봇 + 1.0)/2 ≥ 0.53 이라 빙판을 못 겪는다.
#   -> 지형 결합을 multiply 로 (실제 마찰 = 로봇 값 0.02~1.5 그대로). 시험장 빙판(multiply 0.05)과 같은 방식
@configclass
class AntFunJoint200MatWideEventCfg(AntFunJoint200MatEventCfg):
    robot_material = EventTerm(
        func=randomize_robot_material_per_env,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.02, 1.5),
            "dynamic_friction_range": (0.02, 1.5),
            "restitution_range": (0.0, 0.9),
            "num_buckets": 256,
            "make_consistent": True,
            "nominal_fraction": 0.1,
        },
    )


def make_hard_terrain(cfg, max_start_row: int = 15):
    """어려운 은행 + 출발 행 0~max_start_row + 마찰 multiply(반발 max). v12·v13 용."""
    use_course(cfg, ANT_HARD_BANK_CFG)
    cfg.scene.terrain.physics_material = cfg.scene.terrain.physics_material.replace(friction_combine_mode="multiply", restitution_combine_mode="max")
    cfg.curriculum.terrain_teacher.params["max_start_row"] = max_start_row


#   v13 = 예쁜 풍차 v11 에 "점차 어렵게"(어려운 은행 + 출발 0~15행 + 마찰 multiply 0.02~1.5)를 얹음. v11 체크포인트에서 이어서
@configclass
class AntFunRoll13EnvCfg(AntFunRoll11EnvCfg):
    """[구르기 v13] v11 + 점차 어렵게 (Isaac-Ant-Fun-Roll13-v0). v11 에서 이어서 학습"""

    events: AntFunJoint200MatWideEventCfg = AntFunJoint200MatWideEventCfg()

    def __post_init__(self):
        super().__post_init__()
        make_hard_terrain(self)


@configclass
class AntFunJoint200MatLowEventCfg(AntFunJoint200MatEventCfg):
    """학습용: 관절 ±100° + 마찰 0.02~1.0 균등(기본값 비율 0 → 환경의 30 % 가 μ < 0.3), 반발 0~0.5."""

    robot_material = EventTerm(
        func=randomize_robot_material_per_env,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.02, 1.0),
            "dynamic_friction_range": (0.02, 1.0),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 256,
            "make_consistent": True,
            "nominal_fraction": 0.0,
        },
    )


@configclass
class AntFunRoll14EnvCfg(AntFunRoll13EnvCfg):
    """[구르기 v14 미끄러운 바닥 보강] v13 조건(어려운 은행, multiply) + 낮은 마찰 비중 높인 무작위 (Isaac-Ant-Fun-Roll14-v0). 최종 모델에서 이어서"""

    events: AntFunJoint200MatLowEventCfg = AntFunJoint200MatLowEventCfg()


#   v14(낮은 마찰 비중↑) 조건 위에 (1) 발 헛돎 벌점(−0.1): 미끄러우면 살살 가속하는 법을 배우게, (2) 더 빠르게: 구르기 점수의 속도 상한 6 → 12 m/s, 회전 상한 15 → 30 rad/s,
#   회전 목표 8 → 20 rad/s, 전진 1.0 → 2.0, 에너지 벌점 0. 최종 모델에서 이어서 학습 (관측은 그대로라 이어 받기 가능)
@configclass
class AntFunRoll15RewardsCfg(AntFunRoll11RewardsCfg):
    progress = RewTerm(func=ant_mdp.progress_reward, weight=2.0, params={"target_pos": (1000.0, 0.0, 0.0)})
    energy = RewTerm(func=ant_mdp.power_consumption, weight=0.0, params={"gear_ratio": {".*": 15.0}})
    roll = RewTerm(func=roll_on_ground, weight=1.0, params={"max_rate": 30.0, "max_speed": 12.0, "threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces")})
    rate = RewTerm(func=roll_rate_target, weight=1.0, params={"target": 20.0, "sigma": 8.0})
    slip = RewTerm(func=foot_slip_speed, weight=-0.1, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")})


@configclass
class AntFunRoll15EnvCfg(AntFunRoll14EnvCfg):
    """[구르기 v15 더 빠르게 + 미끄러움 적응] v14 조건 + 헛돎 벌점 + 속도 상한 완화 (Isaac-Ant-Fun-Roll15-v0). 최종 모델에서 이어서"""

    rewards: AntFunRoll15RewardsCfg = AntFunRoll15RewardsCfg()


# #####################################################################
#   팀원 패키지(source/.../classic/ant_heldout/, ~/IsaacLab_RS/ant_maps/)는 손대지 않는다. 그쪽 SharedEval 과 같은 세 가지(100마리, 팀원 맵, 원본 보상 7개)를
#   우리 평가 설정(Final 가족: 관측 240·행동·바닥 기준 탈락·접촉 여유) 위에 얹고, 여기에 "지형 타일 끝 x = 80 m 를 넘으면 완주 종료" (mdp.py 31단계) 를 더한다.
#   태스크: Isaac-Ant-TeamMap-Final-v0 (걷기), -Final-J200C-v0 (구르기). 원본 모델은 팀원의 Isaac-Ant-Heldout-v0 그대로
#   실행: tools/eval_teammap.sh, 결과표 tools/make_teammap_table.py (로그의 [COURSE END] 줄 = 완주)
# #####################################################################
TEAMMAP_X_END = 80.0  # [m] 지형 타일 마지막 행의 끝. 여기 닿으면 완주


TEAMMAP_LONG_USD = os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([".."] * 6))), "ant_maps", "generated", "heldout_seed20261008_friction_spectrum_x2.usd")  # 긴 맵 (tools/make_teammap_long.py 가 만듦)


def make_teammap(cfg, x_end: float = TEAMMAP_X_END, long_map: bool = False):
    """우리 평가 설정 cfg 에 팀원 맵 + 원본 보상 7개 (+ 짧은 맵이면 완주 종료) 를 얹는다 (팀원 패키지는 태스크를 만들 때만 읽음: 다른 태스크에 영향 없음).
    long_map=True: 타일 40행짜리 x2 맵을 TeamMapTerrainImporter 로 불러온다. 출발점에서 끝까지 250~315 m 라 완주 종료 없음."""
    from isaaclab_tasks.manager_based.classic.ant_heldout.ant_env_cfg import OriginalV0RewardsCfg, evaluation_terrain_cfg

    cfg.scene.num_envs = 100  # 맵의 출발점 100개 (검증 코드가 100 이 아니면 거부)
    if long_map:
        cfg.scene.terrain = TeamMapTerrainImporterCfg(prim_path="/World/ground", terrain_type="usd", collision_group=-1, usd_path=TEAMMAP_LONG_USD, debug_vis=False)
    else:
        cfg.scene.terrain = evaluation_terrain_cfg()
        cfg.terminations.course_end = DoneTerm(func=reached_course_end, params={"x_end": x_end})
    cfg.rewards = OriginalV0RewardsCfg()


@configclass
class AntTeamMapFinalEnvCfg(AntFinalEnvCfg):
    """[팀원 맵 채점, 걷기 모델용] Final 설정(관절·토크 원본) + 팀원 맵 + 완주 종료 (Isaac-Ant-TeamMap-Final-v0)"""

    def __post_init__(self):
        super().__post_init__()
        make_teammap(self)


@configclass
class AntTeamMapJ200CEnvCfg(AntFinalJ200CEnvCfg):
    """[팀원 맵 채점, 구르기 v11~v15 용] 관절 ±100° + 자기 충돌 (Isaac-Ant-TeamMap-Final-J200C-v0)"""

    def __post_init__(self):
        super().__post_init__()
        make_teammap(self)


@configclass
class AntTeamMap2BaseEnvCfg(AntEnvCfg):
    """[긴 팀원 맵, 수업 원본 모델용] 원본 Isaac-Ant-v0 설정(관측 60, 절대 높이 탈락) + 긴 맵 + 원본 보상 (Isaac-Ant-TeamMap2-Base-v0)"""

    def __post_init__(self):
        super().__post_init__()
        make_teammap(self, long_map=True)


@configclass
class AntTeamMap2FinalEnvCfg(AntFinalEnvCfg):
    """[긴 팀원 맵, 걷기 모델용] (Isaac-Ant-TeamMap2-Final-v0)"""

    def __post_init__(self):
        super().__post_init__()
        make_teammap(self, long_map=True)


@configclass
class AntTeamMap2J200CEnvCfg(AntFinalJ200CEnvCfg):
    """[긴 팀원 맵, 구르기 v11~v15 용] (Isaac-Ant-TeamMap2-Final-J200C-v0)"""

    def __post_init__(self):
        super().__post_init__()
        make_teammap(self, long_map=True)


# #####################################################################
#   팀원 맵에는 0.31 m 보다 깊은 구덩이가 있어 원본 탈락 규칙(절대 높이)으로는 구덩이 = 즉사. 팀 내부 비교용 태스크 = 채점 태스크와 전부 같고 torso_height 탈락만 뺌 (16초 또는 완주 종료만).
#   이름: Isaac-Ant-TeamMapN-<가족>-v0 (짧은 맵, 완주 종료 있음) / Isaac-Ant-TeamMap2N-<가족>-v0 (긴 맵). 가족: Base(원본 설정)·Final·J200C
# #####################################################################
def _teammap_cfg(base_cls, family: str, long_map: bool):
    @configclass
    class TeamMapNoRuleEnvCfg(base_cls):
        def __post_init__(self):
            super().__post_init__()
            make_teammap(self, long_map=long_map)
            self.terminations.torso_height = None  # 팀 내부 비교: 0.31 m 절대 높이 탈락 끔

    TeamMapNoRuleEnvCfg.__name__ = TeamMapNoRuleEnvCfg.__qualname__ = f"AntTeamMap{'2' if long_map else ''}N_{family}"
    return TeamMapNoRuleEnvCfg


for _family, _base in (("Base", AntEnvCfg), ("Final", AntFinalEnvCfg), ("J200C", AntFinalJ200CEnvCfg)):
    for _long in (False, True):
        _cls = _teammap_cfg(_base, _family, _long)
        globals()[_cls.__name__] = _cls

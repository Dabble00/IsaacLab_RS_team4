# 원본 Isaac-Ant-v0 와 다른 점

제출 태스크 `Isaac-Ant-Final-J200C-v0` 기준. 원본 Isaac Lab 파일은 고치지 않았고, 모든 변경은 `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant_robust/` 안에 있다.
USD·링크 길이·질량·관절 개수는 그대로다. 관절 가동 범위·자기 충돌·접촉 여유는 Python 설정(startup 이벤트, spawn 설정)으로 바꿨다.

## 채점 때도 적용되는 것

| 바꾼 것 | 원본 | 우리 | 코드 | 이유 |
| --- | --- | --- | --- | --- |
| 관측 | 60 | 240 = 원본 60 + 앞쪽 높이 센서 180 | `AntC6SceneCfg.height_scanner`, `AntFullObservationsCfg` | 지형을 보게. RayCaster 격자 0.2 m, 몸 뒤 0.8 m ~ 앞 3.0 m × 좌우 ±0.8 m, 몸통 요(yaw) 기준 |
| 로봇 접촉 여유 (contact offset) | 0.0032 m | 0.04 m | `AntKEventCfg.robot_contact_offset`, `FINAL_CONTACT_OFFSET` | 메시 땅에서 발이 박힌 뒤 튕겨 뒤집힘. 학습 없이 T2 39.7 → 86.3 |
| 관절 가동 범위 | 엉덩이 ±40°, 발목 30~100° | 8개 모두 ±100° | `AntFinalJ200EventCfg.widen_joints` (`write_joint_position_limit_to_sim`) | 다리를 바퀴살처럼 뻗으려면 발목 0° 가 필요 |
| 자기 충돌 | 꺼짐 | 켬 | `enable_self_collision` | 관절을 넓히면 다리가 몸통·다른 다리를 뚫는다 |
| PhysX 접촉 버퍼 | 기본 | 2배 | `AntFullEnvCfg.__post_init__` | 큰 메시 지형에서 버퍼가 안 터지게. 물리 결과 동일 |
| 장면 복제 | `clone_in_fabric=True` | False | `AntFullEnvCfg` | 센서와 fabric 복제는 같이 못 쓴다 |

안 바꾼 것: USD, 링크 길이·질량·형태·관절 개수, 토크(행동 배율 7.5, 액추에이터 한계), 보상 7개, 종료 규칙(16초 또는 몸통 절대 높이 0.31 m 미만), 바닥(`scene.terrain` 을 건드리지 않음), PPO 설정(신경망 [400, 200, 100], 4096 × 32 걸음).

## 학습 때만 쓴 것 (`Isaac-Ant-Fun-Roll15-v0`)

| 항목 | 내용 | 코드 |
| --- | --- | --- |
| 지형 | 생성기 은행 40줄 (도형 11종 무작위 조합 20 + Isaac Lab 기본 지형 보통 7 + 어려운 9(구덩이·틈·레일·징검돌 포함) + 평지 4. 구르기 v11 때는 29 + 7 + 4), 어려운 판, 출발 0~15행, 학습 중 점수 낮은 줄을 더 자주 냄 (은행 40줄은 고정) | `train_env_cfg.py` (`LevelBankTerrainGenerator`), `AntC38TrainEnvCfg`, `make_hard_terrain` |
| 출제자 | 칸별 점수로 "배울 수 있는" 지형을 더 자주 (PLR) | `AntPLRCurriculumCfg` (`PrioritizedTerrainReplay`) |
| 로봇 재질 무작위 | 환경마다 정·동마찰 0.02~1.0, 반발 0~0.5. 바닥과 multiply 결합 | `AntFunJoint200MatLowEventCfg`, `make_hard_terrain` |
| 넘어짐 판정 | 원본 절대 높이 0.31 m 에 바닥 기준 0.31 m 추가 | `AntGroundAwareTerminationsCfg` |
| 접촉 센서 | 몸 전체 접촉 힘 (보상 계산용, 관측에는 없음) | `AntC39SceneCfg.contact_forces` |
| 보상 | 구르기 학습 보상 16개 ([REWARDS.md](REWARDS.md)) | `AntFunRoll15RewardsCfg` |
| 학습률 | 긴 학습 구간(v13 → 11,700회)만 고정 5e-5 | `agent.algorithm.schedule=fixed agent.algorithm.learning_rate=5e-5` |

## 한계

- 빙판(마찰 0.05)에서는 어떤 이동 방식도 0 근처. 마찰 0.1~0.3 에서는 걷기 모델이 구르기보다 낫다 (RESULTS.md 3절).
- 원본 리셋 결함: 리셋 때 발끝이 땅에 박힌 채 생성되어 첫 걸음에 최대 2 m 튀고 가끔 뒤집힌다. 모든 팀 공통이라 제출은 원본 리셋 그대로.
- 높이 센서 격자는 몸통 요 기준이라 구를 때 흔들린다. 진행 방향 기준 센서로 바꾼 실험은 점수가 더 오르지 않아 쓰지 않았다.
- 높이 센서는 구르기에도 필요하다. 같은 절차·같은 seed 로 센서 관측만 뺀 모델은 시험장 35종 98.5 (센서 있음 164.5), T2 105.7 (182.8). 학습 지형 점수는 비슷하므로 센서는 처음 보는 지형 일반화에 쓰인다.

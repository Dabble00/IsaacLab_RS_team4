# 보상 함수

채점(`play_one_episode.py`)은 원본 보상 7개 그대로다. 제출 태스크 `Isaac-Ant-Final-J200C-v0` 는 보상을 바꾸지 않는다.
아래 학습 보상은 학습 태스크 `Isaac-Ant-Fun-Roll15-v0` 에서만 쓴다. 코드: `full_env_cfg.py` 의 `AntFunRoll15RewardsCfg` 와 부모 클래스, 함수는 `mdp.py`.

## 원본 보상 7개 (채점용, `classic/ant/ant_env_cfg.py`)

| 항 | 가중치 | 뜻 |
| --- | ---: | --- |
| progress | 1.0 | 목표(+x 1000 m) 쪽으로 간 거리. 점수의 대부분 (1 m ≈ 1점) |
| alive | 0.5 | 살아 있으면 매 걸음 |
| upright | 0.1 | 몸통이 똑바로 서 있으면 |
| move_to_target | 0.5 | 목표 방향으로 움직이면 |
| action_l2 | −0.005 | 행동 크기 벌점 |
| energy | −0.05 | 토크 × 관절 속도 벌점 |
| joint_pos_limits | −0.1 | 관절 한계 가까이 가면 벌점 |

## 구르기 v15 학습 보상 16개 (`AntFunRoll15RewardsCfg`)

원본 7개 중 upright·move_to_target 은 끄고(구르면 몸통이 서 있지 않다) 구르기 보상을 더했다.

| 항 | 함수 | 가중치 | 뜻 |
| --- | --- | ---: | --- |
| progress | progress_reward | 2.0 | 앞으로 간 거리 (원본 1.0 → 2.0) |
| alive | is_alive | 0.5 | 원본 그대로 |
| action_l2 | action_l2 | −0.005 | 원본 그대로 |
| joint_pos_limits | joint_pos_limits_penalty_ratio | −0.1 | 원본 그대로 (한계는 ±100°) |
| energy | power_consumption | 0.0 | 원본 −0.05 → 0 |
| roll | roll_on_ground | 1.0 | 몸이 땅에 닿아 있을 때만, 앞으로 구르는 각속도 × 전진 속도 (상한 30 rad/s, 12 m/s) |
| streak | roll_streak | 2.0 | 끊기지 않고 구른 시간 (2초면 1.0) |
| rate | roll_rate_target | 1.0 | 회전 속도 20 rad/s 근처면 1 |
| wobble | roll_wobble_l2 | −0.005 | 구르는 축 외 흔들림 벌점 |
| too_high | height_above | −2.0 | 땅에서 0.9 m 넘게 뜨면 벌점 |
| bounce | vertical_speed_l2 | −0.02 | 위아래 속도 벌점 |
| sideways | lateral_speed_l1 | −0.3 | 옆으로 새는 속도 벌점 |
| wheel | wheel_posture | 0.5 | 몸통 평면이 수직 (바퀴 자세) |
| straight | legs_straight | 0.5 | 다리를 곧게 뻗은 X 자 |
| up | torso_above | 0.5 | 몸통이 땅에서 0.45 m 위 |
| slip | foot_slip_speed | −0.1 | 땅에 닿은 발이 미끄러지는 속도 벌점 |

## 바뀐 순서

| 단계 | 바뀐 보상 | 이유 | T2 |
| --- | --- | --- | ---: |
| v2~v4 | 구르기, 연속 구르기(streak·rate·wobble) | 튕기며 뒤집기만 해서 | 공중에서만 돎 |
| v8 | 땅에 닿았을 때만 구르기 점수, 튕김·옆으로 새기 벌점, 전진 1.0 | 땅에서 굴러 앞으로 | 141 (v10) |
| v11 | wheel·straight·up, too_high 0.6 → 0.9 m | 다리가 몸통을 뚫고 괴상하게 굴러서. 자기 충돌 켬 | 165.2 |
| v15 | 전진 2.0, 구르기 상한 6 → 12 m/s, 회전 목표 8 → 20 rad/s, 에너지 0, slip −0.1 | 더 빠르게, 미끄러운 바닥에서 살살 | 182.8 |

보상은 매 걸음 `weight × dt(1/60 s)` 로 더해진다.

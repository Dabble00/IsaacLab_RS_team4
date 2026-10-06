# Ant 자체 평가 맵

`ant_maps_evaluation.zip`은 수업용 `IsaacLab_RS`의 `Isaac-Ant-v0` 기반 모델을 **학습에 쓰지 않은 고정 맵**에서 평가하는 파일입니다. 기준 환경은 Isaac Lab 2.3.0 / Isaac Sim 5.1.0입니다. ZIP에는 맵과 평가 태스크만 있으며, 학습 코드·체크포인트·맵 생성기는 없습니다.

## 설치와 실행

본인의 Isaac Lab Python 환경을 활성화한 터미널에서 실행하세요. `~/IsaacLab_RS/scripts/reinforcement_learning/rsl_rl/play_one_episode.py`와 **본인이 학습한 `.pt` 체크포인트**가 먼저 있어야 합니다. 아래 `/absolute/path/to/your/model.pt`를 실제 체크포인트 경로로 바꾸세요.

```bash
cd ~/IsaacLab_RS
unzip /path/to/ant_maps_evaluation.zip -d .
CHECKPOINT="/absolute/path/to/your/model.pt"
ls scripts/reinforcement_learning/rsl_rl/play_one_episode.py "$CHECKPOINT"

ANT_EVAL_BASE_TASK=Isaac-Ant-v0 \
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-SharedEval-v0 \
  --seed 24 --num_envs 100 \
  --checkpoint "$CHECKPOINT" --headless
```

`ANT_EVAL_BASE_TASK`에는 **그 체크포인트를 학습시킨 등록 태스크 ID**를 넣으세요. 원본 `Isaac-Ant-v0`로 학습했다면 위 명령 그대로입니다. 새 태스크 ID로 학습했다면 그 ID로 바꾸고, 관측·행동 함수와 PPO 설정을 포함한 학습 태스크 코드도 본인 프로젝트에 설치해야 합니다. 기존 `Isaac-Ant-v0` ID의 코드를 직접 수정해 학습했다면 ID는 그대로 두고 수정한 학습 설정을 유지하세요. 학습 시 Hydra로 관측이나 네트워크를 덮어썼다면 평가에도 같은 설정을 전달해야 합니다.

완료되면 `[INFO] Completed first episodes: 100/100`과 `[RESULT] Episode reward total: mean=..., std=...`가 출력됩니다. 이 **평균·표준편차**를 발표 PPT에 적으세요. 화면으로 보려면 `--headless`를 빼면 됩니다.

## 평가 조건

- `--seed 24 --num_envs 100`으로 저장된 평지 출발점 100개를 사용합니다. 출발 지면은 z=0, Ant 몸통은 원본 v0처럼 z=0.5입니다.
- 맵은 8m 타일 20행×10열이며, 평지 20개와 Isaac Lab 기본 험지 6종 180개입니다. 각 험지 종류에 정지/동마찰 **0.05/0.03, 0.25/0.20, 0.50/0.40, 0.75/0.60, 1.00/0.80, 1.25/1.00**을 고르게 배치했습니다. 평지·외곽은 1.0/1.0입니다. 접촉 마찰은 로봇 발 재질과 `multiply` 방식으로 결합됩니다.
- 출발점 바로 앞(+x)에 6종 험지 × 대표 마찰 3단계(얼음·중간·고무)가 배치됩니다. 실제 접촉 지형은 로봇의 이동 경로에 따라 달라집니다.
- 관측·행동·로봇·센서·리셋·종료 조건과 PPO 설정은 학습 태스크를 따르고, **보상 7항만 원본 `Isaac-Ant-v0`**로 바꿉니다. 원본 보상 함수가 들어 있는 `isaaclab_tasks.manager_based.classic.humanoid.mdp` 구현을 직접 수정했다면 원본으로 복원해야 합니다.
- 100개 출발점이 19개 평지 타일을 재사용하므로 화면상 로봇이 겹쳐 보일 수 있습니다. 환경 간 충돌을 차단하는 원본 설정 `scene.filter_collisions=True`를 유지하세요.

압축을 풀면 `ant_maps/generated/heldout_seed20261008_friction_spectrum.usd`와 `.json`, `ant_maps/evaluation_metadata.py`, 평가 태스크 Python 파일 2개, 이 README가 설치됩니다. 맵 재생성은 필요하지 않습니다. 이 자체 평가 맵은 조교의 비공개 평가 맵과 다릅니다.

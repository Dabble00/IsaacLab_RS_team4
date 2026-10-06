# 받아서 바로 돌리기

이 저장소는 수업 저장소(`cailab-hy/IsaacLab_RS`)에 아래를 더한 것이다.

| 추가된 것 | 위치 |
| --- | --- |
| 태스크 코드 (학습·평가·시험장·지형 생성기) | `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant_robust/` |
| 팀원 평가맵 | `source/.../classic/ant_heldout/`, `ant_maps/` |
| 체크포인트 + 학습 설정 | `logs/rsl_rl/ant/2026-10-02_12-19-07_fun17_roll15/`, 사본 `docs/assignment1/models/` |
| 학습 로그 (v11 → v13 → 긴 학습 → v15) | `logs/rsl_rl/ant/*roll11`, `*roll13`, `*roll13long_s42_g030`, `*roll15`, `logs/accel_roll13long_s42/` |
| 영상·그림·결과표 | `docs/assignment1/` |
| 스크립트 | `scripts/assignment1/` |

## 설치

A. 이미 `~/IsaacLab_RS` 가 있으면 폴더만 복사한다. 원본 파일은 덮어쓰지 않는다.

```bash
git clone https://github.com/Dabble00/IsaacLab_RS_team4.git /tmp/team4
cd ~/IsaacLab_RS
cp -r /tmp/team4/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant_robust  source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/
cp -r /tmp/team4/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant_heldout source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/
cp -r /tmp/team4/ant_maps .
mkdir -p docs/assignment1 && cp -r /tmp/team4/docs/assignment1/models docs/assignment1/
cp -r /tmp/team4/scripts/assignment1 scripts/
```

`isaaclab_tasks` 가 `manager_based/` 아래 폴더를 자동으로 읽으므로 등록 파일을 고칠 필요 없다. `trimesh` 가 필요하다 (Isaac Lab 설치에 포함).

B. 저장소를 통째로 쓰려면 clone 뒤 `./isaaclab.sh --install` 로 extension 을 다시 설치한다.

## 점수 재기 (seed 24, 100마리, 원본 보상)

```bash
conda activate lerobot-arena
cd ~/IsaacLab_RS
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-Final-J200C-v0 --seed 24 --num_envs 100 --headless \
  --checkpoint docs/assignment1/models/roll15_model_13195.pt
```

마지막에 `[RESULT] Episode reward total: mean=... std=...` 가 나온다 (평지 182.5 ± 20.0). `--video --video_length 960` 을 붙이면 0번 개미 영상이 체크포인트 폴더의 `videos/play/` 에 생긴다.

| 태스크 | 지형 |
| --- | --- |
| `Isaac-Ant-Final-J200C-v0` | 평지 (제출 태스크. `scene.terrain` 만 바꾸면 그 지형) |
| `Isaac-Ant-Eval-StockRough-Final-J200C-v0` | T1 = Isaac Lab 기본 험지 |
| `Isaac-Ant-Eval-AntRough-Final-J200C-v0` | T2 = 개미 크기 울퉁불퉁 메시 |
| `Isaac-Ant-Battery-<종류>-J200C-v0` | 시험장 35종 중 하나 (목록은 `ant_robust/__init__.py` 의 `BATTERY_NAMES`) |
| `Isaac-Ant-TeamMap-Final-J200C-v0` | 팀원 평가맵 (맵 끝 x = 80 m 에 닿으면 완주 종료) |
| `Isaac-Ant-TeamMap2-Final-J200C-v0` | 두 벌 이어 붙인 긴 맵 (`scripts/assignment1/make_teammap_long.py` 로 먼저 생성) |
| `Isaac-Ant-TeamMapN-J200C-v0`, `-TeamMap2N-` | 위 둘에서 0.31 m 탈락만 끈 것 (팀 내부 비교용) |

수업 원본 모델은 접미사 없는 가족(`Isaac-Ant-v0`, `Isaac-Ant-Eval-StockRough-v0`, `Isaac-Ant-Eval-AntRough-v0`, `Isaac-Ant-Battery-<종류>-v0`), 걷기 모델(관측 240, 관절 원본)은 `-Final-v0` 가족으로 잰다.

35종 한 번에: `bash scripts/assignment1/eval_all.sh <태그> <체크포인트> J200C` → `results/<태그>_*.log`, `results/bt_<태그>_<종류>.log`

## 학습 다시 하기

`scripts/assignment1/train_lineage.sh` 에 4단계 명령이 있다 (RTX 4070 8 GB: 1,000회 ≈ 22분).

| 단계 | 태스크 | 회수 | 시작점 |
| --- | --- | ---: | --- |
| 1. v11 | `Isaac-Ant-Fun-Roll11-v0` | 1,500 | 처음부터 |
| 2. v13 | `Isaac-Ant-Fun-Roll13-v0` | +1,500 | v11 마지막 |
| 3. 긴 학습 | `Isaac-Ant-Fun-Roll13-v0` (고정 학습률 5e-5) | +8,700 (11,700까지) | v13 마지막. 50회 세대로 나눠 세대마다 채점·지형 은행 진화, 최고 세대 30 사용 |
| 4. v15 | `Isaac-Ant-Fun-Roll15-v0` | +1,500 (13,200) | 긴 학습 최고 세대 |

지형 생성기 인자 (은행 저장·진화):

```
env.scene.terrain.terrain_generator.bank_out=results/bank_<이름>.json
env.scene.terrain.terrain_generator.evolve_seed=42
env.scene.terrain.terrain_generator.evolve_num_replace=8 env.scene.terrain.terrain_generator.evolve_num_new=4
env.scene.terrain.terrain_generator.evolve_min_kind_lanes=1 env.scene.terrain.terrain_generator.evolve_max_kind_lanes=10
env.sim.physx.gpu_max_rigid_patch_count=327680
```

이어서 배울 때는 `--resume --load_run <이전 run 폴더> --load_checkpoint model_<n>.pt`.

## 지형 생성기 (`ant_robust/train_env_cfg.py`)

- `LevelBankTerrainGenerator`: 줄(lane)마다 도형 11종(계단·상자·구덩이·언덕·물결·돌·틈 …)을 무작위로 조합한 높이맵을 만들고, 줄 40개를 은행(JSON)으로 저장·불러온다.
- 세대 진화: 칸별 점수를 보고 점수가 낮거나 배울 게 없는 줄을 새 줄로 바꾼다 (`evolve_*`).
- `LIBRARY_LANES`, `LIBRARY_LANES_HARD`: Isaac Lab 기본 지형(계단·경사·상자 …) 줄. 평가용 시험장(`eval_env_cfg.py`)과는 종류를 나눴다.

## 긴 팀원 맵 (선택)

```bash
pip install usd-core
python scripts/assignment1/make_teammap_long.py
```

`ant_maps/generated/heldout_seed20261008_friction_spectrum_x2.usd` 가 생긴다 (41 MB, 저장소에 없음).

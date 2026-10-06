#!/usr/bin/env bash
# 구르기 v15 계보를 처음부터 다시 배우는 명령 4단계 (RTX 4070 8 GB: 1,000회 ≈ 22분, 전체 약 5시간). 각 단계는 앞 단계의 마지막 체크포인트에서 이어서.
#   사용법: bash scripts/assignment1/train_lineage.sh            (중간에 멈췄으면 STEP=3 처럼 환경변수로 단계부터 다시)
#   지형 은행은 run 마다 results/bank_<이름>.json 으로 저장되고 세대 진화 인자(evolve_*)로 학습 중 바뀐다 (train_env_cfg.py).
set -eo pipefail
cd "$(dirname "$0")/../.."
G=env.scene.terrain.terrain_generator
EVOLVE="$G.evolve_seed=42 $G.evolve_num_replace=8 $G.evolve_num_new=4 $G.evolve_min_kind_lanes=1 $G.evolve_max_kind_lanes=10"
PATCH=env.sim.physx.gpu_max_rigid_patch_count=327680
TRAIN="./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --headless --seed 42 agent.save_interval=250 $PATCH"
last_run() { ls -d logs/rsl_rl/ant/*_$1 | tail -1; }
last_ckpt() { ls "$1"/model_*.pt | sort -V | tail -1; }
mkdir -p results
STEP=${STEP:-1}
if [ "$STEP" -le 1 ]; then  # 1. v11: 관절 ±100° + 자기 충돌 + 풍차 자세 보상, 처음부터 1,500회
  $TRAIN --task Isaac-Ant-Fun-Roll11-v0 --run_name roll11 --max_iterations 1500 $G.bank_out=results/bank_roll11.json $EVOLVE
fi
if [ "$STEP" -le 2 ]; then  # 2. v13: 어려운 지형 은행 + 마찰 multiply, v11 에서 이어서 1,500회
  R=$(last_run roll11); $TRAIN --task Isaac-Ant-Fun-Roll13-v0 --run_name roll13 --max_iterations 1500 --resume --load_run "$(basename "$R")" --load_checkpoint "$(basename "$(last_ckpt "$R")")" $G.bank_out=results/bank_roll13.json $EVOLVE
fi
if [ "$STEP" -le 3 ]; then  # 3. 긴 학습: v13 조건 그대로 8,700회 더 (고정 학습률 5e-5). 우리는 50회짜리 세대로 나눠 세대마다 채점·은행 진화를 했고 최고 세대 30(11,700회)을 썼다
  R=$(last_run roll13); $TRAIN --task Isaac-Ant-Fun-Roll13-v0 --run_name roll13long --max_iterations 8700 --resume --load_run "$(basename "$R")" --load_checkpoint "$(basename "$(last_ckpt "$R")")" \
    agent.algorithm.schedule=fixed agent.algorithm.learning_rate=5e-5 $G.bank_in=$PWD/results/bank_roll13.json $G.bank_out=results/bank_roll13long.json $EVOLVE
fi
if [ "$STEP" -le 4 ]; then  # 4. v15: 헛돎 벌점 + 속도 상한 완화 + 낮은 마찰 비중, 긴 학습에서 이어서 1,500회 (= 제출 모델)
  R=$(last_run roll13long); $TRAIN --task Isaac-Ant-Fun-Roll15-v0 --run_name roll15 --max_iterations 1500 --resume --load_run "$(basename "$R")" --load_checkpoint "$(basename "$(last_ckpt "$R")")" $G.bank_out=results/bank_roll15.json $EVOLVE
fi
echo "끝: $(last_ckpt "$(last_run roll15)")  → bash scripts/assignment1/eval_all.sh roll15 <이 경로> J200C"

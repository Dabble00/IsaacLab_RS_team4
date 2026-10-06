#!/usr/bin/env bash
# 한 모델을 채점 방식(seed 24, 100마리, 원본 보상)으로 평지·T1·T2 + 시험장 35종에서 재고 로그를 남긴다 (영상 없음, 약 40분).
#   사용법: bash scripts/assignment1/eval_all.sh <태그> <체크포인트> [가족]
#     가족: J200C = 구르기 v15 (기본) / Final = 걷기 모델(관측 240, 관절 원본) / Base = 수업 원본 모델(관측 60)
#   결과:   results/<태그>_E1_plane.log, _E2_T1.log, _E3_T2.log, results/bt_<태그>_<시험장>.log → 마지막에 시험장 평균
set -o pipefail
TAG=$1; CKPT=$2; FAM=${3:-J200C}
[ -n "$CKPT" ] || { echo "사용법: $0 <태그> <체크포인트> [J200C|Final|Base]"; exit 1; }
cd "$(dirname "$0")/../.."
case $FAM in
  Base)  PLANE=Isaac-Ant-v0;             T1=Isaac-Ant-Eval-StockRough-v0;             T2=Isaac-Ant-Eval-AntRough-v0;             SUF="";;
  Final) PLANE=Isaac-Ant-Final-v0;       T1=Isaac-Ant-Eval-StockRough-Final-v0;       T2=Isaac-Ant-Eval-AntRough-Final-v0;       SUF="-Final";;
  J200C) PLANE=Isaac-Ant-Final-J200C-v0; T1=Isaac-Ant-Eval-StockRough-Final-J200C-v0; T2=Isaac-Ant-Eval-AntRough-Final-J200C-v0; SUF="-J200C";;
  *) echo "가족은 Base / Final / J200C 중 하나"; exit 1;;
esac
mkdir -p results
run() { ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py --task "$2" --checkpoint "$CKPT" --seed 24 --num_envs 100 --headless 2>&1 | tee "results/$1.log" | grep "\[RESULT\]"; }
run ${TAG}_E1_plane $PLANE
run ${TAG}_E2_T1 $T1
run ${TAG}_E3_T2 $T2
NAMES=$(python -c "import ast,re; s=open('source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant_robust/__init__.py',encoding='utf-8').read(); print(' '.join(ast.literal_eval(re.search(r'BATTERY_NAMES = (\(.*?\n\))', s, re.S).group(1))))")
for n in $NAMES; do run bt_${TAG}_$n Isaac-Ant-Battery-$n$SUF-v0; done
python - "$TAG" <<'PY'
import re, sys, glob
tag = sys.argv[1]; vals = []
for f in sorted(glob.glob(f"results/bt_{tag}_*.log")):
    m = re.search(r"reward total: mean=([-\d.]+)", open(f, encoding="utf-8", errors="ignore").read())
    if m:
        vals.append(float(m.group(1))); print(f"{f[len('results/bt_' + tag) + 1:-4]:>22}: {float(m.group(1)):7.1f}")
print(f"시험장 평균 ({len(vals)}종): {sum(vals) / len(vals):.1f}")
PY

#!/bin/bash
# Multi-seed robustness study (round-3, Phase 5, reduced scope: 3 seeds
# not 5, due to per-seed retrain cost -- stated honestly, not silently
# shrunk). For each seed: regenerate data, retrain everything, run the
# canonical evaluate.py, capture the headline numbers. Restores seed=42
# (this repo's documented default) at the end so the delivered state
# matches every other doc in this repo.
set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)
RESULTS_FILE="$ROOT/evaluation/seed_robustness_results.json"

echo "[" > "$RESULTS_FILE"
FIRST=1

for SEED in 42 101 2026; do
  echo "=================================================="
  echo "SEED $SEED"
  echo "=================================================="
  cd "$ROOT/data" && python3 generate_data.py "$SEED" > /tmp/seed_${SEED}_gen.log 2>&1
  cd "$ROOT/ml-service" && python3 train.py > /tmp/seed_${SEED}_train.log 2>&1
  python3 two_stage_policy.py > /tmp/seed_${SEED}_stage1.log 2>&1
  cd "$ROOT/evaluation" && python3 evaluate.py > /tmp/seed_${SEED}_eval.log 2>&1

  LIFT_BASELINE=$(python3 -c "import json; print(json.load(open('results.json'))['lift_over_baseline_pct'])")
  LIFT_MLONLY=$(python3 -c "import json; print(json.load(open('results.json'))['lift_over_ml_only_pct'])")
  MEAN_REGRET=$(python3 -c "import json; print(json.load(open('results.json'))['regret_vs_oracle_optimal']['mean'])")
  TOP1=$(python3 -c "import json; print(json.load(open('results.json'))['top_k_accuracy']['top1_optimal_action_rate'])")
  RECOUP_REWARD=$(python3 -c "import json; print(json.load(open('results.json'))['recoup']['mean_net_reward'])")

  echo "  lift_over_baseline_pct=$LIFT_BASELINE lift_over_ml_only_pct=$LIFT_MLONLY mean_regret=$MEAN_REGRET top1=$TOP1"

  if [ $FIRST -eq 0 ]; then echo "," >> "$RESULTS_FILE"; fi
  FIRST=0
  cat >> "$RESULTS_FILE" << EOF
  {
    "seed": $SEED,
    "lift_over_baseline_pct": $LIFT_BASELINE,
    "lift_over_ml_only_pct": $LIFT_MLONLY,
    "mean_regret": $MEAN_REGRET,
    "top1_optimal_action_rate": $TOP1,
    "recoup_mean_net_reward": $RECOUP_REWARD
  }
EOF
done

echo "]" >> "$RESULTS_FILE"

echo ""
echo "=================================================="
echo "Restoring default seed=42 as the repo's committed state..."
cd "$ROOT/data" && python3 generate_data.py 42 > /dev/null 2>&1
cd "$ROOT/ml-service" && python3 train.py > /dev/null 2>&1
python3 two_stage_policy.py > /dev/null 2>&1
cd "$ROOT/evaluation" && python3 evaluate.py > /dev/null 2>&1

echo "Done. Results across 3 seeds saved to $RESULTS_FILE"
python3 -c "
import json
results = json.load(open('$RESULTS_FILE'))
import statistics as s
lifts = [r['lift_over_baseline_pct'] for r in results]
regrets = [r['mean_regret'] for r in results]
print(f'lift_over_baseline_pct: mean={s.mean(lifts):.1f} std={s.pstdev(lifts):.1f} min={min(lifts)} max={max(lifts)}')
print(f'mean_regret: mean={s.mean(regrets):.1f} std={s.pstdev(regrets):.1f} min={min(regrets)} max={max(regrets)}')
"

#!/bin/bash
# Final multi-seed T-learner vs DR validation (V13, Step 2-6).
# Regenerates data, retrains T-learner AND DR, and records BOTH policies'
# metrics for each seed -- unlike round-3's seed study (which only
# tested the single production T-learner policy), this specifically
# compares T-learner vs DR head-to-head per seed, so we can see whether
# V12's "+11.4%" DR win holds up or was single-seed luck, exactly like
# round 3 found for the original T-learner headline number.
set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)
RESULTS_FILE="$ROOT/evaluation/dr_multiseed_results.json"

echo "[" > "$RESULTS_FILE"
FIRST=1

for SEED in 42 101 2026 7 555; do
  echo "=================================================="
  echo "SEED $SEED"
  echo "=================================================="
  cd "$ROOT/data" && python3 generate_data.py "$SEED" > /tmp/dr_seed_${SEED}_gen.log 2>&1
  cd "$ROOT/ml-service" && python3 train.py > /tmp/dr_seed_${SEED}_train.log 2>&1
  cd "$ROOT/evaluation" && python3 dr_cross_fitting.py > /tmp/dr_seed_${SEED}_dr.log 2>&1

  T_REWARD=$(python3 -c "import json; print(json.load(open('dr_cross_fitting_results.json'))['main_policy_comparison']['t_learner']['mean_net_reward'])")
  T_REGRET=$(python3 -c "import json; print(json.load(open('dr_cross_fitting_results.json'))['main_policy_comparison']['t_learner']['mean_regret'])")
  T_TOP1=$(python3 -c "import json; print(json.load(open('dr_cross_fitting_results.json'))['main_policy_comparison']['t_learner']['top1_pct'])")
  DR_REWARD=$(python3 -c "import json; print(json.load(open('dr_cross_fitting_results.json'))['main_policy_comparison']['dr']['mean_net_reward'])")
  DR_REGRET=$(python3 -c "import json; print(json.load(open('dr_cross_fitting_results.json'))['main_policy_comparison']['dr']['mean_regret'])")
  DR_TOP1=$(python3 -c "import json; print(json.load(open('dr_cross_fitting_results.json'))['main_policy_comparison']['dr']['top1_pct'])")
  REWARD_DELTA=$(python3 -c "import json; print(json.load(open('dr_cross_fitting_results.json'))['main_policy_comparison']['reward_delta_pct'])")
  REGRET_DELTA=$(python3 -c "import json; print(json.load(open('dr_cross_fitting_results.json'))['main_policy_comparison']['regret_delta_pct'])")
  TOP1_DELTA=$(python3 -c "import json; print(json.load(open('dr_cross_fitting_results.json'))['main_policy_comparison']['top1_delta_pp'])")

  echo "  T-learner: reward=$T_REWARD regret=$T_REGRET top1=$T_TOP1"
  echo "  DR:        reward=$DR_REWARD regret=$DR_REGRET top1=$DR_TOP1"
  echo "  Delta:     reward=$REWARD_DELTA% regret=$REGRET_DELTA% top1=${TOP1_DELTA}pp"

  if [ $FIRST -eq 0 ]; then echo "," >> "$RESULTS_FILE"; fi
  FIRST=0
  cat >> "$RESULTS_FILE" << EOF
  {
    "seed": $SEED,
    "t_learner_reward": $T_REWARD, "t_learner_regret": $T_REGRET, "t_learner_top1": $T_TOP1,
    "dr_reward": $DR_REWARD, "dr_regret": $DR_REGRET, "dr_top1": $DR_TOP1,
    "reward_delta_pct": $REWARD_DELTA, "regret_delta_pct": $REGRET_DELTA, "top1_delta_pp": $TOP1_DELTA
  }
EOF
done

echo "]" >> "$RESULTS_FILE"

echo ""
echo "=================================================="
echo "Restoring default seed=42 as the repo's committed state..."
cd "$ROOT/data" && python3 generate_data.py 42 > /dev/null 2>&1
cd "$ROOT/ml-service" && python3 train.py > /dev/null 2>&1
cd "$ROOT/evaluation" && python3 dr_cross_fitting.py > /dev/null 2>&1

echo "Done. Results across 5 seeds saved to $RESULTS_FILE"
python3 -c "
import json, statistics as s
results = json.load(open('$RESULTS_FILE'))
reward_deltas = [r['reward_delta_pct'] for r in results]
regret_deltas = [r['regret_delta_pct'] for r in results]
top1_deltas = [r['top1_delta_pp'] for r in results]
n_dr_wins_reward = sum(1 for d in reward_deltas if d > 0)
print(f'reward_delta_pct (DR vs T-learner): mean={s.mean(reward_deltas):.1f} std={s.pstdev(reward_deltas):.1f} min={min(reward_deltas)} max={max(reward_deltas)}')
print(f'regret_delta_pct (DR vs T-learner): mean={s.mean(regret_deltas):.1f} std={s.pstdev(regret_deltas):.1f} min={min(regret_deltas)} max={max(regret_deltas)}')
print(f'top1_delta_pp (DR vs T-learner):    mean={s.mean(top1_deltas):.1f} std={s.pstdev(top1_deltas):.1f} min={min(top1_deltas)} max={max(top1_deltas)}')
print(f'DR beat T-learner on reward in {n_dr_wins_reward}/{len(results)} seeds')
"

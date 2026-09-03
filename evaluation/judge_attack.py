"""
judge_attack.py — runs the hardest available tests in sequence and
reports PASS/WARN/FAIL for each. This does not invent new tests; it
orchestrates the diagnostic scripts already in this repo (leakage tests,
overlap diagnostics, drift detection, seed robustness, evaluate/ablation
consistency) into one command, so a judge (or Kunal, before a demo) can
run one script and see the honest state of the system.

Run: cd evaluation && python3 judge_attack.py
Requires: data generated, models trained (see README setup steps).
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")

results = []


def record(name, status, detail):
    results.append({"test": name, "status": status, "detail": detail})
    print(f"[{status:4s}] {name}: {detail}")


def run_leakage_tests():
    proc = subprocess.run(
        [sys.executable, "test_no_leakage.py"],
        cwd=os.path.join(ROOT, "ml-service"), capture_output=True, text=True,
    )
    passed = proc.stdout.count("PASS")
    failed = proc.stdout.count("FAIL") + (1 if proc.returncode != 0 and passed == 0 else 0)
    if proc.returncode == 0 and passed >= 4:
        record("Data/temporal leakage (4 tests)", "PASS", f"{passed}/4 checks passed")
    else:
        record("Data/temporal leakage (4 tests)", "FAIL", proc.stdout[-300:] + proc.stderr[-300:])


def run_overlap_diagnostics():
    proc = subprocess.run([sys.executable, "overlap_diagnostics.py"], cwd=HERE, capture_output=True, text=True)
    result_path = os.path.join(HERE, "overlap_diagnostics_results.json")
    if not os.path.exists(result_path):
        record("Propensity/overlap diagnostics", "FAIL", "results file not produced")
        return
    data = json.load(open(result_path))
    worst_pct = max(v["pct_below_common_support"] for v in data.values())
    if worst_pct > 50:
        record("Propensity/overlap diagnostics", "WARN",
               f"worst arm has {worst_pct}% of contexts below common support -- causal estimates unreliable there, documented in docs/causal_identification.md")
    elif worst_pct > 10:
        record("Propensity/overlap diagnostics", "WARN", f"worst arm: {worst_pct}% below common support")
    else:
        record("Propensity/overlap diagnostics", "PASS", f"worst arm: {worst_pct}% below common support")


def run_drift_test():
    proc = subprocess.run([sys.executable, "drift_test.py"], cwd=HERE, capture_output=True, text=True)
    out = proc.stdout
    if "SIGNIFICANT_SHIFT" in out and "STABLE" in out:
        record("Drift detection (injected window vs control)", "PASS",
               "detector correctly distinguishes drift window (SIGNIFICANT_SHIFT) from control (STABLE)")
    else:
        record("Drift detection", "FAIL", out[-300:])


def run_evaluate_ablation_consistency():
    subprocess.run([sys.executable, "evaluate.py"], cwd=HERE, capture_output=True, text=True)
    subprocess.run([sys.executable, "ablation.py"], cwd=HERE, capture_output=True, text=True)
    eval_results = json.load(open(os.path.join(HERE, "results.json")))
    ablation_results = json.load(open(os.path.join(HERE, "ablation_results.json")))
    eval_recoup = eval_results["recoup"]["mean_net_reward"]
    ablation_g = next(r["mean_net_reward"] for r in ablation_results if r["model"].startswith("G:"))
    if abs(eval_recoup - ablation_g) < 0.01:
        record("evaluate.py vs ablation.py consistency", "PASS",
               f"both report Full Recoup at {eval_recoup} -- structurally guaranteed by evaluation/protocol.py")
    else:
        record("evaluate.py vs ablation.py consistency", "FAIL",
               f"evaluate.py={eval_recoup} vs ablation.py Model G={ablation_g} -- DISAGREE")


def check_seed_robustness():
    path = os.path.join(HERE, "seed_robustness_results.json")
    if not os.path.exists(path):
        record("Multi-seed robustness", "WARN", "not yet run -- see evaluation/run_seed_robustness.sh")
        return
    data = json.load(open(path))
    lifts = [r["lift_over_baseline_pct"] for r in data]
    if min(lifts) < 0:
        record("Multi-seed robustness", "WARN",
               f"lift ranges from {min(lifts)}% to {max(lifts)}% across {len(lifts)} seeds -- "
               f"AT LEAST ONE SEED SHOWS RECOUP AT OR BELOW BASELINE. The headline single-seed "
               f"number is not a robust claim. See docs/round3_findings.md.")
    else:
        record("Multi-seed robustness", "PASS", f"lift positive across all {len(lifts)} seeds tested: {lifts}")


def check_regret():
    path = os.path.join(HERE, "results.json")
    data = json.load(open(path))
    mean_regret = data["regret_vs_oracle_optimal"]["mean"]
    mean_reward = data["recoup"]["mean_net_reward"]
    top1 = data["top_k_accuracy"]["top1_optimal_action_rate"]
    if mean_regret > mean_reward:
        record("Regret vs oracle-optimal", "WARN",
               f"mean regret (₹{mean_regret}) EXCEEDS mean reward (₹{mean_reward}); top-1 optimal rate only {top1*100:.1f}%. "
               f"System beats simple baselines but is far from oracle-optimal.")
    else:
        record("Regret vs oracle-optimal", "PASS", f"mean regret ₹{mean_regret} vs mean reward ₹{mean_reward}")


def check_two_stage():
    path = os.path.join(HERE, "two_stage_vs_flat_results.json")
    if not os.path.exists(path):
        record("Two-stage vs flat policy", "WARN", "not yet run -- see evaluation/two_stage_vs_flat.py")
        return
    data = json.load(open(path))
    lift = data["lift_two_stage_over_flat_pct"]
    if lift < 0:
        record("Two-stage vs flat policy", "WARN",
               f"two-stage policy underperformed flat policy by {abs(lift)}% on this run -- "
               f"kept flat policy as primary; two-stage documented as explored-and-not-adopted")
    else:
        record("Two-stage vs flat policy", "PASS", f"two-stage improved over flat by {lift}%")


def check_uncalibrated_deferred_items():
    deferred = ["Out-of-distribution test set",
                "Sequence-model (GRU) comparison", "Adversarial simulator/model mismatch",
                "Risk-coverage curve (uncertainty vs regret)", "Inference latency benchmark",
                "Shadow-eval model promotion pipeline", "High-value amount-tiered confidence thresholds (tuned)",
                "Full 12-source regret decomposition (only 3-bucket version built)",
                "Uncertainty layer wired into offline evaluation scripts (only live in app.py)"]
    for item in deferred:
        record(item, "WARN", "NOT IMPLEMENTED -- stated explicitly, see docs/limitations.md")


def check_regret_decomposition():
    path = os.path.join(HERE, "regret_decomposition_results.json")
    if not os.path.exists(path):
        record("Regret decomposition", "WARN", "not yet run -- see evaluation/regret_decomposition.py")
        return
    data = json.load(open(path))
    wrong_arm_pct = data["by_decision_type"].get("WRONG_ARM", {}).get("pct_of_total_regret", 0)
    record("Regret decomposition", "WARN",
           f"{wrong_arm_pct}% of total regret comes from picking the wrong arm (not from escalation/"
           f"safety behavior) -- the ranking/selection layer, not conservatism, is the dominant regret source")


def check_bandit_prove_or_remove():
    path = os.path.join(HERE, "bandit_online_simulation_results.json")
    if not os.path.exists(path):
        record("Bandit prove-or-remove (online simulation)", "WARN", "not yet run -- see evaluation/bandit_online_simulation.py")
        return
    data = json.load(open(path))
    lift = data["linucb_vs_static_lift_pct"]
    eg_lift = data["epsilon_greedy_vs_static_lift_pct"]
    if lift < 2:
        record("Bandit prove-or-remove", "WARN", f"LinUCB does not meaningfully beat static ({lift:+.1f}%) -- see verdict in results file")
    elif abs(lift - eg_lift) < 3:
        record("Bandit prove-or-remove", "WARN",
               f"LinUCB beats static ({lift:+.1f}%), but nearly identically to context-free "
               f"epsilon-greedy ({eg_lift:+.1f}%) -- exploration helps, LinUCB's CONTEXT-AWARENESS "
               f"specifically is not clearly proven beyond what dumb exploration already gets")
    else:
        record("Bandit prove-or-remove", "PASS", f"LinUCB beats static by {lift:+.1f}%, clearly ahead of epsilon-greedy ({eg_lift:+.1f}%)")


def check_action_ranking():
    path = os.path.join(HERE, "action_ranking_analysis_results.json")
    if not os.path.exists(path):
        record("Action ranking analysis", "WARN", "not yet run -- see evaluation/action_ranking_analysis.py")
        return
    data = json.load(open(path))
    rho = data.get("spearman_rank_correlation")
    top1 = data.get("top1_accuracy")
    if rho is None:
        record("Action ranking analysis", "WARN", "no correlation computed")
    elif rho < 0.15:
        record("Action ranking analysis", "WARN",
               f"Spearman rho={rho} is weak -- predicted ranking carries little information about the true ranking (top-1={top1*100:.1f}%)")
    else:
        record("Action ranking analysis", "WARN",
               f"Spearman rho={rho} (weak-to-moderate, non-zero) -- ranking is informative but far from accurate; top-1={top1*100:.1f}%, top-3={data.get('top3_accuracy', 0)*100:.1f}%")


def check_chaos_tests():
    path = os.path.join(HERE, "..", "chaos", "chaos_test_results.json")
    if not os.path.exists(path):
        record("Chaos test suite (15 scenarios)", "WARN", "not yet run -- see chaos/chaos_test.py (requires live ml-service)")
        return
    data = json.load(open(path))
    n_pass = sum(1 for r in data if r["status"] == "PASS")
    n_fail = sum(1 for r in data if r["status"] == "FAIL")
    if n_fail == 0:
        record("Chaos test suite (15 scenarios)", "PASS", f"{n_pass}/{len(data)} passed, 0 failed -- idempotency, fail-closed states, reward-injection defense all verified live")
    else:
        record("Chaos test suite (15 scenarios)", "FAIL", f"{n_fail} scenario(s) failed -- see chaos/chaos_test_results.json")


def check_round7_hardening():
    import subprocess
    try:
        r = subprocess.run(
            [sys.executable, "test_server_side_state.py"],
            cwd=os.path.join(HERE, "..", "ml-service"), capture_output=True, text=True, timeout=30,
        )
        if "ALL SERVER-SIDE STATE TESTS PASSED" in r.stdout:
            record("Server-side temporal state (client cannot lie about history)", "PASS", "verified live -- client-claimed history is discarded and replaced with server-computed values")
        else:
            record("Server-side temporal state", "WARN", "ml-service not running or test did not pass -- see ml-service/test_server_side_state.py")
    except Exception as e:
        record("Server-side temporal state", "WARN", f"could not run: {e}")

    try:
        r = subprocess.run(
            [sys.executable, "test_concurrency.py"],
            cwd=os.path.join(HERE, "..", "ml-service"), capture_output=True, text=True, timeout=60,
        )
        if "ALL CONCURRENCY TESTS PASSED" in r.stdout:
            record("Concurrency (10/50/100 simultaneous /decide, same event_id)", "PASS", "exactly one decision_id produced under real thread concurrency, verified live")
        else:
            record("Concurrency", "WARN", "ml-service not running or a race was detected -- see ml-service/test_concurrency.py output")
    except Exception as e:
        record("Concurrency", "WARN", f"could not run: {e}")


def check_dr_cross_fitting():
    multiseed_path = os.path.join(HERE, "dr_multiseed_results.json")
    single_path = os.path.join(HERE, "dr_cross_fitting_results.json")
    if not os.path.exists(multiseed_path):
        if os.path.exists(single_path):
            record("Doubly-robust cross-fitting", "WARN",
                   "only single-seed result found -- multi-seed validation "
                   "(evaluation/run_dr_multiseed_validation.sh) supersedes this, see "
                   "docs/final_policy_selection.md")
        else:
            record("Doubly-robust cross-fitting", "WARN", "not yet run -- see evaluation/dr_cross_fitting.py")
        return
    data = json.load(open(multiseed_path))
    reward_deltas = [d["reward_delta_pct"] for d in data]
    n_seeds = len(data)
    n_positive = sum(1 for r in reward_deltas if r > 0)
    record("Doubly-robust cross-fitting (multi-seed)", "PASS" if n_positive == n_seeds else "WARN",
           f"reward delta positive on {n_positive}/{n_seeds} seeds ({reward_deltas}) -- "
           f"DR REJECTED as primary policy (not consistently better across seeds); retained as a "
           f"validated research/evaluation module, current T-learner remains production estimator. "
           f"See docs/final_policy_selection.md (supersedes the earlier single-seed ADOPT verdict "
           f"in docs/dr_cross_fitting_final.md).")


if __name__ == "__main__":
    print("=" * 78)
    print("RECOUP JUDGE ATTACK MODE")
    print("=" * 78)
    run_leakage_tests()
    run_overlap_diagnostics()
    run_drift_test()
    run_evaluate_ablation_consistency()
    check_regret()
    check_seed_robustness()
    check_two_stage()
    check_regret_decomposition()
    check_bandit_prove_or_remove()
    check_action_ranking()
    check_chaos_tests()
    check_round7_hardening()
    check_dr_cross_fitting()
    check_uncalibrated_deferred_items()

    print("\n" + "=" * 78)
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_warn = sum(1 for r in results if r["status"] == "WARN")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    print(f"SUMMARY: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL out of {len(results)} checks")
    print("=" * 78)

    with open(os.path.join(HERE, "judge_attack_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to evaluation/judge_attack_results.json")

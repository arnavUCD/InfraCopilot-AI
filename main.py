"""
InfraCopilot AI — Main Pipeline v5
=====================================
End-to-end: generate → train → fleet recommendations → export

Usage:
  python main.py                          # 50k rows, full pipeline
  python main.py --n-samples 5000        # fast dev run (~5s)
  python main.py --skip-gen              # reuse existing CSV
  python main.py --no-plots              # skip matplotlib output
"""

import argparse
import os
import time


def parse_args():
    p = argparse.ArgumentParser(description="InfraCopilot AI v5")
    p.add_argument("--n-samples",    type=int,   default=50_000)
    p.add_argument("--failure-rate", type=float, default=0.03)
    p.add_argument("--data",         type=str,   default="data/charger_data.csv")
    p.add_argument("--skip-gen",     action="store_true", help="Reuse existing CSV")
    p.add_argument("--no-plots",     action="store_true")
    p.add_argument("--top-k",        type=int,   default=10, help="Rows shown in console")
    return p.parse_args()


def main():
    args   = parse_args()
    t_wall = time.perf_counter()
    SEP    = "═" * 64

    from data_generator   import generate_dataset, load_dataset
    from train_model      import train
    from inference_engine import InferenceEngine

    # ── Step 1: Dataset ───────────────────────────────────────────────────────
    print(f"\n{SEP}\n  STEP 1 / 4 — Dataset\n{SEP}")
    if args.skip_gen and os.path.exists(args.data):
        df = load_dataset(args.data)
    else:
        df = generate_dataset(
            n_samples=args.n_samples,
            failure_rate_target=args.failure_rate,
            save_path=args.data,
        )

    # ── Step 2: Train ─────────────────────────────────────────────────────────
    print(f"\n{SEP}\n  STEP 2 / 4 — Train\n{SEP}")
    metadata = train(
        csv_path=args.data,
        n_samples=args.n_samples,
        save_plots=not args.no_plots,
    )

    # ── Step 3: Load inference engine ─────────────────────────────────────────
    print(f"\n{SEP}\n  STEP 3 / 4 — Inference Engine\n{SEP}")
    engine = InferenceEngine()

    # ── Step 4: Fleet Copilot ─────────────────────────────────────────────────
    print(f"\n{SEP}\n  STEP 4 / 4 — Fleet Copilot\n{SEP}")
    fleet = engine.generate_fleet_recommendations(df)
    engine.print_fleet_summary(fleet, top_n=args.top_k)
    engine.save_outputs(fleet)

    # ── Final summary ─────────────────────────────────────────────────────────
    m = metadata["metrics"]
    s = fleet.summary
    elapsed = time.perf_counter() - t_wall

    print(f"\n{SEP}")
    print(f"  PIPELINE COMPLETE")
    print(SEP)
    print(f"  Wall-clock time    : {elapsed:.1f}s")
    print(f"  Dataset            : {len(df):,} rows")
    print(f"  Model              : {metadata['model_name']}  (v{metadata['model_version']})")
    print(f"  Threshold          : {metadata['cost_opt_threshold']}")
    print(f"")
    print(f"  ── Model Performance ──────────────────────────")
    print(f"  ROC-AUC            : {m['roc_auc']:.4f}")
    print(f"  PR-AUC             : {m['pr_auc']:.4f}")
    print(f"  Failure Recall     : {m['failure_recall']:.4f}  ← PRIMARY METRIC")
    print(f"  Failure Precision  : {m['failure_precision']:.4f}")
    print(f"  Failure F1         : {m['failure_f1']:.4f}")
    print(f"  Cost savings       : ${m['cost_savings_usd']:,}")
    print(f"")
    print(f"  ── Fleet ──────────────────────────────────────")
    print(f"  Chargers scored    : {s.total_chargers:,}")
    print(f"  Flagged            : {s.flagged_count:,}")
    print(f"  Critical           : {s.critical_count:,}")
    print(f"  Potential savings  : ${s.total_savings_usd:,.0f}")
    print(f"")
    print(f"  ── Artifacts ──────────────────────────────────")
    print(f"  models_v5/model.joblib")
    print(f"  models_v5/scaler.joblib")
    print(f"  models_v5/metadata.json")
    print(f"  outputs/fleet_recommendations.csv")
    print(f"  outputs/fleet_recommendations.json")
    if not args.no_plots:
        print(f"  outputs/pr_curve.png")
    print(SEP + "\n")


if __name__ == "__main__":
    main()

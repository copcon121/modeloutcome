"""
Backtest sweep for multiple sequence quality models across thresholds.
Reads training_summary_seq_v2.json and evaluates each model on validation set.
"""

import json
from pathlib import Path
from typing import List
import argparse

import os
from phase6_seq_quality.eval_seq_backtest import evaluate_model_on_dataset

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = Path(os.getenv("PHASE6_OUTPUT_DIR", ROOT / "output" / "phase6_seq_quality"))


def sweep_models(summary_path: Path, thresholds: List[float], dataset_path: Path, normalizer_path: Path):
    with open(summary_path, "r") as f:
        summaries = json.load(f)

    results = {}
    for entry in summaries:
        config_name = entry["config"]
        model_path = Path(entry["model_path"])
        print(f"\nEvaluating config: {config_name}")

        by_threshold = []
        best_expectancy = None
        best_record = None

        for th in thresholds:
            metrics = evaluate_model_on_dataset(
                model_path=model_path,
                normalizer_path=normalizer_path,
                dataset_path=dataset_path,
                threshold=th,
            )
            record = {
                "threshold": th,
                "num_trades": metrics["num_trades"],
                "expectancy_R": metrics["expectancy"],
                "max_drawdown_R": metrics["max_dd_r"],
                "winrate": metrics["winrate"],
            }
            by_threshold.append(record)

            if best_expectancy is None or metrics["expectancy"] > best_expectancy:
                best_expectancy = metrics["expectancy"]
                best_record = record

        results[config_name] = {
            "best_threshold": best_record["threshold"] if best_record else None,
            "best_expectancy_R": best_record["expectancy_R"] if best_record else None,
            "best_max_drawdown_R": best_record["max_drawdown_R"] if best_record else None,
            "best_num_trades": best_record["num_trades"] if best_record else None,
            "by_threshold": by_threshold,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Sweep backtest for seq models")
    parser.add_argument("--summary", type=str, default=str(OUTPUT_DIR / "training_summary_seq_v2.json"))
    parser.add_argument("--normalizer", type=str, default=str(OUTPUT_DIR / "normalizer_stats_seq.pt"))
    parser.add_argument("--dataset", type=str, default=str(ROOT / "output" / "phase4_quality" / "dataset_p2_quality_v1_val.pt"))
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--out", type=str, default=str(OUTPUT_DIR / "seq_backtest_summary_v2.json"))
    args = parser.parse_args()

    results = sweep_models(Path(args.summary), args.thresholds, Path(args.dataset), Path(args.normalizer))

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nBacktest sweep completed. Summary:")
    for name, res in results.items():
        print(
            f"\nConfig: {name}\n"
            f"  best_threshold={res['best_threshold']}\n"
            f"  expectancy={res['best_expectancy_R']}R\n"
            f"  maxDD={res['best_max_drawdown_R']}R\n"
            f"  trades={res['best_num_trades']}"
        )
    print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    main()

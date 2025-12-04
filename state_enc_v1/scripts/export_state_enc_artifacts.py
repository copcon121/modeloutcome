#!/usr/bin/env python
"""
Script to export STATE-ENC artifacts for deployment.

Exports:
- state_enc_v1.pt: Model weights
- model_config.json: Model architecture config
- feature_config.json: Feature normalization config

Usage:
    python scripts/export_state_enc_artifacts.py --run-dir artifacts/runs/run_YYYYMMDD_HHMMSS
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch


def find_latest_run(runs_dir: Path) -> Path:
    """Find the most recent run directory"""
    run_dirs = sorted(runs_dir.glob("run_*"))
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {runs_dir}")
    return run_dirs[-1]


def export_artifacts(run_dir: str, output_dir: str, feature_config_path: str = None):
    """
    Export model artifacts for deployment.
    
    Args:
        run_dir: Path to training run directory
        output_dir: Path to output directory (artifacts/final)
        feature_config_path: Path to feature config (optional)
    """
    run_path = Path(run_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting from: {run_path}")
    print(f"Exporting to: {output_path}")
    
    # Find best model checkpoint
    best_model_path = run_path / "best_model.pt"
    if not best_model_path.exists():
        # Try to find any checkpoint
        checkpoints = list(run_path.glob("checkpoint_*.pt"))
        if checkpoints:
            best_model_path = sorted(checkpoints)[-1]
        else:
            raise FileNotFoundError(f"No model checkpoint found in {run_path}")
    
    print(f"Loading checkpoint: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location="cpu")
    
    # Extract model state dict
    if "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
    else:
        model_state = checkpoint
    
    # Save model weights only
    model_output_path = output_path / "state_enc_v1.pt"
    torch.save(model_state, model_output_path)
    print(f"Saved model weights: {model_output_path}")
    
    # Save model config
    if "config" in checkpoint:
        model_config = checkpoint["config"]
    else:
        # Try to load from run directory
        config_path = run_path / "model_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                model_config = json.load(f)
        else:
            print("WARNING: Model config not found in checkpoint")
            model_config = {}
    
    model_config_path = output_path / "model_config.json"
    with open(model_config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"Saved model config: {model_config_path}")
    
    # Copy feature config
    if feature_config_path:
        src_feature_config = Path(feature_config_path)
    else:
        # Default location
        src_feature_config = Path("state_enc_v1/artifacts/feature_config.json")
    
    if src_feature_config.exists():
        dst_feature_config = output_path / "feature_config.json"
        shutil.copy(src_feature_config, dst_feature_config)
        print(f"Copied feature config: {dst_feature_config}")
    else:
        print(f"WARNING: Feature config not found at {src_feature_config}")
    
    # Save export metadata
    metadata = {
        "source_run": str(run_path),
        "source_checkpoint": str(best_model_path),
        "export_timestamp": str(Path(output_path).stat().st_mtime) if output_path.exists() else None,
        "metrics": checkpoint.get("metrics", {})
    }
    
    metadata_path = output_path / "export_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved export metadata: {metadata_path}")
    
    print("\n" + "=" * 50)
    print("EXPORT COMPLETE")
    print("=" * 50)
    print(f"\nArtifacts exported to: {output_path}")
    print("\nFiles:")
    for f in output_path.iterdir():
        print(f"  - {f.name}")
    
    return {
        "model_path": str(model_output_path),
        "model_config_path": str(model_config_path),
        "output_dir": str(output_path)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export STATE-ENC artifacts for deployment"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to training run directory (default: latest run)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="state_enc_v1/artifacts/final",
        help="Output directory for artifacts"
    )
    parser.add_argument(
        "--feature-config",
        type=str,
        default=None,
        help="Path to feature config JSON"
    )
    
    args = parser.parse_args()
    
    # Find run directory
    if args.run_dir:
        run_dir = args.run_dir
    else:
        runs_dir = Path("state_enc_v1/artifacts/runs")
        if not runs_dir.exists():
            print(f"ERROR: Runs directory not found: {runs_dir}")
            return 1
        try:
            run_dir = str(find_latest_run(runs_dir))
            print(f"Using latest run: {run_dir}")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return 1
    
    try:
        export_artifacts(
            run_dir=run_dir,
            output_dir=args.output_dir,
            feature_config_path=args.feature_config
        )
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

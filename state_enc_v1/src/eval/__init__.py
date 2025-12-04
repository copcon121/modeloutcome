"""
Evaluation module for STATE-ENC v1.2
"""

from .semantic_eval import (
    SemanticEvaluator,
    run_future_dir_probe,
    run_regime_probe,
    run_cluster_analysis
)

from .leak_eval import (
    LeakEvaluator,
    check_index_and_future_boundaries,
    check_time_based_split,
    test_label_shuffle_sanity,
    test_future_cheat_upper_bound,
    run_leak_tests
)

__all__ = [
    # Semantic eval
    "SemanticEvaluator",
    "run_future_dir_probe",
    "run_regime_probe", 
    "run_cluster_analysis",
    # Leak eval
    "LeakEvaluator",
    "check_index_and_future_boundaries",
    "check_time_based_split",
    "test_label_shuffle_sanity",
    "test_future_cheat_upper_bound",
    "run_leak_tests"
]

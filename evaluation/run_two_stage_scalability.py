from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from evaluation.benchmark import evaluate_model, relevance_scores_from_interactions, save_results
from cinesense.recommenders.two_stage import CineSenseTwoStage
from evaluation.datasets import (
    ITEM_ID_COL,
    build_eval_users,
    build_positive_interactions,
    filter_users,
    load_anime_catalog,
    load_user_watches,
    split_user_interactions,
)


RESULTS_DIR = Path("evaluation/results")
RUN_SIZES = [10_000, 50_000, None]


def main() -> None:
    setup_start = perf_counter()
    catalog = load_anime_catalog()
    user_watches = load_user_watches()
    positives = build_positive_interactions(
        user_watches,
        catalog_item_ids=catalog[ITEM_ID_COL].unique(),
    )
    filtered_users = filter_users(positives)
    split = split_user_interactions(filtered_users)
    eval_users = build_eval_users(split, use_validation=True)
    relevance_scores_by_user = relevance_scores_from_interactions(split.validation)
    setup_seconds = perf_counter() - setup_start

    fit_start = perf_counter()
    model = CineSenseTwoStage().fit(
        catalog,
        split.train,
        user_ids=[user.user_id for user in eval_users],
    )
    fit_seconds = perf_counter() - fit_start

    reports = []
    print(f"setup_seconds: {setup_seconds:.2f}", flush=True)
    print(f"fit_seconds: {fit_seconds:.2f}", flush=True)
    print(f"eligible_eval_users: {len(eval_users)}", flush=True)

    for run_size in RUN_SIZES:
        selected_users = eval_users if run_size is None else eval_users[:run_size]
        label = "all" if run_size is None else str(run_size)
        start_peak_gb = _peak_memory_gb()
        start_memory_gb = _current_memory_gb()
        run_start = perf_counter()

        results = evaluate_model(
            model,
            selected_users,
            use_validation=True,
            relevance_scores_by_user=relevance_scores_by_user,
        )

        total_seconds = perf_counter() - run_start
        evaluated_users = int(results["evaluated_users"])
        average_latency_ms = (
            (total_seconds / evaluated_users) * 1000 if evaluated_users else 0.0
        )
        current_memory_gb = _current_memory_gb()
        peak_memory_gb = _peak_memory_gb()

        report = {
            "run_size": label,
            "evaluated_users": evaluated_users,
            "total_runtime_seconds": total_seconds,
            "average_recommendation_latency_ms": average_latency_ms,
            "memory_gb": current_memory_gb,
            "peak_memory_gb": peak_memory_gb,
            "peak_memory_delta_gb": max(0.0, peak_memory_gb - start_peak_gb),
            "memory_delta_gb": current_memory_gb - start_memory_gb,
            "metrics": results["metrics"],
        }
        reports.append(report)

        save_results(
            results,
            RESULTS_DIR / f"cinesense_two_stage_scalability_{label}_validation.json",
            model_name=f"cinesense_two_stage_scalability_{label}",
            split_seed=split.seed,
            train_ratio=split.train_ratio,
            val_ratio=split.val_ratio,
            test_ratio=split.test_ratio,
        )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)

    summary_path = RESULTS_DIR / "cinesense_two_stage_scalability_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "setup_seconds": setup_seconds,
                "fit_seconds": fit_seconds,
                "eligible_eval_users": len(eval_users),
                "reports": reports,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"saved summary: {summary_path}", flush=True)


def _current_memory_gb() -> float:
    if sys.platform == "darwin":
        rss_kb = int(
            subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(os.getpid())],
                text=True,
            ).strip()
        )
        return rss_kb / (1024**2)

    page_size = resource.getpagesize()
    with open("/proc/self/statm", encoding="utf-8") as statm:
        resident_pages = int(statm.read().split()[1])
    return resident_pages * page_size / (1024**3)


def _peak_memory_gb() -> float:
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return max_rss / (1024**3)
    return max_rss / (1024**2)


if __name__ == "__main__":
    main()

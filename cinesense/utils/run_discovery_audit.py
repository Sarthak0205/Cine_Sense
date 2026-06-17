import os
import sys
import time
import json
import gc
import numpy as np
import pandas as pd
import psutil
from fastapi.testclient import TestClient

# Ensure cinesense and api are importable
sys.path.insert(0, os.path.abspath("."))

from api.main import app
from cinesense.utils.model_storage import load_model
MODEL_DIR = "cinesense/models/twostage_v1"
from cinesense.services.recommendation import RecommendationService


def run_discovery_perf_audit():
    with TestClient(app) as client:
        # 1. GET /anime/search?q=death
        death_times = []
        for _ in range(100):
            t_start = time.perf_counter()
            response_death = client.get("/anime/search?q=death")
            death_times.append((time.perf_counter() - t_start) * 1000)
        death_payload_size = len(response_death.content)

        # 2. GET /anime/search?q=attack
        attack_times = []
        for _ in range(100):
            t_start = time.perf_counter()
            response_attack = client.get("/anime/search?q=attack")
            attack_times.append((time.perf_counter() - t_start) * 1000)
        attack_payload_size = len(response_attack.content)

        # 3. GET /anime/1535 (Death Note)
        details_times = []
        for _ in range(100):
            t_start = time.perf_counter()
            response_details = client.get("/anime/1535")
            details_times.append((time.perf_counter() - t_start) * 1000)
        details_payload_size = len(response_details.content)

    return {
        "death": {
            "avg": np.mean(death_times),
            "p50": np.percentile(death_times, 50),
            "p95": np.percentile(death_times, 95),
            "payload_bytes": death_payload_size,
        },
        "attack": {
            "avg": np.mean(attack_times),
            "p50": np.percentile(attack_times, 50),
            "p95": np.percentile(attack_times, 95),
            "payload_bytes": attack_payload_size,
        },
        "details": {
            "avg": np.mean(details_times),
            "p50": np.percentile(details_times, 50),
            "p95": np.percentile(details_times, 95),
            "payload_bytes": details_payload_size,
        },
    }


def run_catalog_stats():
    # Load model to access catalog_df
    model, catalog_df, _ = load_model(MODEL_DIR)
    
    total_anime = len(catalog_df)
    
    # Check for non-null/non-empty title_english
    with_eng = catalog_df["title_english"].notna() & (catalog_df["title_english"].str.strip() != "")
    count_with_eng = int(with_eng.sum())
    count_without_eng = total_anime - count_with_eng
    
    return {
        "total_searchable": total_anime,
        "with_english": count_with_eng,
        "without_english": count_without_eng,
    }


def run_startup_impact():
    gc.collect()
    p = psutil.Process()
    mem_before = p.memory_info().rss
    
    start_time = time.perf_counter()
    model, catalog_df, metadata = load_model(MODEL_DIR)
    service = RecommendationService(model, catalog_df)
    total_startup_ms = (time.perf_counter() - start_time) * 1000
    
    gc.collect()
    mem_after = p.memory_info().rss
    rss_growth_mb = (mem_after - mem_before) / (1024 * 1024)
    
    return {
        "startup_ms": total_startup_ms,
        "rss_mb": rss_growth_mb,
    }


def main():
    print("=== DISCOVERY API OPERATIONAL AUDIT ===")
    
    # 1. Performance Measurements
    perf = run_discovery_perf_audit()
    print("\n[Performance Results]")
    for key, stats in perf.items():
        label = f"GET /anime/search?q={key}" if key != "details" else "GET /anime/1535 (details)"
        print(f"{label}:")
        print(f"  Avg Latency: {stats['avg']:.2f} ms")
        print(f"  p50 Latency: {stats['p50']:.2f} ms")
        print(f"  p95 Latency: {stats['p95']:.2f} ms")
        print(f"  Response Size: {stats['payload_bytes']} bytes")

    # 2. Catalog Statistics
    stats = run_catalog_stats()
    print("\n[Catalog Statistics]")
    print(f"Total Searchable Anime: {stats['total_searchable']}")
    print(f"Titles WITH English names: {stats['with_english']}")
    print(f"Titles WITHOUT English names: {stats['without_english']}")

    # 3. Startup Impact
    impact = run_startup_impact()
    print("\n[Startup and Memory Impact]")
    print(f"New Startup Load Time: {impact['startup_ms']:.2f} ms")
    print(f"New RSS Memory Footprint: {impact['rss_mb']:.2f} MB")


if __name__ == "__main__":
    main()

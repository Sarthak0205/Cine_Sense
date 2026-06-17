import os
import sys
import time
import json
import gc
import sys
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
from cinesense.ranking.weighted_b import weighted_max_similarity_to_train_items


def get_deep_size(obj, seen=None):
    """Recursively estimate size of objects in bytes."""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(get_deep_size(k, seen) + get_deep_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        size += sum(get_deep_size(item, seen) for item in obj)
    return size


def run_startup_audit():
    gc.collect()
    p = psutil.Process()
    mem_before = p.memory_info().rss

    start_time = time.perf_counter()

    # 1. Load metadata JSON
    meta_start = time.perf_counter()
    with open(os.path.join(MODEL_DIR, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    meta_time = time.perf_counter() - meta_start

    # 2. Load model numpy assets
    assets_start = time.perf_counter()
    assets = np.load(os.path.join(MODEL_DIR, "model_assets.npz"))
    catalog_embeddings = assets["catalog_embeddings"]
    popularity_scores = assets["popularity_scores"]
    anime_ids = assets["anime_ids"]
    assets_time = time.perf_counter() - assets_start

    # 3. Load catalog Parquet
    parquet_start = time.perf_counter()
    catalog_df = pd.read_parquet(os.path.join(MODEL_DIR, "catalog.parquet"))
    parquet_time = time.perf_counter() - parquet_start

    # 4. Initialize model
    init_start = time.perf_counter()
    hparams = metadata["hyperparameters"]
    from cinesense.recommenders.two_stage import CineSenseTwoStage
    model = CineSenseTwoStage(
        semantic_weight=hparams["semantic_weight"],
        popularity_weight=hparams["popularity_weight"],
        rating_weight_scheme=hparams["rating_weight_scheme"],
        retrieval_candidate_count=hparams["retrieval_candidate_count"],
    )
    model.seed_batch_size = hparams.get("seed_batch_size", 128)
    model.catalog_embeddings = catalog_embeddings
    model.popularity_scores = popularity_scores
    model.anime_ids = anime_ids
    model.item_id_to_index = {
        int(item_id): index for index, item_id in enumerate(anime_ids.tolist())
    }

    # 5. Initialize service
    service = RecommendationService(model, catalog_df)
    init_time = time.perf_counter() - init_start

    total_time = time.perf_counter() - start_time
    gc.collect()
    mem_after = p.memory_info().rss
    rss_growth = mem_after - mem_before

    return {
        "catalog_meta_rows": len(catalog_df),
        "embedding_shape": catalog_embeddings.shape,
        "metadata_load_ms": meta_time * 1000,
        "assets_load_ms": assets_time * 1000,
        "parquet_load_ms": parquet_time * 1000,
        "model_init_ms": init_time * 1000,
        "total_startup_ms": total_time * 1000,
        "rss_growth_mb": rss_growth / (1024 * 1024),
        "service": service,
        "model": model,
        "catalog_df": catalog_df,
    }


def run_latency_and_explanation_audit(service):
    # Select seed IDs present in the model
    all_valid_ids = list(service.recommender.item_id_to_index.keys())
    
    # Seeds combinations: 3, 10, 50
    np.random.seed(42)
    seeds_3 = [int(x) for x in np.random.choice(all_valid_ids, 3, replace=False)]
    seeds_10 = [int(x) for x in np.random.choice(all_valid_ids, 10, replace=False)]
    seeds_50 = [int(x) for x in np.random.choice(all_valid_ids, 50, replace=False)]

    ratings_3 = {int(sid): float(np.random.randint(5, 11)) for sid in seeds_3}
    ratings_10 = {int(sid): float(np.random.randint(5, 11)) for sid in seeds_10}
    ratings_50 = {int(sid): float(np.random.randint(5, 11)) for sid in seeds_50}

    scenarios = [
        ("3 seeds", seeds_3, ratings_3),
        ("10 seeds", seeds_10, ratings_10),
        ("50 seeds", seeds_50, ratings_50),
    ]

    top_ks = [10, 20]
    latency_results = {}

    for name, seeds, ratings in scenarios:
        latency_results[name] = {}
        for k in top_ks:
            # Measure complete recommend()
            times = []
            for _ in range(50):
                t_start = time.perf_counter()
                _ = service.recommend(seeds, ratings, top_k=k)
                times.append((time.perf_counter() - t_start) * 1000)

            # Measure without explanation generation
            times_no_exp = []
            for _ in range(50):
                t_start = time.perf_counter()
                # Run validation
                valid_ids, validated_ratings = service.validate_inputs(seeds, ratings, k)
                # Run recommender
                recommendations = service.recommender.recommend(
                    anime_ids=valid_ids, ratings=validated_ratings, top_k=k
                )
                # Enrich metadata ONLY (no similarity explanations search)
                enriched_no_exp = []
                for rec_id in recommendations:
                    meta = service.catalog_meta.get(rec_id, {})
                    enriched_no_exp.append({
                        "anime_id": rec_id,
                        "title": meta.get("title", ""),
                        "title_english": meta.get("title_english"),
                        "score": 0.0,
                    })
                times_no_exp.append((time.perf_counter() - t_start) * 1000)

            latency_results[name][k] = {
                "avg": np.mean(times),
                "p50": np.percentile(times, 50),
                "p95": np.percentile(times, 95),
                "max": np.max(times),
                "avg_no_exp": np.mean(times_no_exp),
                "p50_no_exp": np.percentile(times_no_exp, 50),
                "p95_no_exp": np.percentile(times_no_exp, 95),
                "max_no_exp": np.max(times_no_exp),
            }

    return latency_results


def run_memory_audit(service, catalog_df):
    model = service.recommender

    # Catalog DataFrame memory
    df_mem_mb = catalog_df.memory_usage(deep=True).sum() / (1024 * 1024)

    # NumPy arrays
    emb_mem_mb = model.catalog_embeddings.nbytes / (1024 * 1024)
    pop_mem_mb = model.popularity_scores.nbytes / (1024 * 1024)
    ids_mem_mb = model.anime_ids.nbytes / (1024 * 1024)

    # Lookup dictionaries
    item_to_idx_bytes = get_deep_size(model.item_id_to_index)
    catalog_meta_bytes = get_deep_size(service.catalog_meta)
    
    idx_mem_mb = (item_to_idx_bytes + catalog_meta_bytes) / (1024 * 1024)

    # RecommendationService footprint (estimated)
    service_mem_mb = get_deep_size(service) / (1024 * 1024)

    return {
        "catalog_meta_df_mb": df_mem_mb,
        "embeddings_mb": emb_mem_mb,
        "popularity_mb": pop_mem_mb,
        "ids_mb": ids_mem_mb,
        "lookup_indexes_mb": idx_mem_mb,
        "service_footprint_mb": service_mem_mb,
    }


def run_api_audit():
    # Setup test client with lifespan loaded
    with TestClient(app) as client:
        # GET /health
        health_times = []
        for _ in range(50):
            t_start = time.perf_counter()
            response = client.get("/health")
            health_times.append((time.perf_counter() - t_start) * 1000)
        health_payload_size = len(response.content)

        # POST /recommend
        all_valid_ids = list(app.state.recommendation_service.recommender.item_id_to_index.keys())
        np.random.seed(42)
        seeds = [int(sid) for sid in np.random.choice(all_valid_ids, 10, replace=False)]
        ratings = {int(sid): float(np.random.randint(5, 11)) for sid in seeds}
        
        payload = {
            "anime_ids": seeds,
            "ratings": ratings,
            "top_k": 10,
        }

        rec_times = []
        for _ in range(50):
            t_start = time.perf_counter()
            response = client.post("/recommend", json=payload)
            rec_times.append((time.perf_counter() - t_start) * 1000)
        rec_payload_size = len(response.content)

    return {
        "health_avg_ms": np.mean(health_times),
        "health_p50_ms": np.percentile(health_times, 50),
        "health_p95_ms": np.percentile(health_times, 95),
        "health_payload_bytes": health_payload_size,
        "recommend_avg_ms": np.mean(rec_times),
        "recommend_p50_ms": np.percentile(rec_times, 50),
        "recommend_p95_ms": np.percentile(rec_times, 95),
        "recommend_payload_bytes": rec_payload_size,
    }


def main():
    print("=== CINESENSE OPERATIONAL AUDIT ===", flush=True)
    
    # 1. Startup Audit
    print("\nRunning Startup Audit...", flush=True)
    startup = run_startup_audit()
    print(f"Catalog Metadata Rows: {startup['catalog_meta_rows']}")
    print(f"Embedding Array Shape: {startup['embedding_shape']}")
    print(f"Metadata load time: {startup['metadata_load_ms']:.2f} ms")
    print(f"NumPy arrays load time: {startup['assets_load_ms']:.2f} ms")
    print(f"Parquet load time: {startup['parquet_load_ms']:.2f} ms")
    print(f"Model/Service init time: {startup['model_init_ms']:.2f} ms")
    print(f"Total Startup load time: {startup['total_startup_ms']:.2f} ms")
    print(f"RSS Memory Growth: {startup['rss_growth_mb']:.2f} MB")

    # 2. Memory Audit
    print("\nRunning Memory Audit...", flush=True)
    memory = run_memory_audit(startup["service"], startup["catalog_df"])
    print(f"Catalog metadata DataFrame memory: {memory['catalog_meta_df_mb']:.2f} MB")
    print(f"Embedding matrix memory: {memory['embeddings_mb']:.2f} MB")
    print(f"Popularity vectors memory: {memory['popularity_mb']:.2f} MB")
    print(f"Anime IDs vector memory: {memory['ids_mb']:.2f} MB")
    print(f"Lookup indexes dictionary memory: {memory['lookup_indexes_mb']:.2f} MB")
    print(f"RecommendationService class overhead: {memory['service_footprint_mb']:.2f} MB")
    
    # 3. Latency & Explanation Cost Audit
    print("\nRunning Recommendation Latency & Explanation Audit...", flush=True)
    latencies = run_latency_and_explanation_audit(startup["service"])
    for name, top_k_data in latencies.items():
        print(f"\nScenario: {name}")
        for k, stats in top_k_data.items():
            print(f"  top_k={k}:")
            print(f"    Complete recommend(): avg={stats['avg']:.2f}ms, p50={stats['p50']:.2f}ms, p95={stats['p95']:.2f}ms, max={stats['max']:.2f}ms")
            print(f"    Recommend (No Exps): avg={stats['avg_no_exp']:.2f}ms, p50={stats['p50_no_exp']:.2f}ms, p95={stats['p95_no_exp']:.2f}ms, max={stats['max_no_exp']:.2f}ms")
            overhead = stats['avg'] - stats['avg_no_exp']
            overhead_pct = (overhead / stats['avg_no_exp']) * 100 if stats['avg_no_exp'] else 0
            print(f"    Explanation Overhead: {overhead:.2f}ms ({overhead_pct:.1f}%)")

    # 4. API Audit
    print("\nRunning API HTTP Endpoint Audit...", flush=True)
    api = run_api_audit()
    print("GET /health:")
    print(f"  Latency: avg={api['health_avg_ms']:.2f}ms, p50={api['health_p50_ms']:.2f}ms, p95={api['health_p95_ms']:.2f}ms")
    print(f"  Payload Size: {api['health_payload_bytes']} bytes")
    print("POST /recommend (10 seeds, top_k=10):")
    print(f"  Latency: avg={api['recommend_avg_ms']:.2f}ms, p50={api['recommend_p50_ms']:.2f}ms, p95={api['recommend_p95_ms']:.2f}ms")
    print(f"  Payload Size: {api['recommend_payload_bytes']} bytes")


if __name__ == "__main__":
    main()

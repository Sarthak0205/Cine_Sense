import os
import unittest
import tempfile
import shutil
import zlib
import numpy as np
import pandas as pd
from unittest.mock import patch

from cinesense.recommenders.two_stage import CineSenseTwoStage
from cinesense.services.recommendation import RecommendationService
from cinesense.utils.model_storage import load_model, save_model


class TestGraphRerank(unittest.TestCase):
    def setUp(self):
        # 1. Instantiate a mock CineSenseTwoStage recommender
        self.recommender = CineSenseTwoStage()
        self.recommender.semantic_weight = 0.85
        self.recommender.popularity_weight = 0.15
        self.recommender.retrieval_candidate_count = 10
        self.recommender.seed_batch_size = 128

        # Setup items: IDs 101, 102, 103, 104, 105
        # 101: Seed
        # 102: Candidate 1 (strong cosine, has Jaccard neighbor, dist 1, high popularity)
        # 103: Candidate 2 (moderate cosine, has Jaccard neighbor, dist 2, moderate popularity)
        # 104: Candidate 3 (weak cosine, no Jaccard, disconnected, low popularity)
        # 105: Candidate 4 (different franchise, etc.)
        self.recommender.anime_ids = np.array([101, 102, 103, 104, 105], dtype=np.int32)
        self.recommender.item_id_to_index = {101: 0, 102: 1, 103: 2, 104: 3, 105: 4}
        
        # Pre-normalized embeddings (dims=2)
        self.recommender.catalog_embeddings = np.array([
            [1.0, 0.0],          # 101 (Seed)
            [0.9, 0.43589],      # 102 (Cosine with 101 = 0.9)
            [0.7, 0.71414],      # 103 (Cosine with 101 = 0.7)
            [0.3, 0.95394],      # 104 (Cosine with 101 = 0.3)
            [0.1, 0.99499],      # 105 (Cosine with 101 = 0.1)
        ], dtype=np.float32)

        # Popularity scores
        self.recommender.popularity_scores = np.array([0.5, 0.99, 0.96, 0.1, 0.2], dtype=np.float32)
        self.recommender.pop_percentiles = np.array([0.4, 0.98, 0.96, 0.1, 0.2], dtype=np.float32)

        # Graph assets setup
        self.recommender.graph_available = True
        self.recommender.supported_anime_ids = np.array([101, 102, 103, 104, 105], dtype=np.int32)
        self.recommender.anime_to_idx = {101: 0, 102: 1, 103: 2, 104: 3, 105: 4}
        self.recommender.col_sums = np.array([1000, 500, 400, 200, 100], dtype=np.int32)

        # neighbor_ids (sorted for binary search)
        self.recommender.neighbor_ids = np.array([
            [102, 103],  # 101 (neighbors: 102, 103)
            [101, 103],  # 102
            [101, 102],  # 103
            [],          # 104
            []           # 105
        ], dtype=object)

        self.recommender.neighbor_jaccards = np.array([
            [0.25, 0.15], # 101
            [0.25, 0.10], # 102
            [0.15, 0.10], # 103
            [],
            []
        ], dtype=object)

        # distance_lookup (2D matrix of shape 5x5, 0 represents disconnected, or distance values)
        # distance 1 between: (101, 102), (101, 103)
        # distance 2 between: (102, 103)
        dist_mat = np.zeros((5, 5), dtype=np.int8)
        dist_mat[0, 1] = dist_mat[1, 0] = 1
        dist_mat[0, 2] = dist_mat[2, 0] = 1
        dist_mat[1, 2] = dist_mat[2, 1] = 2
        self.recommender.distance_lookup = dist_mat

        # Catalog DataFrame
        self.catalog_df = pd.DataFrame({
            "anime_id": [101, 102, 103, 104, 105],
            "title": ["Anime Seed One", "Anime Neighbor A", "Anime Neighbor B", "Anime Far away", "Anime Different"],
            "title_english": ["Seed One", "Neighbor A", "Neighbor B", "Far away", "Different"],
        })

        self.service = RecommendationService(self.recommender, self.catalog_df)

    def test_lookup_jaccard(self):
        self.assertAlmostEqual(self.service._lookup_jaccard(101, 102), 0.25)
        self.assertEqual(self.service._lookup_jaccard(101, 104), 0.0)

    def test_lookup_distance(self):
        self.assertEqual(self.service._lookup_distance(101, 102), 1)
        self.assertEqual(self.service._lookup_distance(102, 103), 2)
        self.assertEqual(self.service._lookup_distance(101, 104), 10)

    def test_safety_guards_cosine_out_of_bounds(self):
        # Cosine > 1.0 or < -1.0 should fallback to raw semantic score (disabling graph contribution)
        self.recommender.rerank_enabled = True
        
        with patch('cinesense.services.recommendation.hybrid_c_retrieval_scores') as mock_ret_scores, \
             patch('cinesense.services.recommendation.top_retrieval_indices') as mock_top_idx, \
             patch('cinesense.services.recommendation.weighted_max_similarity_to_train_items') as mock_sem_scores, \
             patch('cinesense.services.recommendation.rerank_candidates') as mock_rerank_cand:

            mock_ret_scores.return_value = np.zeros(5, dtype=np.float32)
            mock_top_idx.return_value = [1]
            mock_sem_scores.return_value = np.array([1.0, 0.9, 0.7, 0.3, 0.1], dtype=np.float32)
            mock_rerank_cand.return_value = [102]

            # Force an invalid cosine in embeddings
            self.recommender.catalog_embeddings[1] = np.array([2.5, 0.0], dtype=np.float32) # Dot product with 101 = 2.5 (Invalid!)

            res = self.service.recommend([101], mode="discover", user_id="user_test", top_k=1)
            # Rerank score should equal raw semantic score (0.85 * 0.9 + 0.15 * 0.99 = 0.9135) instead of reranked value
            self.assertAlmostEqual(res[0]["score"], 0.9135, places=4)

    def test_safety_guards_jaccard_out_of_bounds(self):
        # Jaccard out of range (>1.0) should fallback to raw semantic score
        self.recommender.rerank_enabled = True
        
        # Inject invalid neighbor Jaccard
        self.recommender.neighbor_jaccards[0] = [2.5, 0.15]
        
        with patch('cinesense.services.recommendation.hybrid_c_retrieval_scores') as mock_ret_scores, \
             patch('cinesense.services.recommendation.top_retrieval_indices') as mock_top_idx, \
             patch('cinesense.services.recommendation.weighted_max_similarity_to_train_items') as mock_sem_scores, \
             patch('cinesense.services.recommendation.rerank_candidates') as mock_rerank_cand:

            mock_ret_scores.return_value = np.zeros(5, dtype=np.float32)
            mock_top_idx.return_value = [1]
            mock_sem_scores.return_value = np.array([1.0, 0.9, 0.7, 0.3, 0.1], dtype=np.float32)
            mock_rerank_cand.return_value = [102]

            res = self.service.recommend([101], mode="discover", user_id="user_test", top_k=1)
            self.assertAlmostEqual(res[0]["score"], 0.9135, places=4)

    def test_safety_guards_distance_out_of_bounds(self):
        # Distance lookup returning an invalid distance value should trigger fallback
        self.recommender.rerank_enabled = True
        
        # Inject invalid distance (e.g. 5)
        self.recommender.distance_lookup[0, 1] = 5
        
        with patch('cinesense.services.recommendation.hybrid_c_retrieval_scores') as mock_ret_scores, \
             patch('cinesense.services.recommendation.top_retrieval_indices') as mock_top_idx, \
             patch('cinesense.services.recommendation.weighted_max_similarity_to_train_items') as mock_sem_scores, \
             patch('cinesense.services.recommendation.rerank_candidates') as mock_rerank_cand:

            mock_ret_scores.return_value = np.zeros(5, dtype=np.float32)
            mock_top_idx.return_value = [1]
            mock_sem_scores.return_value = np.array([1.0, 0.9, 0.7, 0.3, 0.1], dtype=np.float32)
            mock_rerank_cand.return_value = [102]

            res = self.service.recommend([101], mode="discover", user_id="user_test", top_k=1)
            # Rerank score should equal raw semantic score (0.9135) since distance is invalid
            self.assertAlmostEqual(res[0]["score"], 0.9135, places=4)

    def test_ab_testing_routing(self):
        self.recommender.rerank_enabled = True
        
        # 1. 0% traffic should route to Control (no reranking explanations)
        with patch.dict(os.environ, {"CINESENSE_RERANK_TRAFFIC_PERCENT": "0", "CINESENSE_RERANK_ENABLED": "True"}), \
             patch('cinesense.services.recommendation.hybrid_c_retrieval_scores') as mock_ret_scores, \
             patch('cinesense.services.recommendation.top_retrieval_indices') as mock_top_idx, \
             patch('cinesense.services.recommendation.weighted_max_similarity_to_train_items') as mock_sem_scores, \
             patch('cinesense.services.recommendation.rerank_candidates') as mock_rerank_cand:

            from cinesense.config.graph_rerank import GraphRerankConfig
            self.service.rerank_config = GraphRerankConfig.from_env()

            mock_ret_scores.return_value = np.zeros(5, dtype=np.float32)
            mock_top_idx.return_value = [1]
            mock_sem_scores.return_value = np.array([1.0, 0.9, 0.7, 0.3, 0.1], dtype=np.float32)
            mock_rerank_cand.return_value = [102]

            res = self.service.recommend([101], mode="discover", user_id="user_1", top_k=1)
            self.assertEqual(res[0]["explanation"]["reasons"], ["High semantic similarity"])

        # 2. 100% traffic should route to Treatment
        with patch.dict(os.environ, {"CINESENSE_RERANK_TRAFFIC_PERCENT": "100", "CINESENSE_RERANK_ENABLED": "True"}), \
             patch('cinesense.services.recommendation.hybrid_c_retrieval_scores') as mock_ret_scores, \
             patch('cinesense.services.recommendation.top_retrieval_indices') as mock_top_idx, \
             patch('cinesense.services.recommendation.weighted_max_similarity_to_train_items') as mock_sem_scores, \
             patch('cinesense.services.recommendation.rerank_candidates') as mock_rerank_cand:

            from cinesense.config.graph_rerank import GraphRerankConfig
            self.service.rerank_config = GraphRerankConfig.from_env()

            mock_ret_scores.return_value = np.zeros(5, dtype=np.float32)
            mock_top_idx.return_value = [1]
            mock_sem_scores.return_value = np.array([1.0, 0.9, 0.7, 0.3, 0.1], dtype=np.float32)
            mock_rerank_cand.return_value = [102]

            res = self.service.recommend([101], mode="discover", user_id="user_1", top_k=1)
            # Reranking reasons generated dynamically
            self.assertIn("reasons", res[0]["explanation"])
            self.assertNotEqual(res[0]["explanation"]["reasons"], ["High semantic similarity"])

        # 3. 25% split: user_1 crc32 % 100 = 80 -> Control; user_7 crc32 % 100 = 17 -> Treatment
        # user_1: zlib.crc32(b"user_1") % 100 = 80
        # user_7: zlib.crc32(b"user_7") % 100 = 17
        self.assertEqual(zlib.crc32(b"user_1") % 100, 80)
        self.assertEqual(zlib.crc32(b"user_7") % 100, 17)

        with patch.dict(os.environ, {"CINESENSE_RERANK_TRAFFIC_PERCENT": "25", "CINESENSE_RERANK_ENABLED": "True"}), \
             patch('cinesense.services.recommendation.hybrid_c_retrieval_scores') as mock_ret_scores, \
             patch('cinesense.services.recommendation.top_retrieval_indices') as mock_top_idx, \
             patch('cinesense.services.recommendation.weighted_max_similarity_to_train_items') as mock_sem_scores, \
             patch('cinesense.services.recommendation.rerank_candidates') as mock_rerank_cand:

            from cinesense.config.graph_rerank import GraphRerankConfig
            self.service.rerank_config = GraphRerankConfig.from_env()

            mock_ret_scores.return_value = np.zeros(5, dtype=np.float32)
            mock_top_idx.return_value = [1]
            mock_sem_scores.return_value = np.array([1.0, 0.9, 0.7, 0.3, 0.1], dtype=np.float32)
            mock_rerank_cand.return_value = [102]

            # user_1 gets Control
            res_ctrl = self.service.recommend([101], mode="discover", user_id="user_1", top_k=1)
            self.assertEqual(res_ctrl[0]["explanation"]["reasons"], ["High semantic similarity"])

            # user_7 gets Treatment
            res_treat = self.service.recommend([101], mode="discover", user_id="user_7", top_k=1)
            self.assertNotEqual(res_treat[0]["explanation"]["reasons"], ["High semantic similarity"])

    def test_explainability_summary_and_reasons(self):
        # Test explainability text rules based on metric bounds
        self.recommender.rerank_enabled = True
        self.recommender.cosine_power = 2
        self.recommender.popularity_penalty = 0.05

        with patch.dict(os.environ, {"CINESENSE_RERANK_TRAFFIC_PERCENT": "100", "CINESENSE_RERANK_ENABLED": "True"}), \
             patch('cinesense.services.recommendation.hybrid_c_retrieval_scores') as mock_ret_scores, \
             patch('cinesense.services.recommendation.top_retrieval_indices') as mock_top_idx, \
             patch('cinesense.services.recommendation.weighted_max_similarity_to_train_items') as mock_sem_scores, \
             patch('cinesense.services.recommendation.rerank_candidates') as mock_rerank_cand:

            from cinesense.config.graph_rerank import GraphRerankConfig
            self.service.rerank_config = GraphRerankConfig.from_env()

            mock_ret_scores.return_value = np.zeros(5, dtype=np.float32)
            mock_top_idx.return_value = [1]
            mock_sem_scores.return_value = np.array([1.0, 0.9, 0.7, 0.3, 0.1], dtype=np.float32)
            mock_rerank_cand.return_value = [102]

            res = self.service.recommend([101], mode="discover", user_id="user_12", top_k=1)
            
            # For 102: cosine=0.9 (>=0.60), Jaccard=0.25 (>=0.10), distance=1 (in {1,2})
            # Expected reasons: "High semantic similarity", "Strong co-watch overlap", "Frequently watched by similar users"
            # Expected summary: "Recommended because it is semantically similar, frequently co-watched, and watched by similar users."
            exp = res[0]["explanation"]
            self.assertEqual(len(exp["reasons"]), 3)
            self.assertIn("High semantic similarity", exp["reasons"])
            self.assertIn("Strong co-watch overlap", exp["reasons"])
            self.assertIn("Frequently watched by similar users", exp["reasons"])
            self.assertIn("Recommended because it is semantically similar, frequently co-watched, and watched by similar users.", exp["summary"])

    def test_asset_validation_invalid_dimensions(self):
        # Mismatched neighbor_ids and neighbor_jaccards shapes
        temp_dir = tempfile.mkdtemp()
        try:
            save_model(self.recommender, self.catalog_df, temp_dir, "mock_v1", "cat_v1", "emb_v1")
            
            # Write bad dimensions
            np.savez_compressed(
                os.path.join(temp_dir, "graph_assets.npz"),
                neighbor_ids=np.zeros((5, 10), dtype=np.int32),
                neighbor_jaccards=np.zeros((5, 12), dtype=np.float32),  # shape mismatch!
                distance_lookup=self.recommender.distance_lookup,
                supported_anime_ids=self.recommender.supported_anime_ids,
                col_sums=self.recommender.col_sums,
                graph_version=np.array("v1")
            )
            model, _, _ = load_model(temp_dir)
            self.assertFalse(model.graph_available)
        finally:
            shutil.rmtree(temp_dir)

    def test_asset_validation_duplicate_ids(self):
        # Duplicate IDs in supported_anime_ids should trigger validation failure
        temp_dir = tempfile.mkdtemp()
        try:
            save_model(self.recommender, self.catalog_df, temp_dir, "mock_v1", "cat_v1", "emb_v1")
            
            np.savez_compressed(
                os.path.join(temp_dir, "graph_assets.npz"),
                neighbor_ids=np.zeros((5, 200), dtype=np.int32),
                neighbor_jaccards=np.zeros((5, 200), dtype=np.float32),
                distance_lookup=self.recommender.distance_lookup,
                supported_anime_ids=np.array([101, 101, 103, 104, 105], dtype=np.int32),  # duplicates!
                col_sums=self.recommender.col_sums,
                graph_version=np.array("v1")
            )
            model, _, _ = load_model(temp_dir)
            self.assertFalse(model.graph_available)
        finally:
            shutil.rmtree(temp_dir)

    def test_asset_validation_unsorted_rows(self):
        # Rows in neighbor_ids not sorted ascending should trigger validation failure
        temp_dir = tempfile.mkdtemp()
        try:
            save_model(self.recommender, self.catalog_df, temp_dir, "mock_v1", "cat_v1", "emb_v1")
            
            unsorted_row = np.array([130, 120, 110] + [2147483647]*197, dtype=np.int32)
            neighbor_ids_bad = np.zeros((5, 200), dtype=np.int32)
            neighbor_ids_bad[0] = unsorted_row  # unsorted!
            
            np.savez_compressed(
                os.path.join(temp_dir, "graph_assets.npz"),
                neighbor_ids=neighbor_ids_bad,
                neighbor_jaccards=np.zeros((5, 200), dtype=np.float32),
                distance_lookup=self.recommender.distance_lookup,
                supported_anime_ids=self.recommender.supported_anime_ids,
                col_sums=self.recommender.col_sums,
                graph_version=np.array("v1")
            )
            model, _, _ = load_model(temp_dir)
            self.assertFalse(model.graph_available)
        finally:
            shutil.rmtree(temp_dir)

    def test_asset_validation_nan_jaccard(self):
        # NaNs inside neighbor_jaccards should trigger validation failure
        temp_dir = tempfile.mkdtemp()
        try:
            save_model(self.recommender, self.catalog_df, temp_dir, "mock_v1", "cat_v1", "emb_v1")
            
            jaccards_bad = np.zeros((5, 200), dtype=np.float32)
            jaccards_bad[0, 0] = np.nan  # NaN!
            
            np.savez_compressed(
                os.path.join(temp_dir, "graph_assets.npz"),
                neighbor_ids=np.zeros((5, 200), dtype=np.int32),
                neighbor_jaccards=jaccards_bad,
                distance_lookup=self.recommender.distance_lookup,
                supported_anime_ids=self.recommender.supported_anime_ids,
                col_sums=self.recommender.col_sums,
                graph_version=np.array("v1")
            )
            model, _, _ = load_model(temp_dir)
            self.assertFalse(model.graph_available)
        finally:
            shutil.rmtree(temp_dir)

    def test_asset_validation_version_mismatch(self):
        # graph_version != "v1" should trigger validation failure
        temp_dir = tempfile.mkdtemp()
        try:
            save_model(self.recommender, self.catalog_df, temp_dir, "mock_v1", "cat_v1", "emb_v1")
            
            np.savez_compressed(
                os.path.join(temp_dir, "graph_assets.npz"),
                neighbor_ids=np.zeros((5, 200), dtype=np.int32),
                neighbor_jaccards=np.zeros((5, 200), dtype=np.float32),
                distance_lookup=self.recommender.distance_lookup,
                supported_anime_ids=self.recommender.supported_anime_ids,
                col_sums=self.recommender.col_sums,
                graph_version=np.array("v2")  # wrong version!
            )
            model, _, _ = load_model(temp_dir)
            self.assertFalse(model.graph_available)
        finally:
            shutil.rmtree(temp_dir)

    def test_ab_telemetry_counters(self):
        # Verify that every request increments exactly one counter: ab_control_requests OR ab_treatment_requests.
        class DummyTelemetry:
            def __init__(self):
                self.ab_control_requests = 0
                self.ab_treatment_requests = 0

        telemetry = DummyTelemetry()
        service = RecommendationService(self.recommender, self.catalog_df, telemetry=telemetry)
        
        # 1. Treatment request (mode="discover", rerank_enabled=True, user_id="user_7" routes to treatment bucket 17)
        self.recommender.rerank_enabled = True
        service.rerank_config.rerank_enabled = True
        service.rerank_config.traffic_percent = 25
        
        with patch('cinesense.services.recommendation.hybrid_c_retrieval_scores') as mock_ret_scores, \
             patch('cinesense.services.recommendation.top_retrieval_indices') as mock_top_idx, \
             patch('cinesense.services.recommendation.weighted_max_similarity_to_train_items') as mock_sem_scores, \
             patch('cinesense.services.recommendation.rerank_candidates') as mock_rerank_cand:

            mock_ret_scores.return_value = np.zeros(5, dtype=np.float32)
            mock_top_idx.return_value = [1]
            mock_sem_scores.return_value = np.array([1.0, 0.9, 0.7, 0.3, 0.1], dtype=np.float32)
            mock_rerank_cand.return_value = [102]

            # Trigger recommendation
            service.recommend([101], mode="discover", user_id="user_7", top_k=1)
            
            self.assertEqual(telemetry.ab_treatment_requests, 1)
            self.assertEqual(telemetry.ab_control_requests, 0)

            # 2. Control request due to user_id bucket (user_1 routes to 80, which is > 25)
            service.recommend([101], mode="discover", user_id="user_1", top_k=1)
            self.assertEqual(telemetry.ab_treatment_requests, 1)
            self.assertEqual(telemetry.ab_control_requests, 1)

            # 3. Control request due to mode="similar"
            service.recommend([101], mode="similar", user_id="user_7", top_k=1)
            self.assertEqual(telemetry.ab_treatment_requests, 1)
            self.assertEqual(telemetry.ab_control_requests, 2)

            # 4. Control request due to rerank_enabled = False
            service.rerank_config.rerank_enabled = False
            service.recommend([101], mode="discover", user_id="user_7", top_k=1)
            self.assertEqual(telemetry.ab_treatment_requests, 1)
            self.assertEqual(telemetry.ab_control_requests, 3)

            # 5. Control request due to user_id = None
            service.rerank_config.rerank_enabled = True
            service.recommend([101], mode="discover", user_id=None, top_k=1)
            self.assertEqual(telemetry.ab_treatment_requests, 1)
            self.assertEqual(telemetry.ab_control_requests, 4)


if __name__ == "__main__":
    unittest.main()

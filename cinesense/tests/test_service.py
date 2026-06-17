import os
import unittest
import numpy as np
import pandas as pd

from cinesense.recommenders.two_stage import CineSenseTwoStage
from cinesense.services.recommendation import RecommendationService


class TestRecommendationService(unittest.TestCase):
    def setUp(self):
        # Create a mock recommender with simple dummy attributes to avoid encoding raw catalog text
        self.recommender = CineSenseTwoStage()
        self.recommender.semantic_weight = 0.85
        self.recommender.popularity_weight = 0.15
        self.recommender.retrieval_candidate_count = 5
        self.recommender.seed_batch_size = 128

        # Embeddings: 3 items (dims=2), normalized
        self.recommender.catalog_embeddings = np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32
        )
        self.recommender.popularity_scores = np.array([0.9, 0.5, 0.2], dtype=np.float32)
        self.recommender.anime_ids = np.array([101, 102, 103], dtype=np.int32)
        self.recommender.item_id_to_index = {101: 0, 102: 1, 103: 2}

        # Mock catalog metadata
        self.catalog_df = pd.DataFrame(
            {
                "anime_id": [101, 102, 103],
                "title": ["Anime A", "Anime B", "Anime C"],
                "title_english": ["English Anime A", None, "English Anime C"],
            }
        )

        self.service = RecommendationService(self.recommender, self.catalog_df)

    def test_validate_inputs_success(self):
        anime_ids = [101, 102]
        ratings = {101: 9.5, 102: 8}
        valid_ids, validated_ratings = self.service.validate_inputs(anime_ids, ratings, top_k=5)

        self.assertEqual(valid_ids, [101, 102])
        self.assertEqual(validated_ratings, {101: 9.5, 102: 8.0})

    def test_validate_inputs_deduplication(self):
        anime_ids = [101, 102, 101, 102]
        valid_ids, _ = self.service.validate_inputs(anime_ids, None)
        self.assertEqual(valid_ids, [101, 102])

    def test_validate_inputs_unknown_ids(self):
        anime_ids = [101, 999, 102]  # 999 is unknown
        valid_ids, _ = self.service.validate_inputs(anime_ids, None)
        self.assertEqual(valid_ids, [101, 102])

    def test_validate_inputs_invalid_types(self):
        with self.assertRaises(TypeError):
            self.service.validate_inputs("not_a_list")

        with self.assertRaises(TypeError):
            self.service.validate_inputs([101, "invalid_id"])

        with self.assertRaises(TypeError):
            self.service.validate_inputs([101], ratings="not_a_dict")

        with self.assertRaises(TypeError):
            self.service.validate_inputs([101], ratings={101: "not_a_number"})

    def test_validate_inputs_out_of_bounds_ratings(self):
        with self.assertRaises(ValueError):
            self.service.validate_inputs([101], ratings={101: 0.5})

        with self.assertRaises(ValueError):
            self.service.validate_inputs([101], ratings={101: 11.0})

    def test_validate_inputs_negative_k(self):
        with self.assertRaises(ValueError):
            self.service.validate_inputs([101], top_k=-1)

    def test_generate_explanations(self):
        # Anime A (101): [1.0, 0.0]
        # Anime B (102): [0.0, 1.0]
        # Anime C (103): [0.7071, 0.7071]
        # Recommended: Anime C (103). Seeds: Anime A (101) & Anime B (102).
        # Dot product with Anime A: 0.7071. Dot product with Anime B: 0.7071.
        # Let's see: both match equally. Let's make Anime C closer to Anime A [0.8, 0.6]
        self.recommender.catalog_embeddings[2] = np.array([0.8, 0.6], dtype=np.float32)
        # Dot product of C with A: 0.8 * 1.0 + 0.6 * 0.0 = 0.8
        # Dot product of C with B: 0.8 * 0.0 + 0.6 * 1.0 = 0.6
        # Therefore, Anime A (101) should be the strongest matching seed!

        explanation = self.service.generate_explanations(recommended_id=103, seed_ids=[101, 102])

        self.assertEqual(explanation["matched_seed"]["anime_id"], 101)
        self.assertEqual(explanation["matched_seed"]["title"], "Anime A")
        self.assertAlmostEqual(explanation["similarity"], 0.8, places=4)
        self.assertAlmostEqual(explanation["popularity"], 0.2, places=4)
        self.assertIn("highly similar to 'Anime A'", explanation["reason"])

    def test_enrich_recommendations(self):
        recs = [101]
        scores = {101: 0.85}
        enriched = self.service.enrich_recommendations(recs, scores, seed_ids=[102])

        self.assertEqual(len(enriched), 1)
        item = enriched[0]
        self.assertEqual(item["anime_id"], 101)
        self.assertEqual(item["title"], "Anime A")
        self.assertEqual(item["title_english"], "English Anime A")
        self.assertEqual(item["score"], 0.85)
        self.assertEqual(item["explanation"]["matched_seed"]["anime_id"], 102)

    def test_recommend_empty_input(self):
        results = self.service.recommend([])
        self.assertEqual(results, [])

    def test_hybrid_aggregation(self):
        # We test that the hybrid aggregator computes 0.7 * weighted_avg + 0.3 * max_sim.
        # Catalog: 3 items (dims=2).
        # Candidate 0: [1.0, 0.0]
        # Candidate 1: [0.0, 1.0]
        # Candidate 2: [0.7071, 0.7071]
        # Seeds: Candidate 0 (weight=1.0) and Candidate 1 (weight=0.5).
        from cinesense.ranking.weighted_b import weighted_max_similarity_to_train_items
        
        train_indices = np.array([0, 1], dtype=np.int32)
        train_weights = np.array([1.0, 0.5], dtype=np.float32)
        
        scores = weighted_max_similarity_to_train_items(
            train_indices, train_weights, self.recommender.catalog_embeddings, seed_batch_size=128
        )
        
        # Calculations for Candidate 2:
        # sim0 = 0.7071, sim1 = 0.7071
        # weighted_avg = (0.7071 * 1.0 + 0.7071 * 0.5) / 1.5 = (0.7071 * 1.5) / 1.5 = 0.7071
        # max_sim = 0.7071
        # hybrid = 0.7 * 0.7071 + 0.3 * 0.7071 = 0.7071
        self.assertAlmostEqual(scores[2], 0.7071, places=4)

    def test_linear_rating_weights(self):
        # Test linear rating weight mapping (e.g. 10 -> 1.0, 8 -> 0.8)
        self.assertEqual(self.recommender._rating_weight(10), 1.0)
        self.assertEqual(self.recommender._rating_weight(9), 0.9)
        self.assertEqual(self.recommender._rating_weight(8), 0.8)
        self.assertEqual(self.recommender._rating_weight(7), 0.7)

    def test_is_sequel_title(self):
        # Test sequel detection rules
        self.assertTrue(self.service.is_sequel_title("Zero no Tsukaima Season 2"))
        self.assertTrue(self.service.is_sequel_title("Attack on Titan Season 2"))
        self.assertTrue(self.service.is_sequel_title("Attack on Titan Part II"))
        self.assertTrue(self.service.is_sequel_title("Durarara!!x2"))
        self.assertTrue(self.service.is_sequel_title("Code Geass R2"))
        self.assertTrue(self.service.is_sequel_title("Anime Series 3"))
        self.assertTrue(self.service.is_sequel_title("Zero no Tsukaima S2"))
        
        self.assertFalse(self.service.is_sequel_title("Death Note"))
        self.assertFalse(self.service.is_sequel_title("Attack on Titan"))

    def test_discover_mode_franchise_deduplication_and_sequel_filtering(self):
        # We will mock the recommendations returned by the recommender and test the post-processing filter
        # Let's set up a mock catalog with:
        # 201: Anime Alpha
        # 202: Anime Alpha Season 2 (sequel of Anime Alpha)
        # 203: Anime Beta
        # 204: Anime Beta Movie 2 (sequel of Anime Beta)
        # 205: Anime Gamma
        self.recommender.anime_ids = np.array([201, 202, 203, 204, 205], dtype=np.int32)
        self.recommender.item_id_to_index = {201: 0, 202: 1, 203: 2, 204: 3, 205: 4}
        self.recommender.catalog_embeddings = np.array([
            [1.0, 0.0],
            [0.9, 0.43589],
            [0.8, 0.6],
            [0.7, 0.71414],
            [0.6, 0.8]
        ], dtype=np.float32)
        self.recommender.popularity_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
        
        self.service.catalog_meta = {
            201: {"title": "Anime Alpha", "title_english": None, "synopsis": ""},
            202: {"title": "Anime Alpha Season 2", "title_english": None, "synopsis": ""},
            203: {"title": "Anime Beta", "title_english": None, "synopsis": ""},
            204: {"title": "Anime Beta Movie 2", "title_english": None, "synopsis": ""},
            205: {"title": "Anime Gamma", "title_english": None, "synopsis": ""},
        }
        
        # Test Discover Mode:
        # In discover mode, 202 (sequel) and 204 (sequel) should be discarded.
        # If we have 201 and 202 both recommended, 202 is discarded.
        # Also, duplicate franchises should be filtered: only one item per franchise.
        # Let's mock the internal recommend results
        def mock_recommend(anime_ids, ratings, top_k):
            return [201, 202, 203, 204, 205]
        
        self.recommender.recommend = mock_recommend
        
        results = self.service.recommend([205], mode="discover", top_k=5)
        # Seed is 205 (Anime Gamma).
        # We expect only:
        # - 201 (Anime Alpha) - kept
        # - 202 (Anime Alpha Season 2) - sequel, discarded
        # - 203 (Anime Beta) - kept
        # - 204 (Anime Beta Movie 2) - sequel, discarded
        # (Anime Gamma 205 is excluded as it's the seed franchise)
        # So results should only contain 201 and 203!
        self.assertEqual(len(results), 2)
        recommended_ids = [r["anime_id"] for r in results]
        self.assertEqual(sorted(recommended_ids), [201, 203])

    def test_similar_mode_retains_sequels_and_duplicates(self):
        # In similar mode, filters are bypassed
        self.recommender.anime_ids = np.array([201, 202, 203, 204, 205], dtype=np.int32)
        self.recommender.item_id_to_index = {201: 0, 202: 1, 203: 2, 204: 3, 205: 4}
        self.recommender.catalog_embeddings = np.array([
            [1.0, 0.0],
            [0.9, 0.43589],
            [0.8, 0.6],
            [0.7, 0.71414],
            [0.6, 0.8]
        ], dtype=np.float32)
        self.recommender.popularity_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
        
        self.service.catalog_meta = {
            201: {"title": "Anime Alpha", "title_english": None, "synopsis": ""},
            202: {"title": "Anime Alpha Season 2", "title_english": None, "synopsis": ""},
            203: {"title": "Anime Beta", "title_english": None, "synopsis": ""},
            204: {"title": "Anime Beta Movie 2", "title_english": None, "synopsis": ""},
            205: {"title": "Anime Gamma", "title_english": None, "synopsis": ""},
        }
        
        def mock_recommend(anime_ids, ratings, top_k):
            return [201, 202, 203, 204, 205]
        
        self.recommender.recommend = mock_recommend
        
        results = self.service.recommend([205], mode="similar", top_k=5)
        # In similar mode, all 5 items should be returned (since similar mode doesn't filter sequels or duplicate franchises)
        self.assertEqual(len(results), 5)

    def test_multi_seed_balancing_scenario(self):
        # We test that our hybrid multi-seed aggregation correctly balances multiple seeds.
        # Setup catalog embeddings where items align with different seeds.
        # Seed 0 (index 0, ID 101): [1.0, 0.0]
        # Seed 1 (index 1, ID 102): [0.0, 1.0]
        # Candidate 2 (ID 103): [0.8, 0.6] (closer to Seed 0)
        # Candidate 3 (ID 104): [0.6, 0.8] (closer to Seed 1)
        self.recommender.catalog_embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.6],
            [0.6, 0.8],
        ], dtype=np.float32)
        self.recommender.anime_ids = np.array([101, 102, 103, 104], dtype=np.int32)
        self.recommender.item_id_to_index = {101: 0, 102: 1, 103: 2, 104: 3}
        self.recommender.popularity_scores = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        self.service.catalog_meta = {
            101: {"title": "Seed A", "title_english": None},
            102: {"title": "Seed B", "title_english": None},
            103: {"title": "Match A", "title_english": None},
            104: {"title": "Match B", "title_english": None},
        }

        # Query with multiple seeds: rating 101 -> 10.0 (weight 1.0), 102 -> 9.0 (weight 0.9)
        results = self.service.recommend([101, 102], ratings={101: 10.0, 102: 9.0}, mode="similar", top_k=2)
        
        # Check that both recommendations match different seeds:
        # Match A (103) should match Seed A (101)
        # Match B (104) should match Seed B (102)
        self.assertEqual(len(results), 2)
        exps = [r["explanation"]["matched_seed"]["anime_id"] for r in results]
        
        # Both seeds must be represented in the explanations!
        self.assertIn(101, exps)
        self.assertIn(102, exps)
        
        # Compute matched seed distribution and verify pass condition: largest_seed_share < 0.70
        distribution = {}
        for r in results:
            sid = r["explanation"]["matched_seed"]["anime_id"]
            distribution[sid] = distribution.get(sid, 0) + 1
            
        largest_seed_share = max(distribution.values()) / len(results)
        self.assertTrue(largest_seed_share < 0.70, f"Largest seed share {largest_seed_share} is not < 0.70")

    def test_representation_penalty_balances_seeds(self):
        # Set up a recommender with 2 seeds and 4 candidates
        # DN Seed (101): [1.0, 0.0]
        # SG Seed (102): [0.0, 1.0]
        # DN Candidate 1 (103): [0.99, 0.01] (score = 0.99)
        # DN Candidate 2 (104): [0.98, 0.02] (score = 0.98)
        # SG Candidate 1 (105): [0.02, 0.98] (score = 0.98)
        self.recommender.catalog_embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.02, 0.98],
        ], dtype=np.float32)
        self.recommender.anime_ids = np.array([101, 102, 103, 104, 105], dtype=np.int32)
        self.recommender.item_id_to_index = {101: 0, 102: 1, 103: 2, 104: 3, 105: 4}
        self.recommender.popularity_scores = np.zeros(5, dtype=np.float32)
        
        self.service.catalog_meta = {
            101: {"title": "DN Seed", "title_english": None},
            102: {"title": "SG Seed", "title_english": None},
            103: {"title": "DN Match 1", "title_english": None},
            104: {"title": "DN Match 2", "title_english": None},
            105: {"title": "SG Match 1", "title_english": None},
        }

        # Enable penalty
        self.recommender.representation_penalty = True
        self.recommender.representation_lambda = 0.03

        results = self.service.recommend([101, 102], ratings={101: 10.0, 102: 10.0}, mode="similar", top_k=2)

        # Baseline would recommend 103, 104 (both DN Match).
        # With penalty enabled, selecting 103 first penalizes DN by 0.03.
        # Adjusted scores: 104 (DN) becomes 0.98 * 0.85 - 0.03 = 0.803.
        # Adjusted score for 105 (SG) is 0.98 * 0.85 = 0.833.
        # So 105 (SG Match) must be selected second!
        rec_ids = [r["anime_id"] for r in results]
        self.assertEqual(rec_ids, [103, 105])

    def test_representation_penalty_preserves_order_when_single_seed(self):
        # DN Seed (101): [1.0, 0.0]
        # DN Candidate 1 (103): [0.99, 0.01]
        # DN Candidate 2 (104): [0.98, 0.02]
        self.recommender.catalog_embeddings = np.array([
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
        ], dtype=np.float32)
        self.recommender.anime_ids = np.array([101, 103, 104], dtype=np.int32)
        self.recommender.item_id_to_index = {101: 0, 103: 1, 104: 2}
        self.recommender.popularity_scores = np.zeros(3, dtype=np.float32)
        
        self.service.catalog_meta = {
            101: {"title": "DN Seed", "title_english": None},
            103: {"title": "DN Match 1", "title_english": None},
            104: {"title": "DN Match 2", "title_english": None},
        }

        # Enable penalty on a single seed query
        self.recommender.representation_penalty = True
        results_penalty = self.service.recommend([101], mode="similar", top_k=2)

        # Disable penalty
        self.recommender.representation_penalty = False
        results_baseline = self.service.recommend([101], mode="similar", top_k=2)

        self.assertEqual(results_penalty, results_baseline)

    def test_representation_penalty_does_not_break_discovery(self):
        # Test that Discover Mode filters still execute correctly when penalty is active
        self.recommender.anime_ids = np.array([201, 202, 203, 204], dtype=np.int32)
        self.recommender.item_id_to_index = {201: 0, 202: 1, 203: 2, 204: 3}
        self.recommender.catalog_embeddings = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0]
        ], dtype=np.float32)
        self.recommender.popularity_scores = np.zeros(4, dtype=np.float32)
        
        self.service.catalog_meta = {
            201: {"title": "Anime Alpha", "title_english": None},
            202: {"title": "Anime Alpha Season 2", "title_english": None},
            203: {"title": "Anime Beta", "title_english": None},
            204: {"title": "Anime Alpha Movie", "title_english": None},
        }
        
        self.recommender.representation_penalty = True
        results = self.service.recommend([201], mode="discover", top_k=2)
        
        # 201 is seed, 202 is sequel, 204 is movie.
        # Only 203 (Anime Beta) should survive!
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["anime_id"], 203)

    def test_representation_penalty_disabled_matches_baseline(self):
        # Setup multiseed data
        self.recommender.catalog_embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.02, 0.98],
        ], dtype=np.float32)
        self.recommender.anime_ids = np.array([101, 102, 103, 104, 105], dtype=np.int32)
        self.recommender.item_id_to_index = {101: 0, 102: 1, 103: 2, 104: 3, 105: 4}
        self.recommender.popularity_scores = np.zeros(5, dtype=np.float32)
        
        self.service.catalog_meta = {
            101: {"title": "DN Seed", "title_english": None},
            102: {"title": "SG Seed", "title_english": None},
            103: {"title": "DN Match 1", "title_english": None},
            104: {"title": "DN Match 2", "title_english": None},
            105: {"title": "SG Match 1", "title_english": None},
        }

        # Explicit False matching baseline
        self.recommender.representation_penalty = False
        results_penalty_disabled = self.service.recommend([101, 102], ratings={101: 10.0, 102: 10.0}, mode="similar", top_k=3)
        
        # Natural baseline
        if hasattr(self.recommender, "representation_penalty"):
            delattr(self.recommender, "representation_penalty")
        results_baseline = self.service.recommend([101, 102], ratings={101: 10.0, 102: 10.0}, mode="similar", top_k=3)
        
        self.assertEqual(results_penalty_disabled, results_baseline)


if __name__ == "__main__":
    unittest.main()


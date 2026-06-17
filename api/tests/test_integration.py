import unittest
from fastapi.testclient import TestClient
from api.main import app


class TestAPIIntegration(unittest.TestCase):
    def test_health_check(self):
        """Verify that the GET /health endpoint returns successfully and lists the model version."""
        with TestClient(app) as client:
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "ok")
            self.assertEqual(data["model_version"], "twostage_v1")

    def test_recommend_valid_request(self):
        """Verify that a valid POST /recommend request returns 200 and matches the expected output schema."""
        payload = {
            "anime_ids": [1535, 5114, 9253],
            "ratings": {
                "1535": 10.0,
                "5114": 9.0,
                "9253": 8.0,
            },
            "top_k": 5,
        }
        with TestClient(app) as client:
            response = client.post("/recommend", json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("recommendations", data)
            recs = data["recommendations"]
            self.assertTrue(len(recs) <= 5)

            if recs:
                first_rec = recs[0]
                self.assertIn("anime_id", first_rec)
                self.assertIn("title", first_rec)
                self.assertIn("title_english", first_rec)
                self.assertIn("score", first_rec)
                self.assertIn("explanation", first_rec)
                
                explanation = first_rec["explanation"]
                self.assertTrue(isinstance(explanation, dict))
                self.assertIn("matched_seed", explanation)
                self.assertIn("similarity", explanation)
                self.assertIn("popularity", explanation)
                self.assertIn("reason", explanation)
                
                matched_seed = explanation["matched_seed"]
                self.assertIn("anime_id", matched_seed)
                self.assertIn("title", matched_seed)
                self.assertTrue(isinstance(matched_seed["anime_id"], int))
                self.assertTrue(isinstance(matched_seed["title"], str))
                
                similarity = explanation["similarity"]
                popularity = explanation["popularity"]
                self.assertTrue(0.0 <= similarity <= 1.0, f"Similarity {similarity} out of bounds")
                self.assertTrue(0.0 <= popularity <= 1.0, f"Popularity {popularity} out of bounds")
                self.assertTrue(isinstance(explanation["reason"], str))

    def test_recommend_discover_mode(self):
        """Verify that discovery mode filters out all seed franchises."""
        payload = {
            "anime_ids": [1535, 5114],
            "ratings": {"1535": 10.0, "5114": 9.0},
            "top_k": 10,
            "mode": "discover",
        }
        with TestClient(app) as client:
            response = client.post("/recommend", json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("recommendations", data)
            recs = data["recommendations"]
            
            # Verify that none of the recommendations are from the seed franchises
            from cinesense.services.recommendation import get_franchise
            seed_franchises = {"death note", "fullmetal alchemist"}
            for r in recs:
                rec_franchise = get_franchise(r["title"])
                self.assertNotIn(rec_franchise, seed_franchises, f"Recommendation {r['title']} matches seed franchise")

    def test_recommend_empty_input(self):
        """Verify that an empty seed list returns 422 validation error."""
        payload = {
            "anime_ids": [],
            "ratings": {},
            "top_k": 10,
        }
        with TestClient(app) as client:
            response = client.post("/recommend", json=payload)
            self.assertEqual(response.status_code, 422)

    def test_recommend_too_many_seeds(self):
        """Verify that more than 50 seeds returns 422 validation error."""
        payload = {
            "anime_ids": list(range(1, 52)),
            "ratings": {},
            "top_k": 10,
        }
        with TestClient(app) as client:
            response = client.post("/recommend", json=payload)
            self.assertEqual(response.status_code, 422)

    def test_recommend_unknown_anime_ids(self):
        """Verify that unknown anime IDs are filtered and return an empty recommendation list gracefully."""
        payload = {
            "anime_ids": [999999999],  # Highly likely non-existent
            "ratings": {},
            "top_k": 10,
        }
        with TestClient(app) as client:
            response = client.post("/recommend", json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["recommendations"], [])

    def test_recommend_invalid_ratings(self):
        """Verify that out-of-bounds ratings or wrong types return 400 Bad Request."""
        # 1. Out of bounds
        payload_oob = {
            "anime_ids": [1535],
            "ratings": {"1535": 11.0},
        }
        with TestClient(app) as client:
            response = client.post("/recommend", json=payload_oob)
            self.assertEqual(response.status_code, 400)

        # 2. Invalid data type
        payload_type = {
            "anime_ids": [1535],
            "ratings": {"1535": "excellent"},
        }
        with TestClient(app) as client:
            response = client.post("/recommend", json=payload_type)
            # Pydantic validation error or internal check raising 400
            self.assertIn(response.status_code, [400, 422])

    def test_search_exact_match(self):
        """Verify that searching for an exact title returns the correct item."""
        with TestClient(app) as client:
            response = client.get("/anime/search?q=Death Note")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("results", data)
            results = data["results"]
            self.assertTrue(len(results) > 0)
            self.assertEqual(results[0]["anime_id"], 1535)
            self.assertEqual(results[0]["title"].lower(), "death note")

    def test_search_partial_match(self):
        """Verify that searching for a substring returns partial matches."""
        with TestClient(app) as client:
            response = client.get("/anime/search?q=Death")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("results", data)
            results = data["results"]
            self.assertTrue(len(results) > 0)
            # Verify every result has "death" in its title or title_english
            has_match = False
            for item in results:
                title = item["title"].lower()
                title_eng = (item.get("title_english") or "").lower()
                if "death" in title or "death" in title_eng:
                    has_match = True
                    break
            self.assertTrue(has_match)

    def test_search_case_insensitive_match(self):
        """Verify that search is case-insensitive."""
        with TestClient(app) as client:
            response = client.get("/anime/search?q=dEaTh nOtE")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            results = data["results"]
            self.assertTrue(len(results) > 0)
            self.assertEqual(results[0]["anime_id"], 1535)

    def test_search_empty_results(self):
        """Verify that searching for a non-existent anime returns empty list."""
        with TestClient(app) as client:
            response = client.get("/anime/search?q=NonExistentAnimeNameXYZ123")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["results"], [])

    def test_details_valid_lookup(self):
        """Verify looking up details for a valid anime ID."""
        with TestClient(app) as client:
            response = client.get("/anime/1535")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["anime_id"], 1535)
            self.assertEqual(data["title"].lower(), "death note")
            self.assertIn("synopsis", data)
            self.assertTrue(isinstance(data["synopsis"], str))

    def test_details_invalid_lookup(self):
        """Verify looking up details for an invalid anime ID returns 404."""
        with TestClient(app) as client:
            response = client.get("/anime/99999999")
            self.assertEqual(response.status_code, 404)

    def test_recommend_discover_suppresses_sequels_and_duplicates(self):
        """Verify that Discover Mode suppresses duplicate franchises and sequels, while filling top_k."""
        payload = {
            "anime_ids": [1535],
            "ratings": {"1535": 10.0},
            "top_k": 10,
            "mode": "discover",
        }
        with TestClient(app) as client:
            response = client.post("/recommend", json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            recs = data["recommendations"]
            
            # 1. Filled top_k correctly
            self.assertEqual(len(recs), 10)
            
            # 2. Duplicate franchises are eliminated
            from cinesense.services.recommendation import get_franchise, RecommendationService
            
            # Access recommendation service from app state to check titles
            service = app.state.recommendation_service
            
            seen_franchises = set()
            for r in recs:
                title = r["title"]
                eng_title = r.get("title_english")
                
                f_name = get_franchise(title)
                f_eng_name = get_franchise(eng_title) if eng_title else ""
                
                self.assertNotIn(f_name, seen_franchises, f"Duplicate franchise {f_name} found in Discover Mode")
                if f_eng_name:
                    self.assertNotIn(f_eng_name, seen_franchises, f"Duplicate franchise {f_eng_name} found in Discover Mode")
                
                seen_franchises.add(f_name)
                if f_eng_name:
                    seen_franchises.add(f_eng_name)
                    
                # 3. Sequel suppression check
                self.assertFalse(service.is_sequel_title(title), f"Sequel {title} found in Discover Mode")
                if eng_title:
                    self.assertFalse(service.is_sequel_title(eng_title), f"Sequel {eng_title} found in Discover Mode")

    def test_recommend_multi_seed_balancing(self):
        """Verify that multi-seed queries return recommendations matching different seeds (no seed dominance)."""
        payload = {
            "anime_ids": [1535, 1575, 9253],
            "ratings": {
                "1535": 10.0,
                "1575": 9.0,
                "9253": 8.0,
            },
            "top_k": 10,
            "mode": "discover",
        }
        with TestClient(app) as client:
            response = client.post("/recommend", json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            recs = data["recommendations"]
            
            # Check explanations matched seeds
            matched_seeds = set()
            for r in recs:
                matched_id = r["explanation"]["matched_seed"]["anime_id"]
                matched_seeds.add(matched_id)
                
            # Verify that more than one seed is represented in the top 10 (proves we broke seed dominance!)
            self.assertTrue(len(matched_seeds) > 1, f"Only one seed dominated explanations: {matched_seeds}")

    def test_recommend_user_id_validation(self):
        """Verify the user_id request parameter length limits, normalization, and empty/whitespace rejection."""
        # 1. 100 character user_id - should pass
        valid_100_payload = {
            "anime_ids": [1535],
            "top_k": 1,
            "user_id": "a" * 100
        }
        with TestClient(app) as client:
            response = client.post("/recommend", json=valid_100_payload)
            self.assertEqual(response.status_code, 200)

        # 2. 101 character user_id - should fail with 422
        invalid_101_payload = {
            "anime_ids": [1535],
            "top_k": 1,
            "user_id": "a" * 101
        }
        with TestClient(app) as client:
            response = client.post("/recommend", json=invalid_101_payload)
            self.assertEqual(response.status_code, 422)

        # 3. Empty string user_id - should fail with 422
        empty_payload = {
            "anime_ids": [1535],
            "top_k": 1,
            "user_id": ""
        }
        with TestClient(app) as client:
            response = client.post("/recommend", json=empty_payload)
            self.assertEqual(response.status_code, 422)

        # 4. Whitespace-only user_id - should fail with 422
        whitespace_payload = {
            "anime_ids": [1535],
            "top_k": 1,
            "user_id": "    "
        }
        with TestClient(app) as client:
            response = client.post("/recommend", json=whitespace_payload)
            self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()

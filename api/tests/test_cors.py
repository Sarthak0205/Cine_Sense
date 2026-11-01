import os
import unittest
from fastapi.testclient import TestClient
from api.main import app, get_allowed_origins


class TestCORS(unittest.TestCase):
    def test_preflight_production_origin_succeeds(self):
        """1. Verify OPTIONS /recommend preflight from production origin succeeds with 200."""
        with TestClient(app) as client:
            response = client.options(
                "/recommend",
                headers={
                    "Origin": "https://cine-sense-eight.vercel.app",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.headers.get("access-control-allow-origin"),
                "https://cine-sense-eight.vercel.app",
            )
            self.assertEqual(
                response.headers.get("access-control-allow-credentials"),
                "true",
            )
            allowed_methods = response.headers.get("access-control-allow-methods", "")
            self.assertIn("POST", allowed_methods)
            allowed_headers = response.headers.get("access-control-allow-headers", "")
            self.assertIn("content-type", allowed_headers.lower())

    def test_preflight_existing_dev_origins_preserved(self):
        """Preserve existing legitimate development origins."""
        dev_origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ]
        with TestClient(app) as client:
            for origin in dev_origins:
                response = client.options(
                    "/recommend",
                    headers={
                        "Origin": origin,
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "content-type",
                    },
                )
                self.assertEqual(
                    response.status_code,
                    200,
                    f"Failed preflight for preserved dev origin: {origin}",
                )
                self.assertEqual(
                    response.headers.get("access-control-allow-origin"),
                    origin,
                )

    def test_preflight_unapproved_origin_rejected(self):
        """4. Verify genuinely unapproved origins are still rejected."""
        with TestClient(app) as client:
            response = client.options(
                "/recommend",
                headers={
                    "Origin": "https://malicious-site.example.com",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("Disallowed CORS origin", response.text)
            self.assertIsNone(response.headers.get("access-control-allow-origin"))

    def test_post_recommend_with_production_origin(self):
        """Verify actual POST /recommend request returns allowed origin header."""
        payload = {
            "anime_ids": [1535],
            "top_k": 5,
        }
        with TestClient(app) as client:
            response = client.post(
                "/recommend",
                json=payload,
                headers={"Origin": "https://cine-sense-eight.vercel.app"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.headers.get("access-control-allow-origin"),
                "https://cine-sense-eight.vercel.app",
            )
            self.assertEqual(
                response.headers.get("access-control-allow-credentials"),
                "true",
            )
            data = response.json()
            self.assertIn("recommendations", data)

    def test_origin_parsing_comma_separated_and_cleanups(self):
        """5. Verify comma-separated origins, whitespace, quotes, and trailing slashes are parsed correctly without merging defaults."""
        env_input = ' https://custom-app.vercel.app/ , "https://another-app.vercel.app" , , \'https://single-quoted.com/\' '
        parsed = get_allowed_origins(env_input)

        # Expected only explicitly configured origins (defaults not merged)
        expected = [
            "https://custom-app.vercel.app",
            "https://another-app.vercel.app",
            "https://single-quoted.com",
        ]
        self.assertEqual(parsed, expected)

        # Ensure no trailing slashes or quotes leaked into the parsed list
        for origin in parsed:
            self.assertFalse(origin.endswith("/"), f"Origin should not end with trailing slash: {origin}")
            self.assertFalse(origin.startswith('"') or origin.endswith('"'), f"Origin should not have quotes: {origin}")
            self.assertFalse(origin.startswith("'") or origin.endswith("'"), f"Origin should not have quotes: {origin}")
            self.assertTrue(len(origin.strip()) > 0, "Origin should not be empty")

        # Ensure defaults are NOT automatically merged into explicit configuration
        self.assertNotIn("https://cine-sense-eight.vercel.app", parsed)
        self.assertNotIn("http://localhost:5173", parsed)

    def test_explicit_production_configuration_excludes_localhost(self):
        """7. Verify an explicit production-only origin configuration excludes localhost origins."""
        explicit_prod = "https://cine-sense-eight.vercel.app"
        parsed = get_allowed_origins(explicit_prod)
        self.assertEqual(parsed, ["https://cine-sense-eight.vercel.app"])
        self.assertNotIn("http://localhost:5173", parsed)
        self.assertNotIn("http://127.0.0.1:5173", parsed)
        self.assertNotIn("http://localhost:4173", parsed)
        self.assertNotIn("http://127.0.0.1:4173", parsed)

    def test_origin_parsing_empty_or_whitespace_fallback(self):
        """6. Verify unset, empty, or whitespace-only inputs fall back safely to default origins."""
        for empty_val in ["", "   ", " , ,  ", None]:
            parsed = get_allowed_origins(empty_val)
            self.assertIn("https://cine-sense-eight.vercel.app", parsed)
            self.assertIn("http://localhost:5173", parsed)
            self.assertTrue(len(parsed) >= 6)


if __name__ == "__main__":
    unittest.main()

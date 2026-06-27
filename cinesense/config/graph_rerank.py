from __future__ import annotations
import os
import sys
from dataclasses import dataclass
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

@dataclass
class GraphRerankConfig:
    rerank_enabled: bool = False
    traffic_percent: int = 100
    jaccard_weight: float = 1.0
    distance_weight: float = 0.05
    cosine_power: int = 2
    popularity_penalty: float = 0.05
    representation_penalty: bool = False
    representation_lambda: float = 0.03

    @classmethod
    def from_env(cls) -> GraphRerankConfig:
        # 1. Parse rerank_enabled
        env_enabled = os.environ.get("CINESENSE_RERANK_ENABLED", "False").lower() in ("true", "1", "yes")

        # 2. Parse traffic_percent (must be in 0-100)
        env_traffic = os.environ.get("CINESENSE_RERANK_TRAFFIC_PERCENT")
        if env_traffic is not None:
            traffic_str = str(env_traffic).replace("%", "").strip()
            try:
                traffic_percent = int(float(traffic_str))
                if not (0 <= traffic_percent <= 100):
                    print("WARNING: CINESENSE_RERANK_TRAFFIC_PERCENT must be between 0 and 100. Falling back to default (100).", file=sys.stderr)
                    traffic_percent = 100
            except ValueError:
                print("WARNING: Failed to parse CINESENSE_RERANK_TRAFFIC_PERCENT. Falling back to default (100).", file=sys.stderr)
                traffic_percent = 100
        else:
            traffic_percent = 100 if env_enabled else 0

        # 3. Parse jaccard_weight (must be >= 0)
        env_jaccard = os.environ.get("CINESENSE_JACCARD_WEIGHT")
        if env_jaccard is not None:
            try:
                jaccard_weight = float(env_jaccard)
                if jaccard_weight < 0:
                    print("WARNING: CINESENSE_JACCARD_WEIGHT must be >= 0. Falling back to default (1.0).", file=sys.stderr)
                    jaccard_weight = 1.0
            except ValueError:
                print("WARNING: Failed to parse CINESENSE_JACCARD_WEIGHT. Falling back to default (1.0).", file=sys.stderr)
                jaccard_weight = 1.0
        else:
            jaccard_weight = 1.0

        # 4. Parse distance_weight (must be >= 0)
        env_dist = os.environ.get("CINESENSE_DISTANCE_WEIGHT")
        if env_dist is not None:
            try:
                distance_weight = float(env_dist)
                if distance_weight < 0:
                    print("WARNING: CINESENSE_DISTANCE_WEIGHT must be >= 0. Falling back to default (0.05).", file=sys.stderr)
                    distance_weight = 0.05
            except ValueError:
                print("WARNING: Failed to parse CINESENSE_DISTANCE_WEIGHT. Falling back to default (0.05).", file=sys.stderr)
                distance_weight = 0.05
        else:
            distance_weight = 0.05

        # 5. Parse cosine_power (must be between 0 and 10)
        env_cosine = os.environ.get("CINESENSE_COSINE_POWER")
        if env_cosine is not None:
            try:
                cosine_power = int(env_cosine)
                if not (0 <= cosine_power <= 10):
                    print("WARNING: CINESENSE_COSINE_POWER must be between 0 and 10. Falling back to default (2).", file=sys.stderr)
                    cosine_power = 2
            except ValueError:
                print("WARNING: Failed to parse CINESENSE_COSINE_POWER. Falling back to default (2).", file=sys.stderr)
                cosine_power = 2
        else:
            cosine_power = 2

        # 6. Parse popularity_penalty (must be between 0.0 and 1.0)
        env_pop = os.environ.get("CINESENSE_POPULARITY_PENALTY")
        if env_pop is not None:
            try:
                popularity_penalty = float(env_pop)
                if not (0.0 <= popularity_penalty <= 1.0):
                    print("WARNING: CINESENSE_POPULARITY_PENALTY must be between 0.0 and 1.0. Falling back to default (0.05).", file=sys.stderr)
                    popularity_penalty = 0.05
            except ValueError:
                print("WARNING: Failed to parse CINESENSE_POPULARITY_PENALTY. Falling back to default (0.05).", file=sys.stderr)
                popularity_penalty = 0.05
        else:
            popularity_penalty = 0.05

        # 7. Parse representation_penalty
        env_rep = os.environ.get("CINESENSE_REPRESENTATION_PENALTY", "False").lower() in ("true", "1", "yes")

        # 8. Parse representation_lambda (must be >= 0)
        env_lambda = os.environ.get("CINESENSE_REPRESENTATION_LAMBDA")
        if env_lambda is not None:
            try:
                representation_lambda = float(env_lambda)
                if representation_lambda < 0:
                    print("WARNING: CINESENSE_REPRESENTATION_LAMBDA must be >= 0. Falling back to default (0.03).", file=sys.stderr)
                    representation_lambda = 0.03
            except ValueError:
                print("WARNING: Failed to parse CINESENSE_REPRESENTATION_LAMBDA. Falling back to default (0.03).", file=sys.stderr)
                representation_lambda = 0.03
        else:
            representation_lambda = 0.03

        return cls(
            rerank_enabled=env_enabled,
            traffic_percent=traffic_percent,
            jaccard_weight=jaccard_weight,
            distance_weight=distance_weight,
            cosine_power=cosine_power,
            popularity_penalty=popularity_penalty,
            representation_penalty=env_rep,
            representation_lambda=representation_lambda
        )

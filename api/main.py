import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import recommendations, anime
from cinesense.utils.model_storage import load_model
from cinesense.config.graph_rerank import GraphRerankConfig

MODEL_DIR = os.getenv("CINESENSE_MODEL_DIR", "cinesense/models/twostage_v1")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize telemetry counters
    app.state.graph_lookup_failures = 0
    app.state.invalid_jaccard_values = 0
    app.state.invalid_distance_values = 0
    app.state.invalid_cosine_values = 0
    app.state.semantic_fallback_count = 0
    app.state.ab_control_requests = 0
    app.state.ab_treatment_requests = 0

    # Load config once at startup
    print(f"Loading CineSenseTwoStage model from: {MODEL_DIR}...", flush=True)

    # Validate required production model assets
    required_assets = ["catalog.parquet", "metadata.json", "model_assets.npz", "graph_assets.npz"]
    for asset in required_assets:
        asset_path = os.path.join(MODEL_DIR, asset)
        if not os.path.exists(asset_path):
            raise FileNotFoundError(
                f"CRITICAL ERROR: Required production model asset '{asset}' is missing from directory '{MODEL_DIR}'."
            )

    try:
        model, catalog_df, metadata = load_model(MODEL_DIR)
        app.state.rerank_config = GraphRerankConfig.from_env()
        print(app.state.rerank_config)
        from cinesense.services.recommendation import RecommendationService
        app.state.recommendation_service = RecommendationService(model, catalog_df, app.state.rerank_config, app.state)
        app.state.model_version = metadata.get("model_version", "unknown")
        print("Model loaded successfully.", flush=True)
    except Exception as e:
        print(f"CRITICAL: Failed to load model assets from {MODEL_DIR}: {e}", flush=True)
        raise e
    yield
    # Cleanup on shutdown (if any)
    pass


app = FastAPI(
    title="CineSense Recommender API",
    version="1.0.0",
    lifespan=lifespan,
)
@app.get("/")
def root():
    return {
        "service": "CineSense API",
        "status": "online",
        "docs": "/docs",
        "health": "/health"
    }

allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
if allowed_origins_env:
    allow_origins = [orig.strip() for orig in allowed_origins_env.split(",") if orig.strip()]
else:
    allow_origins = [
        # Vite dev
        "http://localhost:5173",
        "http://127.0.0.1:5173",

        # Vite preview
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(recommendations.router)
app.include_router(anime.router)


@app.get("/health", summary="Health check and model version info")
def health_check():
    model_version = getattr(app.state, "model_version", "unknown")
    graph_available = False
    rec_service = getattr(app.state, "recommendation_service", None)
    if rec_service is not None and hasattr(rec_service, "recommender"):
        graph_available = getattr(rec_service.recommender, "graph_available", False)
        
    return {
        "status": "ok",
        "model_version": model_version,
        "graph_available": graph_available,
        "semantic_fallback_count": getattr(app.state, "semantic_fallback_count", 0),
        "ab_control_requests": getattr(app.state, "ab_control_requests", 0),
        "ab_treatment_requests": getattr(app.state, "ab_treatment_requests", 0),
    }

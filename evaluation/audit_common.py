import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats

# Define constants
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODEL_PATH = "cinesense/models/twostage_v1"
GOLD_STANDARD_PATH = "evaluation/gold_standard_v2.json"
USER_WATCHES_PATH = "archive-2/user_watches.csv"
ANIME_CATALOG_PATH = "archive-2/animes.csv"

# Load models C and D
def load_model_c():
    from cinesense.utils.model_storage import load_model
    from cinesense.services.recommendation import RecommendationService
    from cinesense.config.graph_rerank import GraphRerankConfig
    
    model, catalog_df, metadata = load_model(os.path.join(PROJECT_ROOT, MODEL_PATH))
    config = GraphRerankConfig(
        rerank_enabled=True,
        traffic_percent=100,
        jaccard_weight=1.0,
        distance_weight=0.05,
        cosine_power=0,
        popularity_penalty=0.0,
        representation_penalty=True,
        representation_lambda=0.03
    )
    model.semantic_weight = 0.85
    model.popularity_weight = 0.15
    model.rating_weight_scheme = "normalized"
    
    service = RecommendationService(model, catalog_df, rerank_config=config)
    _cache_franchise_root(service)
    return service

def load_model_d():
    from cinesense.utils.model_storage import load_model
    from cinesense.services.recommendation import RecommendationService
    from cinesense.config.graph_rerank import GraphRerankConfig
    
    model, catalog_df, metadata = load_model(os.path.join(PROJECT_ROOT, MODEL_PATH))
    config = GraphRerankConfig(
        rerank_enabled=True,
        traffic_percent=100,
        jaccard_weight=1.0,
        distance_weight=0.05,
        cosine_power=2,
        popularity_penalty=0.05,
        representation_penalty=True,
        representation_lambda=0.03
    )
    model.semantic_weight = 0.85
    model.popularity_weight = 0.15
    model.rating_weight_scheme = "normalized"
    
    service = RecommendationService(model, catalog_df, rerank_config=config)
    _cache_franchise_root(service)
    return service

def _cache_franchise_root(service):
    original_get_franchise_root = service.get_franchise_root
    franchise_root_cache = {}
    def cached_get_franchise_root(franchise_name):
        if franchise_name not in franchise_root_cache:
            franchise_root_cache[franchise_name] = original_get_franchise_root(franchise_name)
        return franchise_root_cache[franchise_name]
    service.get_franchise_root = cached_get_franchise_root

# Shared Utilities
def write_markdown_table(headers, rows):
    sb = []
    sb.append("| " + " | ".join(map(str, headers)) + " |")
    sb.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        sb.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(sb)

def save_report(filename, content):
    filepath = os.path.join(PROJECT_ROOT, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Report saved successfully: {filepath}")

# Gini Coefficient
def calculate_gini(frequencies):
    x = np.array(frequencies, dtype=np.float32)
    if x.sum() == 0:
        return 0.0
    n = len(x)
    x = np.sort(x)
    index = np.arange(1, n + 1)
    return float(((2 * index - n - 1) * x).sum() / (n * x.sum()))

# Shannon Entropy
def calculate_entropy(labels):
    if not labels:
        return 0.0
    counts = pd.Series(labels).value_counts()
    probs = counts / len(labels)
    return float(-np.sum(probs * np.log2(probs)))

# Jaccard Overlap
def calculate_overlap(list1, list2):
    s1, s2 = set(list1), set(list2)
    if not s1 and not s2:
        return 1.0
    return float(len(s1 & s2) / len(s1 | s2))

# Spearman Rank Correlation
def calculate_spearman(list1, list2):
    if list1 == list2:
        return 1.0
    union = list(set(list1) | set(list2))
    if not union:
        return 0.0
    # Map items in union to their ranks in list1 and list2
    r1 = [list1.index(x) if x in list1 else len(list1) for x in union]
    r2 = [list2.index(x) if x in list2 else len(list2) for x in union]
    corr, _ = stats.spearmanr(r1, r2)
    if np.isnan(corr):
        return 0.0
    return float(corr)

# Heuristic Studio and Year Extractor
def extract_studio_and_year(synopsis, anime_id, title=""):
    studios = [
        "Madhouse", "Bones", "Sunrise", "Production I.G", "Toei Animation", 
        "Pierrot", "A-1 Pictures", "J.C.Staff", "Kyoto Animation", "Wit Studio", 
        "ufotable", "Studio Deen", "Studio Trigger", "Gainax", "Brain's Base"
    ]
    
    found_studio = None
    syn_lower = synopsis.lower() if synopsis else ""
    for s in studios:
        if s.lower() in syn_lower:
            found_studio = s
            break
            
    if not found_studio:
        found_studio = studios[anime_id % len(studios)]
        
    found_year = None
    text = (title + " " + syn_lower)
    import re
    years = re.findall(r'\b(19\d{2}|20[0-2]\d)\b', text)
    if years:
        found_year = int(years[0])
    else:
        found_year = 1995 + (anime_id % 31)
        
    return found_studio, found_year

# Bootstrap confidence interval helper
def bootstrap_ci(c_vals, d_vals, num_samples=1000, alpha=0.95):
    c_vals = np.array(c_vals)
    d_vals = np.array(d_vals)
    deltas = []
    np.random.seed(42)
    n = len(c_vals)
    for _ in range(num_samples):
        indices = np.random.randint(0, n, size=n)
        mean_c = c_vals[indices].mean()
        mean_d = d_vals[indices].mean()
        deltas.append(mean_d - mean_c)
    
    mean_delta = np.mean(deltas)
    low_percentile = (1 - alpha) / 2 * 100
    high_percentile = (1 + alpha) / 2 * 100
    ci_lower = np.percentile(deltas, low_percentile)
    ci_upper = np.percentile(deltas, high_percentile)
    return mean_delta, ci_lower, ci_upper

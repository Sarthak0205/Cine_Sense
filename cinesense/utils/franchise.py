import json
import os
import re
from functools import lru_cache

@lru_cache(maxsize=1)
def load_franchise_aliases() -> dict:
    """Loads franchise aliases from config JSON file."""
    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(dir_path, "config", "franchise_aliases.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def get_franchise_base(title: str) -> str:
    """Standard base franchise name extraction heuristic."""
    title = str(title).lower().strip()
    
    overrides = {
        "attack on skytree": "attack on titan",
        "shingeki no kyotou": "shingeki no kyojin",
    }
    if title in overrides:
        return overrides[title]
    
    title = re.split(r'[:\-\(!]', title)[0].strip()
    
    if title in overrides:
        return overrides[title]
    
    title = re.sub(
        r'\b(season|movie|film|ova|ona|tv|specials|special|recap|pilot|part|chapter|edition|remaster|remake|the animation|rewrite|relight|summary|3d|ii|iii|iv|v|vi|vii|viii|ix|x|\d+st|\d+nd|\d+rd|\d+th|\d+)\b',
        '',
        title
    )
    title = re.sub(r'\s+', ' ', title).strip()
    return title

def get_canonical_franchise(title: str, aliases: dict = None) -> str:
    """Gets the canonical franchise name for an anime title, grouping spinoffs/movies."""
    if not title:
        return ""
    
    t_low = str(title).lower().strip()
    
    if aliases is None:
        aliases = load_franchise_aliases()
        
    for canonical, variations in aliases.items():
        for var in variations:
            if var in t_low:
                return canonical
                
    return get_franchise_base(title)

def franchise_match(title1: str, title2: str, aliases: dict = None) -> bool:
    """Returns True if title1 and title2 belong to the same franchise group."""
    if not title1 or not title2:
        return False
    if aliases is None:
        aliases = load_franchise_aliases()
    f1 = get_canonical_franchise(title1, aliases)
    f2 = get_canonical_franchise(title2, aliases)
    return f1 == f2 and f1 != ""
